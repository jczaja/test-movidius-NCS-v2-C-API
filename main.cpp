#include <mvnc.h>
#include <opencv2/opencv.hpp>
#include <time.h>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <streambuf>
#include <sstream>
#include <memory>
#include <x86intrin.h>
#include <sys/types.h>
#include <unistd.h>
#include "gflags/gflags.h"

DEFINE_int32(num_reps, 1,
    "Number of repetitions of convolutions to be performed");

const unsigned int net_data_width = 224;
const unsigned int net_data_height = 224;
const unsigned int net_data_channels = 3;
const cv::Scalar   net_mean(0.40787054*255.0, 0.45752458*255.0, 0.48109378*255.0);

struct platform_info
{
    long num_logical_processors;
    long num_physical_processors_per_socket;
    long num_hw_threads_per_socket;
    unsigned int num_ht_threads; 
    unsigned int num_total_phys_cores;
    unsigned long long tsc;
    unsigned long long max_bandwidth; 
};

class nn_hardware_platform
{
    public:
        nn_hardware_platform() : m_num_logical_processors(0), m_num_physical_processors_per_socket(0), m_num_hw_threads_per_socket(0) ,m_num_ht_threads(1), m_num_total_phys_cores(1), m_tsc(0), m_fmaspc(0), m_max_bandwidth(0)
        {
#ifdef __linux__
            m_num_logical_processors = sysconf(_SC_NPROCESSORS_ONLN);
        
            m_num_physical_processors_per_socket = 0;

            std::ifstream ifs;
            ifs.open("/proc/cpuinfo"); 

            // If there is no /proc/cpuinfo fallback to default scheduler
            if(ifs.good() == false) {
                m_num_physical_processors_per_socket = m_num_logical_processors;
                assert(0);  // No cpuinfo? investigate that
                return;   
            }
            std::string cpuinfo_content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
            std::stringstream cpuinfo_stream(cpuinfo_content);
            std::string cpuinfo_line;
            std::string cpu_name;
            while(std::getline(cpuinfo_stream,cpuinfo_line,'\n')){
                if((m_num_physical_processors_per_socket == 0) && (cpuinfo_line.find("cpu cores") != std::string::npos)) {
                    // convert std::string into number eg. skip colon and after it in the same line  should be number of physical cores per socket
                    std::stringstream( cpuinfo_line.substr(cpuinfo_line.find(":") + 1) ) >> m_num_physical_processors_per_socket; 
                }
                if(cpuinfo_line.find("siblings") != std::string::npos) {
                    // convert std::string into number eg. skip colon and after it in the same line  should be number of HW threads per socket
                    std::stringstream( cpuinfo_line.substr(cpuinfo_line.find(":") + 1) ) >> m_num_hw_threads_per_socket; 
                }

                if(cpuinfo_line.find("model") != std::string::npos) {
                    cpu_name = cpuinfo_line;
                    // convert std::string into number eg. skip colon and after it in the same line  should be number of HW threads per socket
                    float ghz_tsc = 0.0f;
                    std::stringstream( cpuinfo_line.substr(cpuinfo_line.find("@") + 1) ) >> ghz_tsc; 
                    m_tsc = static_cast<unsigned long long>(ghz_tsc*1000000000.0f);
                    
                    // Maximal bandwidth is Xeon 68GB/s , Brix 25.8GB/s
                    if(cpuinfo_line.find("Xeon") != std::string::npos) {
                      m_max_bandwidth = 68000;  //68 GB/s      -- XEONE5
                    } 
                    
                    if(cpuinfo_line.find("i7-4770R") != std::string::npos) {
                      m_max_bandwidth = 25800;  //25.68 GB/s      -- BRIX
                    } 
                }
                
                // determine instruction set (AVX, AVX2, AVX512)
                if(m_fmaspc == 0) {
                    if (cpuinfo_line.find(" avx") != std::string::npos) {
                      m_fmaspc = 8;   // On AVX instruction set we have one FMA unit , width of registers is 256bits, so we can do 8 muls and adds on floats per cycle
                      if (cpuinfo_line.find(" avx2") != std::string::npos) {
                        m_fmaspc = 16;   // With AVX2 instruction set we have two FMA unit , width of registers is 256bits, so we can do 16 muls and adds on floats per cycle
                      }
                      if (cpuinfo_line.find(" avx512") != std::string::npos) {
                        m_fmaspc = 32;   // With AVX512 instruction set we have two FMA unit , width of registers is 512bits, so we can do 32 muls and adds on floats per cycle
                      }
                  }
                }
            }
            // If no FMA ops / cycle was given/found then raise a concern
            if(m_fmaspc == 0) {
              throw std::string("No AVX instruction set found. Please use \"--fmaspc\" to specify\n");
            }

            // There is cpuinfo, but parsing did not get quite right? Investigate it
            assert( m_num_physical_processors_per_socket > 0);
            assert( m_num_hw_threads_per_socket > 0);

            // Calculate how many threads can be run on single cpu core , in case of lack of hw info attributes assume 1
            m_num_ht_threads =  m_num_physical_processors_per_socket != 0 ? m_num_hw_threads_per_socket/ m_num_physical_processors_per_socket : 1;
            // calculate total number of physical cores eg. how many full Hw threads we can run in parallel
            m_num_total_phys_cores = m_num_hw_threads_per_socket != 0 ? m_num_logical_processors / m_num_hw_threads_per_socket * m_num_physical_processors_per_socket : 1;

            std::cout << "Platform:" << std::endl << "  " << cpu_name << std::endl 
                      << "  number of physical cores: " << m_num_total_phys_cores << std::endl; 
            ifs.close(); 

#endif
        }
    // Function computing percentage of theretical efficiency of HW capabilities
    float compute_theoretical_efficiency(unsigned long long start_time, unsigned long long end_time, unsigned long long num_fmas)
    {
      // Num theoretical operations
      // Time given is there
      return 100.0*num_fmas/((float)(m_num_total_phys_cores*m_fmaspc))/((float)(end_time - start_time));
    }

    void get_platform_info(platform_info& pi)
    {
       pi.num_logical_processors = m_num_logical_processors; 
       pi.num_physical_processors_per_socket = m_num_physical_processors_per_socket; 
       pi.num_hw_threads_per_socket = m_num_hw_threads_per_socket;
       pi.num_ht_threads = m_num_ht_threads;
       pi.num_total_phys_cores = m_num_total_phys_cores;
       pi.tsc = m_tsc;
       pi.max_bandwidth = m_max_bandwidth;
    }
    private:
        long m_num_logical_processors;
        long m_num_physical_processors_per_socket;
        long m_num_hw_threads_per_socket;
        unsigned int m_num_ht_threads;
        unsigned int m_num_total_phys_cores;
        unsigned long long m_tsc;
        short int m_fmaspc;
        unsigned long long m_max_bandwidth;
};
// TODO:
// - user params
/*
void prepareTensor(std::unique_ptr<unsigned char[]>& input, std::string& imageName,unsigned int* inputLength)
{
  // load an image using OpenCV
  cv::Mat imagefp32 = cv::imread(imageName, -1);
  if (imagefp32.empty())
    throw std::string("Error reading image: ") + imageName;

  // Convert to expected format
  cv::Mat samplefp32;
  if (imagefp32.channels() == 4 && net_data_channels == 3)
    cv::cvtColor(imagefp32, samplefp32, cv::COLOR_BGRA2BGR);
  else if (imagefp32.channels() == 1 && net_data_channels == 3)
    cv::cvtColor(imagefp32, samplefp32, cv::COLOR_GRAY2BGR);
  else
    samplefp32 = imagefp32;
  
  // Resize input image to expected geometry
  cv::Size input_geometry(net_data_width, net_data_height);

  cv::Mat samplefp32_resized;
  if (samplefp32.size() != input_geometry)
    cv::resize(samplefp32, samplefp32_resized, input_geometry);
  else
    samplefp32_resized = samplefp32;

  // Convert to float32
  cv::Mat samplefp32_float;
  samplefp32_resized.convertTo(samplefp32_float, CV_32FC3);

  // Mean subtract
  cv::Mat sample_fp32_normalized;
  cv::Mat mean = cv::Mat(input_geometry, CV_32FC3, net_mean);
  cv::subtract(samplefp32_float, mean, sample_fp32_normalized);

  // Separate channels (caffe format: NCHW)
  std::vector<cv::Mat> input_channels(net_data_channels);
  cv::split(sample_fp32_normalized, input_channels);

  // convert image data into float16
  input.reset(new unsigned char[sizeof(short)*net_data_width*net_data_height*net_data_channels]);
  floattofp16(input.get(), reinterpret_cast<float*>(sample_fp32_normalized.data),
        net_data_width*net_data_height*net_data_channels);
 
  *inputLength = sizeof(short)*net_data_width*net_data_height*net_data_channels;
}


void printPredictions(void* outputTensor,unsigned int outputLength)
{
	unsigned int net_output_width = outputLength/sizeof(short);

	std::vector<float> predictions(net_output_width);
  fp16tofloat(&predictions[0],reinterpret_cast<unsigned char*>(outputTensor),net_output_width);
	int top1_index= -1;
	float top1_result = -1.0;

	// find top1 results	
	for(unsigned int i = 0; i<net_output_width;++i) {
		if(predictions[i] > top1_result) {
			top1_result = predictions[i];
			top1_index = i;
		}
		std::cout << predictions[i] << std::endl;
	}

	// Print top-1 result (class name , prob)
	std::ifstream synset_words("./synset_words.txt");
  std::string top1_class;
	for (int i=0; i<=top1_index; ++i) {
		std::getline(synset_words,top1_class);	
	}
	std::cout << "top-1: " << top1_result << " (" << top1_class << ")" << std::endl;
}


void loadGraphFromFile(std::unique_ptr<char[]>& graphFile, const std::string& graphFileName, unsigned int* graphSize)
{
  std::ifstream ifs;
  ifs.open(graphFileName, std::ifstream::binary);

  if (ifs.good() == false) {
    throw std::string("Error: Unable to open graph file: ") + graphFileName;
  }

  // Get size of file
  ifs.seekg(0, ifs.end);
  *graphSize = ifs.tellg();
  ifs.seekg(0, ifs.beg);


  graphFile.reset(new char[*graphSize]);

  ifs.read(graphFile.get(),*graphSize);

  // TODO: check if whole file was read
  
  ifs.close();
}

void printProfiling(float* dataPtr, unsigned int numEntries)
{
	std::cout << "Performance profiling:" << std::endl;
	float totalTime = 0.0f;
	for(int i=0; i<numEntries; ++i) {
		std::cout << "	" << std::to_string(dataPtr[i]) << " ms"<<std::endl; 			
		totalTime += dataPtr[i];
	}
	std::cout << "Total time: " << std::to_string(totalTime) << " ms"<< std::endl; 			
}
*/
int main(int argc, char** argv) {

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  gflags::SetUsageMessage("Perform NCS classification.\n"
        "Usage:\n"
        "    test_ncs [FLAGS]\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  void * graphHandle = nullptr;
	void* dev_handle = 0;
	std::vector<std::string> ncs_names;
  const std::string graphFileName("myGoogleNet-shave12");
  nn_hardware_platform machine;
  platform_info pi;
  machine.get_platform_info(pi);
  int exit_code = 0;
  ncStatus_t ret = NC_OK;
/*
  try {
    
    if(argc != 2 ) {
      throw std::string("ERROR: Wrong syntax. Valid syntax:\n \
               test-ncs <name of image to process> \n \
                ");
    }
    std::string imageFileName(argv[1]);
		// Set verbose mode for a work with NCS device
		int verbosity = 2;	// Be more verbose
	  ret = mvncSetGlobalOption(0, reinterpret_cast<const void*>(&verbosity),sizeof(int));
    if(ret != MVNC_OK) {
      throw std::string("Error: Could not set global option of loggin ") + std::string(" Error code: " + std::to_string(ret));
    }

    char tmpncsname[200]; // How to determine max size automatically
    int index = 0;  // Index of device to query for
    while(ret == MVNC_OK) {
      ret = mvncGetDeviceName(index++,tmpncsname,200); // hardcoded max name size 
      if (ret == MVNC_OK) {
        ncs_names.push_back(tmpncsname); 
        std::cout << "Found NCS: " << tmpncsname << std::endl;
      }
    }

    // If not devices present the exit
    if (ncs_names.size() == 0) {
      throw std::string("Error: No Intel Movidius identified in a system!\n");
    }

    // Using first device
    // TODO: run workload on many devices
    ret = mvncOpenDevice(ncs_names[0].c_str(), &dev_handle);
    if(ret != MVNC_OK) {
      throw std::string("Error: Could not open NCS device: ") + ncs_names[0] ;
    }


    // Allocate graph
    unsigned int graphSize = 0;
    std::unique_ptr<char[]> graphFile;
    loadGraphFromFile(graphFile, graphFileName, &graphSize);

    ret = mvncAllocateGraph(dev_handle,&graphHandle,static_cast<void*>(graphFile.get()),graphSize);
    if (ret != MVNC_OK) {
      throw std::string("Error: Graph allocation on NCS failed!");
    }
    

    int dontBlockValue = 0;
    ret = mvncSetGraphOption(graphHandle,
        MVNC_DONTBLOCK, &dontBlockValue, sizeof(int));
    if (ret != MVNC_OK)
    {
      throw std::string("Error: Setting MVNC_DONTBLOCK graph option failed!");
    }

    // Loading tensor, tensor is of a HalfFloat data type 
    std::unique_ptr<unsigned char[]> tensor;
    unsigned int inputLength;
    prepareTensor(tensor, imageFileName, &inputLength);
    auto t1 = __rdtsc();
    void* outputTensor;
    unsigned int outputLength;
    for(int i=0; i< FLAGS_num_reps; ++i) {
    ret = mvncLoadTensor(graphHandle, tensor.get(), inputLength,
                         nullptr);  // TODO: What are user params??? 
    if (ret != MVNC_OK) {
      throw std::string("Error: Loading Tensor failed!");
    }

		void* userParam;
    // This function normally blocks till results are available
    ret = mvncGetResult(graphHandle,&outputTensor, &outputLength,&userParam);
		}
    auto t2 = __rdtsc();

    std::cout << "---> NCS execution including memory transfer takes " << ((t2 - t1)/(float)FLAGS_num_reps) << " RDTSC cycles time[ms]: " << (t2 -t1)*1000.0f/((float)pi.tsc*FLAGS_num_reps);
    
    if (ret != MVNC_OK) {
      throw std::string("Error: Getting results from NCS failed!");
    }
    printPredictions(outputTensor, outputLength);

		// print some performance info
		unsigned int dataLength = 0;
		float * data_ptr = nullptr;
		ret = mvncGetGraphOption(graphHandle,MVNC_TIME_TAKEN,reinterpret_cast<void**>(&data_ptr),&dataLength);
    if (ret != MVNC_OK) {
      throw std::string("Error: Getting Time taken results failed!");
    }

		// implement printing of profiling info
		printProfiling(data_ptr, dataLength/sizeof(float));
  }
  catch (std::string err) {
    std::cout << err << std::endl;
    exit_code = -1;
  }
		*/

/*
  // Cleaning up
  ret = mvncDeallocateGraph(graphHandle);
  if (ret != MVNC_OK) {
    std::cerr << "Error: Deallocation of Graph failed!" << std::endl;
  }

	// Close Device
	if (dev_handle != 0) {
		ret = mvncCloseDevice(dev_handle);
		if (ret != MVNC_OK) {
			std::cerr << "Error: Closing of device: "<< ncs_names[0] <<"failed!" << std::endl;
		}
	}
*/
  return exit_code;
}
