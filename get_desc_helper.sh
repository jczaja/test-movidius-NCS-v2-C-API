#!/bin/env sh


echo "Downloading..."
wget -c http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz

echo "Unzipping..."
tar -xzf caffe_ilsvrc12.tar.gz synset_words.txt

echo "Cleaning up.."
rm caffe_ilsvrc12.tar.gz 


