#!/bin/bash


git pull

cd ./nets/Fast-SRGAN
git pull
cd ../..

cd ./nets/RIFE
git pull
cd ../..

cd ./nets/DeepLab
git pull
cd ../..

cd ./nets/flownet
git pull
cd ../..


exit 0
