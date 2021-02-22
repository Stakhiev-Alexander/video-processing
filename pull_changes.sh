#!/bin/bash


git pull

cd ./src/nets/Fast-SRGAN
git pull
cd ../../../

cd ./src/nets/RIFE
git pull
cd ../../../

cd ./src/nets/DeepLab
git pull
cd ../../../

cd ./src/nets/flownet
git pull
cd ../../../


exit 0
