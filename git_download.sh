#!/bin/bash

function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

##
##  Nets download
##
mkdir -p ./src/nets

git clone https://github.com/Stakhiev-Alexander/Fast-SRGAN ./src/nets/Fast-SRGAN
git clone https://github.com/Egazaga/RAFT ./src/nets/RAFT

git clone https://github.com/Egazaga/pytorch-deeplab-xception.git ./src/nets/DeepLab
cd ./src/nets/DeepLab
git checkout infer-triple-input
cd ../../../

git clone https://github.com/Egazaga/flownet2-pytorch ./src/nets/flownet

##
##  Weights download
##

# rm ./src/nets/Fast-SRGAN/models/generator.h5
# gdrive_download 15iVCa-GNYbakU_9yINMrDjWAzVo3HVRf ./src/nets/Fast-SRGAN/models/generator.h5

# git clone https://github.com/meisamrf/ivhc-estimator ./src/net/IVHC
# mv ./src/IVHC/Python/libs/ivhc.cpython-36m-x86_64-linux-gnu.so ./ivhc.cpython-36m-x86_64-linux-gnu.so

#cd ./src/nets/RAFT
#download_models.sh
#cd ../../..

# gdrive_download 1OJgJRlRrJG9HV28RTJlGV9VsCfRhYabD ./src/nets/DeepLab

#cd ./src/nets/flownet
#install.sh
#cd ../../..

exit 0 
