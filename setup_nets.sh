#!/bin/bash

function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

##
##  Nets download
##
mkdir -p ./nets

git clone https://github.com/Stakhiev-Alexander/Fast-SRGAN ./nets/Fast-SRGAN
git clone https://github.com/Egazaga/arXiv2020-RIFE ./nets/RIFE

git clone https://github.com/Egazaga/pytorch-deeplab-xception.git ./nets/DeepLab
cd ./nets/DeepLab
git checkout infer-triple-input
cd ../..

git clone https://github.com/Egazaga/flownet2-pytorch ./nets/flownet

##
##  Weights download
##

if [ ! -e ./nets/Fast-SRGAN/models/generator.h5 ]; then
  mkdir -p ./nets/Fast-SRGAN/models
  gdrive_download 15iVCa-GNYbakU_9yINMrDjWAzVo3HVRf ./nets/Fast-SRGAN/models/generator.h5
fi

git clone https://github.com/meisamrf/ivhc-estimator ./nets/IVHC
cp ./nets/IVHC/Python/libs/ivhc.cpython-36m-x86_64-linux-gnu.so ./nets/IVHC/ivhc.cpython-36m-x86_64-linux-gnu.so

if [ ! -e ./nets/RIFE/train_log/unet.pkl ]; then
  gdrive_download 1wsQIhHZ3Eg4_AfCXItFKqqyDMB4NS0Yd ./nets/RIFE/model.zip
  cd ./nets/RIFE/
  unzip model.zip
  rm model.zip
  cd ../..
fi

if [ ! -e ./nets/DeepLab/DLv3+torch.pth.tar ]; then
  gdrive_download 1OJgJRlRrJG9HV28RTJlGV9VsCfRhYabD ./nets/DeepLab/DLv3+torch.pth.tar
fi

if [ ! -e ./nets/flownet/FlowNet2_checkpoint.pth.tar ]; then
  gdrive_download 1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da ./nets/flownet/FlowNet2_checkpoint.pth.tar
fi

if [ ! -e ./nets/flownet/FlowNet2-CSS_checkpoint.pth.tar ]; then
  gdrive_download 157zuzVf4YMN6ABAQgZc8rRmR5cgWzSu8  ./nets/flownet/FlowNet2-CSS_checkpoint.pth.tar 
fi

if [ ! -e ./nets/flownet/FlowNet2-CS_checkpoint.pth.tar ]; then
  gdrive_download 1iBJ1_o7PloaINpa8m7u_7TsLCX0Dt_jS  ./nets/flownet/FlowNet2-CS_checkpoint.pth.tar
fi

#cd ./nets/flownet
#source install.sh
#cd ../..

##
##  Make scripts executable
##

for filename in run_scripts/*.sh; do
    [ -e "$filename" ] || continue
    chmod +x "$filename"
done
