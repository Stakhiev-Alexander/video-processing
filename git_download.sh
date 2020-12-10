#!/bin/bash

function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

mkdir -p ./src/nets

git clone https://github.com/baowenbo/DAIN ./src/nets/DAIN
git clone https://github.com/Stakhiev-Alexander/Fast-SRGAN ./src/nets/Fast-SRGAN

#git clone https://github.com/ZurMaD/fastdvdnet ./src/nets/FastDVDNet
#git clone https://github.com/Stakhiev-Alexander/Colab-Super-SloMo.git ./src/nets/superslomo

# cd ./src/superslomo
# gdrive_download 1IvobLDbRiBgZr3ryCRrWL8xDbMZ-KnpF SuperSloMo.ckpt 

rm ./src/nets/Fast-SRGAN/models/generator.h5
gdrive_download 15iVCa-GNYbakU_9yINMrDjWAzVo3HVRf ./src/nets/Fast-SRGAN/models/generator.h5

# git clone https://github.com/meisamrf/ivhc-estimator ./src/IVHC
# mv ./src/IVHC/Python/libs/ivhc.cpython-36m-x86_64-linux-gnu.so ./ivhc.cpython-36m-x86_64-linux-gnu.so 

exit 0 
