#!/bin/bash

function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

git clone https://github.com/ZurMaD/fastdvdnet ./src/FastDVDNet
git clone https://github.com/baowenbo/DAIN ./src/DAIN
git clone https://github.com/Stakhiev-Alexander/Colab-Super-SloMo.git ./src/superslomo

cd ./src/superslomo
gdrive_download 1IvobLDbRiBgZr3ryCRrWL8xDbMZ-KnpF SuperSloMo.ckpt


# git clone https://github.com/meisamrf/ivhc-estimator ./src/IVHC
# mv ./src/IVHC/Python/libs/ivhc.cpython-36m-x86_64-linux-gnu.so ./ivhc.cpython-36m-x86_64-linux-gnu.so 

exit 0 
