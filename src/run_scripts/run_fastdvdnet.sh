#!/bin/bash

while [ "$1" != "" ]; do
    case $1 in
        -i | --input_path )     shift
                                input_path="$1"
                                ;;
        -s | --sigma )          shift
                                sigma="$1"
                                ;;
        -o | --output_path )    shift
                                output_path="$1"
                                ;;
        * )                     echo "invalid option $1"
                                exit 1
    esac
    shift
done

if [ -z "$input_path" ]; then
    echo "input_path unset"
    exit 1
fi

if [ -z "$sigma" ]; then
    echo "sigma unset"
    exit 1
fi

if [ -z "$output_path" ]; then
    echo "output_path unset"
    exit 1
fi


echo "input_path: $input_path";
echo "sigma: $sigma";
echo "output_path: $output_path";


conda activate fastdvdnet

python3 ./src/nets/FastDVDNet/test_fastdvdnet.py --test_path $input_path --noise_sigma $sigma --save_path $output_path --model_file ./src/nets/FastDVDNet/model.pth

exit 0 
