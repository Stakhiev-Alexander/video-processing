#!/bin/bash

while [ "$1" != "" ]; do
    case $1 in
        -i | --input_path )     shift
                                input_path="$1"
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

if [ -z "$output_path" ]; then
    echo "output_path unset"
    exit 1
fi


echo "input_path: $input_path";
echo "output_path: $output_path";

source ~/anaconda3/etc/profile.d/conda.sh
conda activate dl_hifill

python3 ./src/nets/infer_dl_hifill.py --image_dir $input_path --output_dir $output_path

exit 0 