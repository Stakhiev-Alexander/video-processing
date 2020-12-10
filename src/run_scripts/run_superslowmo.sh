#!/bin/bash

while [ "$1" != "" ]; do
    case $1 in
        -i | --input_path )     shift
                                input_path="$1"
                                ;;
        -f | --factor )         shift
                                factor="$1"
                                ;;

        --batch_size )          shift
                                batch_size="$1"
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

if [ -z "$factor" ]; then
    echo "factor unset"
    exit 1
fi

if [ -z "$output_path" ]; then
    echo "output_path unset"
    exit 1
fi


echo "input_path: $input_path";
echo "factor: $factor";
echo "batch_size: $batch_size";
echo "output_path: $output_path";


conda activate superslomo

python ./src/nets/superslomo/video_to_slomo.py --checkpoint ./src/nets/superslomo/SuperSloMo.ckpt --input_path input_path --sf factor --batch_size batch_size --output output_path

exit 0 
