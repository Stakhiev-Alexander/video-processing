#!/bin/bash

while [ "$1" != "" ]; do
  case $1 in
  -i | --input_path)
    shift
    input_path="$1"
    ;;
  -o | --output_path)
    shift
    output_path="$1"
    ;;
  -f | --downscale-factor)
    shift
    downscale_factor="$1"
    ;;
  *)
    echo "invalid option $1"
    exit 1
    ;;
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

if [ -z "$downscale_factor" ]; then
  echo "downscale_factor unset"
  exit 1
fi

echo "input_path: $input_path"
echo "output_path: $output_path"
echo "downscale_factor: $downscale_factor"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate artefacts

python3 nets/infer_artefacts.py -i $input_path -o $output_path --downscale-factor $downscale_factor

exit 0
