#!/bin/sh
#TARGET=$1
#ARCH=$2
#MODEL=$3

if [ -z "$1" ]
  then
    gpu="0"
  else
    gpu="${1}"
fi

#TARGET=market1501
#ARCH=densenet
#MODEL=logs/xxxx/xxxx-MEB-Net/checkpoint.pt.pth
#
#CUDA_VISIBLE_DEVICES=${gpu} \
#python main/model_test.py -b 256 -j 8 \
#	--dataset-target ${TARGET} -a ${ARCH} --resume ${MODEL}
