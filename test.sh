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

test='it is test '${gpu}' tw'
echo ${test}
#if [ ${gpu} -eq "0" ]
#  then
##################
##################
#    SOURCE=dukemtmc
#    TARGET=market1501
#    ARCH=densenet
#    RESUME='logs/dukemtmc_market1501/_pretrain_duke_unet0615_ID_0616/densenet_inceptionv3_resnet50_2021-06-17T02:20_66.08'
#
#    python3 main/model_test.py -dt ${TARGET} \
#      -b 48 --arch ${ARCH} --rerank --gpu ${gpu} \
#      --resume ${RESUME} \
#
##################
#    SOURCE=dukemtmc
#    TARGET=market1501
#    ARCH=densenet
#    RESUME='logs/dukemtmc_market1501/_pretrain_duke_unet0615_ID_0616/densenet_inceptionv3_resnet50_2021-06-17T02:20_66.08'
#
#    python3 main/model_test.py -dt ${TARGET} \
#      -b 48 --arch ${ARCH} --rerank --gpu ${gpu} \
#      --resume ${RESUME} \
#
##################
#    pass
#  elif [ ${gpu} -eq "1" ]
#  then
##################
#    SOURCE=dukemtmc
#    TARGET=market1501
#    ARCH=densenet
#    RESUME='logs/dukemtmc_market1501/_pretrain_duke_unet0615_ID_0616/densenet_inceptionv3_resnet50_2021-06-17T02:20_66.08'
#
#    python3 main/model_test.py -dt ${TARGET} \
#      -b 48 --arch ${ARCH} --rerank --gpu ${gpu} \
#      --resume ${RESUME} \
##################
#    pass
#  elif [ ${gpu} -eq "2" ]
#  then
##################
#    pass
#fi
#MODEL=logs/xxxx/xxxx-MEB-Net/checkpoint.pt.pth
#
#CUDA_VISIBLE_DEVICES=${gpu} \
#python main/model_test.py -b 256 -j 8 \
#	--dataset-target ${TARGET} -a ${ARCH} --resume ${MODEL}
