#!/bin/sh
#SOURCE=$1
#TARGET=$2
#ARCH=$3

#1/usr/bin/env bash

if [ -z "$1" ]
  then
    gpu="0"
  else
    gpu="${1}"
fi

CUDA_VISIBLE_DEVICES=${gpu} \
#python3 main/source_pretrain.py -ds ${SOURCE} -dt ${TARGET} -a ${ARCH} --margin 0.0 \
#	--num-instances 4 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 80 --eval-step 5 \
#	--logs-dir logs/${SOURCE}TO${TARGET}/${ARCH}-pretrain

arch_set='resnet50 inceptionv3 densenet'

if [ ${gpu} -eq "0" ]
  then

#################
#################

    AE_set='./logs/__unet/_unet_0501_dtom/unet_dukemtmc_market1501_resnet50_F_0.005* ./logs/__unet/_unet_0501_dtom/unet_dukemtmc_market1501_resnet50_F_0.010*'

    for AEtransfer in ${AE_set}
    do
      for arch in ${arch_set}
      do
        python3 main/source_pretrain.py -ds dukemtmc -dt market1501 -a ${arch} --gpu ${gpu}\
          --margin 0.0 --num-instances 4 -b 24 -j 1 --warmup-step 10 --lr 0.00035 \
	        --milestones 40 70 --iters 200 --epochs 80 --eval-step 5 \
	        --logs-dir ./logs/_pretrain_duke_arg2_0604 \
	        --AEtransfer ${AEtransfer}/checkpoint_79.pth.tar
      done
    done

#################
    pass
  elif [ ${gpu} -eq "1" ]
  then

#################
#################
     AE_set='./logs/__unet/_unet_0528_mtod/*'

    for arch in ${arch_set}
    do
      for AEtransfer in ${AE_set}
      do
        python3 main/source_pretrain.py -ds market1501 -dt dukemtmc -a ${arch} --gpu ${gpu}\
          --margin 0.0 --num-instances 4 -b 16 -j 1 --warmup-step 10 --lr 0.00035 \
	        --milestones 40 70 --iters 200 --epochs 80 --eval-step 5 \
	        --AEtransfer ${AEtransfer}/checkpoint_79.pth.tar
      done
    done


#################
#################

#    python3 main/source_pretrain.py -ds dukemtmc  -dt market1501  -a resnet50 --gpu ${gpu}\
#      --margin 0.0 --num-instances 4 -b 16 -j 1 --warmup-step 10 --lr 0.00035 \
#	    --milestones 40 70 --iters 200 --epochs 80 --eval-step 5 \
#	    --AEtransfer './logs/_before_0505_typeF/unet_dukemtmc_market1501_resnet50_F_0.500_0.5_0.0700_2021-05-04T17:23/checkpoint_79.pth.tar'

#    AE_set='./logs/_before_0505_typeF/unet_dukemtmc_market1501_resnet50_F_0.500_0.5_0.0700_2021-05-04T17:23/checkpoint_79.pth.tar \
#    ./logs/_before_0501_typeF/unet_dukemtmc_market1501_resnet50_F_0.001_0.5_0.0010_2021-04-30T22:04/checkpoint_79.pth.tar \
#    ./logs/_before_0501_typeF/unet_dukemtmc_market1501_resnet50_F_0.050_0.5_0.0100_2021-05-01T07:43/checkpoint_79.pth.tar \
#    ./logs/_before_0501_typeF/unet_dukemtmc_market1501_resnet50_F_0.010_0.5_0.0200_2021-05-01T06:56/checkpoint_79.pth.tar'
#
#    for arch in ${arch_set}
#    do
#      for AEtransfer in ${AE_set}
#      do
#        python3 main/source_pretrain.py -ds dukemtmc  -dt market1501  -a ${arch} --gpu ${gpu}\
#          --margin 0.0 --num-instances 4 -b 16 -j 1 --warmup-step 10 --lr 0.00035 \
#	        --milestones 40 70 --iters 200 --epochs 80 --eval-step 5 \
#	        --AEtransfer ${AEtransfer}
#      done
#    done


#################
    pass
  elif [ ${gpu} -eq "2" ]
  then

#################
#################
    AE_set=' ./logs/__unet/_unet_0501_dtom/unet_dukemtmc_market1501_resnet50_F_0.001*'

    for AEtransfer in ${AE_set}
    do
      for arch in ${arch_set}
      do
        python3 main/source_pretrain.py -ds dukemtmc -dt market1501 -a ${arch} --gpu ${gpu}\
          --margin 0.0 --num-instances 4 -b 24 -j 1 --warmup-step 10 --lr 0.00035 \
	        --milestones 40 70 --iters 200 --epochs 80 --eval-step 5 \
	        --logs-dir ./logs/_pretrain_duke_arg1_0604 \
	        --AEtransfer ${AEtransfer}/checkpoint_79.pth.tar
      done
    done

#################
    pass
fi
#
#
