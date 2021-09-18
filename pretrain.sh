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

    logsdir='./logs/_pretrain_duke_unet0615_ID_0616'

    AE_set='./logs/__unet/_unet_0615_mtod_ID/unet_market1501_dukemtmc_resnet50_F_0.001_0.5_0.0000_0.0700_2021-06-16T04:07'
    arch_set='resnet50'

    for AEtransfer in ${AE_set}
    do
      for arch in ${arch_set}
      do
        python3 main/source_pretrain.py -ds dukemtmc -dt market1501 -a ${arch} --gpu ${gpu}\
          --margin 0.0 --num-instances 4 -b 24 -j 1 --warmup-step 10 --lr 0.00035 \
	        --milestones 40 70 --iters 200 --epochs 80 --eval-step 5 \
	        --logs-dir ${logsdir} \
	        --AEtransfer ${AEtransfer}/checkpoint_79.pth.tar
      done
    done

    AE_set='./logs/__unet/_unet_0615_mtod_ID/unet_market1501_dukemtmc_inceptionv3_F_0.001_0.5_0.0000_0.0700_2021-06-15T21:21'
    arch_set='inceptionv3'

    for AEtransfer in ${AE_set}
    do
      for arch in ${arch_set}
      do
        python3 main/source_pretrain.py -ds dukemtmc -dt market1501 -a ${arch} --gpu ${gpu}\
          --margin 0.0 --num-instances 4 -b 24 -j 1 --warmup-step 10 --lr 0.00035 \
	        --milestones 40 70 --iters 200 --epochs 80 --eval-step 5 \
	        --logs-dir ${logsdir} \
	        --AEtransfer ${AEtransfer}/checkpoint_79.pth.tar
      done
    done

    AE_set='./logs/__unet/_unet_0615_mtod_ID/unet_market1501_dukemtmc_densenet_F_0.001_0.5_0.0000_0.0700_2021-06-15T14:42'
    arch_set='densenet'

    for AEtransfer in ${AE_set}
    do
      for arch in ${arch_set}
      do
        python3 main/source_pretrain.py -ds dukemtmc -dt market1501 -a ${arch} --gpu ${gpu}\
          --margin 0.0 --num-instances 4 -b 24 -j 1 --warmup-step 10 --lr 0.00035 \
	        --milestones 40 70 --iters 200 --epochs 80 --eval-step 5 \
	        --logs-dir ${logsdir} \
	        --AEtransfer ${AEtransfer}/checkpoint_79.pth.tar
      done
    done

#################
#################

#    AE_set='./logs/__unet/_unet_0501_dtom/unet_dukemtmc_market1501_resnet50_F_0.005* ./logs/__unet/_unet_0501_dtom/unet_dukemtmc_market1501_resnet50_F_0.010*'
#    logsdir='./logs/_pretrain_duke_arg2_0604'
#
#    for AEtransfer in ${AE_set}
#    do
#      for arch in ${arch_set}
#      do
#        python3 main/source_pretrain.py -ds dukemtmc -dt market1501 -a ${arch} --gpu ${gpu}\
#          --margin 0.0 --num-instances 4 -b 24 -j 1 --warmup-step 10 --lr 0.00035 \
#	        --milestones 40 70 --iters 200 --epochs 80 --eval-step 5 \
#	        --logs-dir ${logsdir} \
#	        --AEtransfer ${AEtransfer}/checkpoint_79.pth.tar
#      done
#    done

#################
    pass
  elif [ ${gpu} -eq "1" ]
  then

#################
#################

    arch_set='densenet'
    AE_set='./logs/__unet/_unet_0613_mtod_ID/unet_market1501_dukemtmc_densenet_*'
    logsdir='./logs/__pretrain/_pretrain_duke_unet0613_ID_0621'

    for AEtransfer in ${AE_set}
    do
      for arch in ${arch_set}
      do
        python3 main/source_pretrain.py -ds dukemtmc -dt market1501 -a ${arch} --gpu ${gpu}\
          --margin 0.0 --num-instances 4 -b 24 -j 1 --warmup-step 10 --lr 0.00035 \
	        --milestones 40 70 --iters 200 --epochs 80 --eval-step 5 \
	        --logs-dir ${logsdir} \
	        --AEtransfer ${AEtransfer}/checkpoint_79.pth.tar
      done
    done

    arch_set='inceptionv3'
    AE_set='./logs/__unet/_unet_0613_mtod_ID/unet_market1501_dukemtmc_inceptionv3_*'

    for AEtransfer in ${AE_set}
    do
      for arch in ${arch_set}
      do
        python3 main/source_pretrain.py -ds dukemtmc -dt market1501 -a ${arch} --gpu ${gpu}\
          --margin 0.0 --num-instances 4 -b 24 -j 1 --warmup-step 10 --lr 0.00035 \
	        --milestones 40 70 --iters 200 --epochs 80 --eval-step 5 \
	        --logs-dir ${logsdir} \
	        --AEtransfer ${AEtransfer}/checkpoint_79.pth.tar
      done
    done

    arch_set='resnet50'
    AE_set='./logs/__unet/_unet_0613_mtod_ID/unet_market1501_dukemtmc_resnet50_*'

    for AEtransfer in ${AE_set}
    do
      for arch in ${arch_set}
      do
        python3 main/source_pretrain.py -ds dukemtmc -dt market1501 -a ${arch} --gpu ${gpu}\
          --margin 0.0 --num-instances 4 -b 24 -j 1 --warmup-step 10 --lr 0.00035 \
	        --milestones 40 70 --iters 200 --epochs 80 --eval-step 5 \
	        --logs-dir ${logsdir} \
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
    pass
fi
#
#
