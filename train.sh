#!/bin/sh
#SOURCE=$1
#TARGET=$2
#ARCH1=$3
#ARCH2=$4
#ARCH3=$5

if [ -z "$1" ]
  then
    gpu="0"
  else
    gpu="${1}"
fi

#SOURCE=dukemtmc
#TARGET=market15Q01
#ARCH1=densenet
#ARCH2=resnet50
#ARCH3=inceptionv3

AE_set='./logs/_before_0505_typeF/unet_dukemtmc_market1501_resnet50_F_0.500_0.5_0.0700_2021-05-04T17:23/checkpoint_79.pth.tar
./logs/_before_0501_typeF/unet_dukemtmc_market1501_resnet50_F_0.001_0.5_0.0010_2021-04-30T22:04/checkpoint_79.pth.tar
./logs/_before_0501_typeF/unet_dukemtmc_market1501_resnet50_F_0.050_0.5_0.0100_2021-05-01T07:43/checkpoint_79.pth.tar
./logs/_before_0501_typeF/unet_dukemtmc_market1501_resnet50_F_0.010_0.5_0.0200_2021-05-01T06:56/checkpoint_79.pth.tar'

if [ ${gpu} -eq "0" ]
  then

    SOURCE=dukemtmc
    TARGET=market1501
    ARCH1_set='densenet_2021-05-25T13:16 densenet_2021-05-25T15:55
    densenet_2021-05-25T18:36 densenet_2021-05-25T21:15'
    ARCH2_set='resnet50_2021-05-24T19:34 resnet50_2021-05-24T21:48
    resnet50_2021-05-24T23:51 resnet50_2021-05-25T01:55'
    ARCH3_set='inceptionv3_2021-05-25T03:58 inceptionv3_2021-05-25T06:03
    inceptionv3_2021-05-25T08:18'


    # inceptionv3_2021-05-25T10:43

    for ARCH1 in ${ARCH1_set}
    do
      for ARCH2 in ${ARCH2_set}
      do
        for ARCH3 in ${ARCH3_set}
        do
          python3 main/target_train.py -dt ${TARGET} --gpu ${gpu}\
            --num-instances 4 --lr 0.00035 --iters 800 -b 8 --epochs 80 \
            --init-1 logs/_pretrain_0525/pretrain_${SOURCE}_${TARGET}_${ARCH1}/model_best.pth.tar \
            --init-2 logs/_pretrain_0525/pretrain_${SOURCE}_${TARGET}_${ARCH2}/model_best.pth.tar \
            --init-3 logs/_pretrain_0525/pretrain_${SOURCE}_${TARGET}_${ARCH3}/model_best.pth.tar \
            --logs-dir logs/${SOURCE}_${TARGET}/${ARCH1}-${ARCH2}-${ARCH3}
        done
      done
    done

#################
#################
#    SOURCE=market1501
#    TARGET=dukemtmc
#    ARCH1=densenet_2020-10-22T18:51
#    ARCH2=resnet50_2020-10-22T20:29
#    ARCH3=inceptionv3_2020-10-23T05:02
#
#    for unet in ${AE_set}
#    do
#      python3 main/target_train.py -dt ${TARGET} --gpu ${gpu}\
#        --num-instances 4 --lr 0.00035 --iters 800 -b 8 --epochs 40 \
#        --init-1 logs/_pre/pretrain_${SOURCE}_${TARGET}_${ARCH1}/model_best.pth.tar \
#        --init-2 logs/_pre/pretrain_${SOURCE}_${TARGET}_${ARCH2}/model_best.pth.tar \
#        --init-3 logs/_pre/pretrain_${SOURCE}_${TARGET}_${ARCH3}/model_best.pth.tar \
#        --logs-dir logs/${SOURCE}_${TARGET}/${ARCH1}-${ARCH2}-${ARCH3}-MEB-Net \
#        --unet ${unet}
#    done

#################
    pass
  elif [ ${gpu} -eq "1" ]
  then

    SOURCE=dukemtmc
    TARGET=market1501

####################
####################

    ARCH1_set='logs/__pretrain/_pretrain_duke_0531/pretrain_dukemtmc_market1501_densenet_2021-05-30T12:05_61.8'
    ARCH2_set='logs/__pretrain/_pretrain_duke_0531/pretrain_dukemtmc_market1501_inceptionv3_2021-05-29T08:16_56.8'
    ARCH3_set='logs/__pretrain/_pretrain_duke_0531/pretrain_dukemtmc_market1501_resnet50_2021-05-28T04:04_62.3'

    for ARCH1 in ${ARCH1_set}
    do
      for ARCH2 in ${ARCH2_set}
      do
        for ARCH3 in ${ARCH3_set}
        do
          python3 main/target_train.py -dt ${TARGET} --gpu ${gpu}\
            --num-instances 4 --lr 0.00035 --iters 800 -b 48 --epochs 80 \
            --init-1 ${ARCH1}/model_best.pth.tar \
            --init-2 ${ARCH2}/model_best.pth.tar \
            --init-3 ${ARCH3}/model_best.pth.tar \
            --logs-dir logs/${SOURCE}_${TARGET}
        done
      done
    done

####################
####################

#    ARCH1_set='logs/dukemtmc_market1501/densenet_resnet50_inceptionv3_2021-06-01T05:49_67.1/model0_checkpoint.pth.tar'
#    ARCH2_set='logs/dukemtmc_market1501/densenet_resnet50_inceptionv3_2021-06-01T05:49_67.1/model1_checkpoint.pth.tar'
#    ARCH3_set='logs/dukemtmc_market1501/densenet_resnet50_inceptionv3_2021-06-01T05:49_67.1/model2_checkpoint.pth.tar'
#
#    for ARCH1 in ${ARCH1_set}
#    do
#      for ARCH2 in ${ARCH2_set}
#      do
#        for ARCH3 in ${ARCH3_set}
#        do
#          python3 main/target_train.py -dt ${TARGET} --gpu ${gpu}\
#            --num-instances 4 --lr 0.00035 --iters 800 -b 48 --epochs 80 \
#            --init-1 ${ARCH1} \
#            --init-2 ${ARCH2} \
#            --init-3 ${ARCH3} \
#            --start-epoch 64 \
#            --logs-dir logs/${SOURCE}_${TARGET}
#        done
#      done
#    done

#################
    pass
  elif [ ${gpu} -eq "2" ]
  then

    SOURCE=dukemtmc
    TARGET=market1501

    ARCH1_set='logs/_pretrain_duke_0531/pretrain_dukemtmc_market1501_densenet_2021-05-30T12:05_61.8'
    ARCH2_set='logs/_pretrain_duke_0531/pretrain_dukemtmc_market1501_inceptionv3_2021-05-29T08:16_56.8'
    ARCH3_set='logs/_pretrain_duke_0531/pretrain_dukemtmc_market1501_resnet50_2021-05-28T04:04_62.3'

    for ARCH1 in ${ARCH1_set}
    do
      for ARCH2 in ${ARCH2_set}
      do
        for ARCH3 in ${ARCH3_set}
        do
          python3 main/target_train.py -dt ${TARGET} --gpu ${gpu}\
            --num-instances 4 --lr 0.00035 --iters 800 -b 48 --epochs 80 \
            --init-1 ${ARCH1}/model_best.pth.tar \
            --init-2 ${ARCH2}/model_best.pth.tar \
            --init-3 ${ARCH3}/model_best.pth.tar \
            --logs-dir logs/${SOURCE}_${TARGET}
        done
      done
    done

#    SOURCE=market1501
#    TARGET=dukemtmc
#    ARCH1=densenet_2020-10-22T18:51
#    ARCH2=resnet50_2020-10-22T20:29
#    ARCH3=inceptionv3_2020-10-23T05:02
#
#    for unet in ${AE_set}
#    do
#      python3 main/target_train.py -dt ${TARGET} --gpu ${gpu}\
#        --num-instances 4 --lr 0.00035 --iters 800 -b 32 --epochs 40 \
#        --init-1 logs/_pre/pretrain_${SOURCE}_${TARGET}_${ARCH1}/model_best.pth.tar \
#        --init-2 logs/_pre/pretrain_${SOURCE}_${TARGET}_${ARCH2}/model_best.pth.tar \
#        --init-3 logs/_pre/pretrain_${SOURCE}_${TARGET}_${ARCH3}/model_best.pth.tar \
#        --logs-dir logs/${SOURCE}_${TARGET}/${ARCH1}-${ARCH2}-${ARCH3}-MEB-Net \
#        --unet ${unet}
#    done



#################
    pass
fi
