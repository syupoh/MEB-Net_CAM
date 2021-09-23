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
#################
##### 1. pretrain
#################

    SOURCE=market1501
    TARGET=dukemtmc

#    SOURCE=dukemtmc
#    TARGET=market1501

    logsdirname='_pretrain_'${SOURCE}'_unet0720_ID_0707_unet0729_dtom_ID_unet0808_dtom'
    unetdirname='_unet0720_ID_0707_unet0729_dtom_ID_unet0808_dtom_unet0811_dtom'

#    logsdir='./logs/__pretrain/'${logsdirname}
#    arch_set='resnet50 densenet inceptionv3'
#    for arch in ${arch_set}
#    do
#      python3 main/source_pretrain.py -ds ${SOURCE} -dt ${TARGET} -a ${arch} --gpu ${gpu}\
#        --margin 0.0 --num-instances 4 -b 24 -j 1 --warmup-step 10 --lr 0.000035 \
#        --milestones 40 70 --iters 200 --epochs 80 --eval-step 5 \
#        --logs-dir ${logsdir}
#    done

#################
##### 2. unet-train with pretrain gradcam
#################

    pretrain_dir=${logsdir}
    dense_path_set=${pretrain_dir}'/pretrain_'${SOURCE}'_'${TARGET}'_densenet*'
    incep_path_set=${pretrain_dir}'/pretrain_'${SOURCE}'_'${TARGET}'_inceptionv3*'
    resnet_path_set=${pretrain_dir}'/pretrain_'${SOURCE}'_'${TARGET}'_resnet50*'
    alpha_set='0.001'
#    alpha_set='0.5 0.05 0.01 0.001'
    beta_set='0.5'
    delta_set='0'
    gamma_set='0.5'
    lr2_set='0.07'
#    lr2_set='0.1 0.3 0.5 0.7'
    type_set='F'

    for alpha in ${alpha_set}
    do
      for beta in ${beta_set}
      do
        for delta in ${delta_set}
        do
          for gamma in ${gamma_set}
          do
            for lr2 in ${lr2_set}
            do
              for type in ${type_set}
              do
                for dense_path in ${dense_path_set}
                do
                  python3 main/mine_unet_only.py -ds ${TARGET} -dt ${SOURCE}  \
                   --margin 0.0 --num-instance 4 -b 16 -j 1 --warmup-step 10 --lr 0.00035 --lr2 ${lr2} \
                   --milestones 40 70 --epoch 80 --eval-step 5 --gpu ${gpu} --arch densenet \
                   --arch-resume ${dense_path}/model_best.pth.tar \
                   --logs-dir './logs/__unet/'${unetdirname} \
                   --alpha ${alpha} --beta ${beta} --type ${type} --delta ${delta} --gamma ${gamma}
                done

                for incep_path in ${incep_path_set}
                do
                  python3 main/mine_unet_only.py -ds ${TARGET} -dt ${SOURCE}  \
                   --margin 0.0 --num-instance 4 -b 16 -j 1 --warmup-step 10 --lr 0.00035 --lr2 ${lr2} \
                   --milestones 40 70 --epoch 80 --eval-step 5 --gpu ${gpu} --arch inceptionv3 \
                   --arch-resume ${incep_path}/model_best.pth.tar \
                   --logs-dir './logs/__unet/'${unetdirname} \
                   --alpha ${alpha} --beta ${beta} --type ${type} --delta ${delta} --gamma ${gamma}
                done

                for resnet_path in ${resnet_path_set}
                do
                  python3 main/mine_unet_only.py -ds ${TARGET} -dt ${SOURCE}  \
                   --margin 0.0 --num-instance 4 -b 16 -j 1 --warmup-step 10 --lr 0.00035 --lr2 ${lr2} \
                   --milestones 40 70 --epoch 80 --eval-step 5 --gpu ${gpu} --arch resnet50 \
                   --arch-resume ${resnet_path}/model_best.pth.tar \
                   --logs-dir './logs/__unet/'${unetdirname} \
                   --alpha ${alpha} --beta ${beta} --type ${type} --delta ${delta} --gamma ${gamma}
                done
              done
            done
          done
        done
      done
    done

#################
##### 3. pretrain with unet-train
#################

    unetlog='./logs/__unet/'${unetdirname}
    logsdir='./logs/__pretrain/_pretrain_'${SOURCE}${unetdirname}

    arch_set='resnet50'
    AE_set=${unetlog}'/unet_'${TARGET}'_'${SOURCE}_${arch_set}'*'

    for AEtransfer in ${AE_set}
    do
      for arch in ${arch_set}
      do
        python3 main/source_pretrain.py -ds ${SOURCE} -dt ${TARGET} -a ${arch} --gpu ${gpu}\
          --margin 0.0 --num-instances 4 -b 24 -j 1 --warmup-step 10 --lr 0.000035 \
	        --milestones 40 70 --iters 200 --epochs 80 --eval-step 5 \
	        --logs-dir ${logsdir} \
	        --AEtransfer ${AEtransfer}/checkpoint_79.pth.tar
      done
    done

    arch_set='inceptionv3'
    AE_set=${unetlog}'/unet_'${TARGET}'_'${SOURCE}_${arch_set}'*'

    for AEtransfer in ${AE_set}
    do
       for arch in ${arch_set}
      do
        python3 main/source_pretrain.py -ds ${SOURCE} -dt ${TARGET} -a ${arch} --gpu ${gpu}\
          --margin 0.0 --num-instances 4 -b 24 -j 1 --warmup-step 10 --lr 0.000035 \
	        --milestones 40 70 --iters 200 --epochs 80 --eval-step 5 \
	        --logs-dir ${logsdir} \
	        --AEtransfer ${AEtransfer}/checkpoint_79.pth.tar
      done
    done

    arch_set='densenet'
    AE_set=${unetlog}'/unet_'${TARGET}'_'${SOURCE}_${arch_set}'*'

    for AEtransfer in ${AE_set}
    do
      for arch in ${arch_set}
      do
        python3 main/source_pretrain.py -ds ${SOURCE} -dt ${TARGET} -a ${arch} --gpu ${gpu}\
          --margin 0.0 --num-instances 4 -b 24 -j 1 --warmup-step 10 --lr 0.000035 \
	        --milestones 40 70 --iters 200 --epochs 80 --eval-step 5 \
	        --logs-dir ${logsdir} \
	        --AEtransfer ${AEtransfer}/checkpoint_79.pth.tar
      done
    done

##################
###### 4. target train
##################

    pretrain_dir='_pretrain_'${SOURCE}${unetdirname}
    ARCH1_set='logs/__pretrain/'${pretrain_dir}'/pretrain_'${SOURCE}'_'${TARGET}'_densenet* '
    ARCH2_set='logs/__pretrain/'${pretrain_dir}'/pretrain_'${SOURCE}'_'${TARGET}'_inceptionv3* '
    ARCH3_set='logs/__pretrain/'${pretrain_dir}'/pretrain_'${SOURCE}'_'${TARGET}'_resnet50* '


    for ARCH1 in ${ARCH1_set}
    do
      for ARCH2 in ${ARCH2_set}
      do
        for ARCH3 in ${ARCH3_set}
        do
          python3 main/target_train.py -dt ${TARGET} --gpu ${gpu}\
            --num-instances 4 --lr 0.000035 --iters 800 -b 16 --epochs 120 \
            --init-1 ${ARCH1}/model_best.pth.tar \
            --init-2 ${ARCH2}/model_best.pth.tar \
            --init-3 ${ARCH3}/model_best.pth.tar \
            --logs-dir logs/${SOURCE}_${TARGET}/${pretrain_dir}

        done
      done
    done


#################
    pass
  elif [ ${gpu} -eq "1" ]
  then

    SOURCE=dukemtmc
    TARGET=market1501

####################
####################
    pretrain_dir='_pretrain_duke_unet0615_ID_0616'
    ARCH1_set='logs/__pretrain/'${pretrain_dir}'/pretrain_dukemtmc_market1501_densenet_2021-06-16T18:21_66.23'
    ARCH2_set='logs/__pretrain/'${pretrain_dir}'/pretrain_dukemtmc_market1501_inceptionv3_2021-06-16T15:13_59.37'
    ARCH3_set='logs/__pretrain/'${pretrain_dir}'/pretrain_dukemtmc_market1501_resnet50_2021-06-16T12:19_67.28'
#    lr_set='0.0007 0.0014 0.0028 0.0056 0.0112'
    lr_set='0.000001 0.000035 0.00007 0.00014'
    batch_set='48'

    for ARCH1 in ${ARCH1_set}
    do
      for ARCH2 in ${ARCH2_set}
      do
        for ARCH3 in ${ARCH3_set}
        do
          for lr in ${lr_set}
          do
            for batch in ${batch_set}
            do
              python3 main/target_train.py -dt ${TARGET} --gpu ${gpu}\
                --num-instances 4 --lr ${lr} --iters 800 -b ${batch} --epochs 120 \
                --init-1 ${ARCH1}/model_best.pth.tar \
                --init-2 ${ARCH2}/model_best.pth.tar \
                --init-3 ${ARCH3}/model_best.pth.tar \
                --logs-dir logs/${SOURCE}_${TARGET}/${pretrain_dir}
            done
          done
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

#
##################
###### 2. unet-train with pretrain gradcam
##################
#

    alpha_set='0.001'
#    alpha_set='0.5 0.05 0.01 0.001'
    beta_set='0.5'
    delta_set='0'
    gamma_set='0.1 0.3'
    lr2_set='0.07'
#    lr2_set='0.1 0.3 0.5 0.7'
    type_set='F'

    for alpha in ${alpha_set}
    do
      for beta in ${beta_set}
      do
        for delta in ${delta_set}
        do
          for gamma in ${gamma_set}
          do
            for lr2 in ${lr2_set}
            do
              for type in ${type_set}
              do
                python3 main/mine_unet_only.py -ds ${TARGET} -dt ${SOURCE}  \
                 --margin 0.0 --num-instance 4 -b 16 -j 1 --warmup-step 10 --lr 0.00035 --lr2 ${lr2} \
                 --milestones 40 70 --epoch 80 --eval-step 5 --gpu ${gpu} --arch densenet \
                 --logs-dir './logs/__unet/_unet_0923_mtod' \
                 --alpha ${alpha} --beta ${beta} --type ${type} --delta ${delta} --gamma ${gamma}

                python3 main/mine_unet_only.py -ds ${SOURCE} -dt ${TARGET}  \
                 --margin 0.0 --num-instance 4 -b 16 -j 1 --warmup-step 10 --lr 0.00035 --lr2 ${lr2} \
                 --milestones 40 70 --epoch 80 --eval-step 5 --gpu ${gpu} --arch densenet \
                 --logs-dir './logs/__unet/_unet_0923_dtom' \
                 --alpha ${alpha} --beta ${beta} --type ${type} --delta ${delta} --gamma ${gamma}

              done
            done
          done
        done
      done
    done







#################
    pass
fi

#################
#################


#    python3 main/source_pretrain.py -ds dukemtmc  -dt market1501  -a resnet50 --gpu ${gpu}\
#      --margin 0.0 --num-instances 4 -b 16 -j 1 --warmup-step 10 --lr 0.00035 \
#	    --milestones 40 70 --iters 200 --epochs 80 --eval-step 5 \
#	    --AEtransfer './logs/_before_0505_typeF/unet_dukemtmc_market1501_resnet50_F_0.500_0.5_0.0700_2021-05-04T17:23/checkpoint_79.pth.tar'
