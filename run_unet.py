import os
from subprocess import call
import argparse

#########################################
parser = argparse.ArgumentParser(description="Pre-training on the source domain")
# data

parser.add_argument('--gpu', type=int, default='0',
                        help='gpu_ids: e.g. 0  0,1,2  0,2')

args = parser.parse_args()

gpu = args.gpu
#########################################
arch_set='resnet50'
alpha_set='0.005 0.010 0.050'
beta_set='0.5'
resume_set='logs/unet_dukemtmc_market1501_resnet50_0.001_0.1_fake_2021-04-06T09:23/checkpoint_79.pth.tar \
logs/unet_dukemtmc_market1501_resnet50_0.001_0.2_fake_2021-04-06T10:16/checkpoint_79.pth.tar \
logs/unet_dukemtmc_market1501_resnet50_0.001_0.5_fake_2021-04-06T09:17/checkpoint_79.pth.tar \
logs/unet_dukemtmc_market1501_resnet50_0.001_0.6_fake_2021-04-06T10:30/checkpoint_79.pth.tar \
'

resume_set = resume_set.split(' ')
if gpu==0:
    pass
elif gpu==1:
    alpha = '0.001'
    beta = '0.4'
    alpha_set = alpha_set.split(' ')
    beta_set = beta_set.split(' ')
    cmd = 'python3 main/mine_unet_only.py -ds dukemtmc -dt market1501 --margin 0.0 --num-instance 4 '\
          '-j 4 --warmup-step 10 --lr 0.0035 --milestones 40 70 --iters 200 --eval-step 5 --epoch 80 '\
          '-b 16 --gpu {gpu} --alpha {alpha} --beta {beta} '\
          .format(gpu=gpu, alpha=alpha, beta=beta)
    print(cmd)
    os.system(cmd)

    # alpha_set = '0.005 0.010 0.050'
    # beta_set = '0.5'
    #
    # alpha_set = alpha_set.split(' ')
    # for alpha in alpha_set:
    #     for beta in beta_set:
    #         for resume in resume_set:
    #             cmd = 'python3 main/mine_unet_only.py -ds dukemtmc -dt market1501 --margin 0.0 --num-instance 4 '\
    #             '-j 4 --warmup-step 10 --lr 0.0035 --milestones 40 70 --iters 200 --eval-step 5 --epoch 80 '\
    #             '-b 16 --gpu {gpu} --alpha {alpha} --beta {beta} '\
    #                 .format(gpu=gpu, alpha=alpha, beta=beta)
    #             print(cmd)
    #             os.system(cmd)
elif gpu==2:
    alpha_set = '0.001'
    beta_set = '0.8 0.9 1'

    alpha_set = alpha_set.split(' ')
    beta_set = beta_set.split(' ')
    for alpha in alpha_set:
        for beta in beta_set:
            for resume in resume_set:
                cmd = 'python3 main/mine_unet_only.py -ds dukemtmc -dt market1501 --margin 0.0 --num-instance 4 '\
                '-j 4 --warmup-step 10 --lr 0.0035 --milestones 40 70 --iters 200 --eval-step 5 --epoch 80 '\
                '-b 16 --gpu {gpu} --alpha {alpha} --beta {beta} '\
                .format(gpu=gpu, alpha=alpha, beta=beta)
                print(cmd)
                os.system(cmd)


#resume_set='logs/before_fake/unet_dukemtmc_market1501_resnet50_0.001_0.1_2021-04-05T21:24/checkpoint_79.pth.tar
#logs/before_fake/unet_dukemtmc_market1501_resnet50_0.001_0.1_2021-04-02T17:01/checkpoint_79.pth.tar
#logs/before_fake/unet_dukemtmc_market1501_resnet50_0.001_0.1_2021-04-05T13:33/checkpoint_79.pth.tar
#logs/before_fake/unet_dukemtmc_market1501_resnet50_0.001_0.5_2021-04-05T13:33/checkpoint_79.pth.tar
#logs/before_fake/unet_dukemtmc_market1501_resnet50_0.001_0.5_2021-04-05T14:47/checkpoint_79.pth.tar
#logs/before_fake/unet_dukemtmc_market1501_resnet50_0.005_0.5_2021-04-05T14:38/checkpoint_79.pth.tar
#logs/before_fake/unet_dukemtmc_market1501_resnet50_0.010_0.5_2021-04-05T15:39/checkpoint_79.pth.tar
#'



