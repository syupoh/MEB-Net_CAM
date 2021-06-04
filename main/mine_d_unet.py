# -*- conding: utf-8 -*-

from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import os
import sys
import cv2
from PIL import Image

sys.path.insert(0, os.getcwd())
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

import time
from mebnet import datasets
from mebnet import models
from mebnet.trainers import PreTrainer, Trainer_Unet
from mebnet.evaluators import Evaluator
from mebnet.utils.data import IterLoader
from mebnet.utils.data import transforms as T
from mebnet.utils.data.sampler import RandomMultipleGallerySampler
from mebnet.utils.data.preprocessor import Preprocessor
from mebnet.utils.logging import Logger
from mebnet.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from mebnet.utils.lr_scheduler import WarmupMultiStepLR
import torchvision.utils as v_utils
import numpy as np
import pdb

from datetime import datetime
from torch.autograd import Variable


start_epoch = best_mAP = 0

def compute_gradient_penalty(D, real_samples, fake_samples):
    Tensor = torch.cuda.FloatTensor
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates, _ = D(interpolates)
    fake = Variable(Tensor(np.ones(d_interpolates.shape)), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)

def preprocess_image(img, args):

    Tresize = T.Compose([
        T.Resize((args.height, args.width), interpolation=3),
    ])

    # It means imagenet stastics
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    preprocessing = T.Compose([
        T.Resize(args.height, args.width),
        T.ToTensor(),
        normalize,
    ])

    resized_img = Tresize(img.copy())

    return preprocessing(img).unsqueeze(0), resized_img


# def get_data(name, data_dir, height, width, batch_size, workers, num_instances, iters=200):
def get_data(name, args, num_instances, iters=200):
    # root = osp.join(data_dir, name)i
    root = args.data_dir
    height = args.height
    width = args.width
    batch_size = args.batch_size
    workers = args.workers

    dataset = datasets.create(name, root)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.train
    num_classes = dataset.num_train_pids

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer
    ])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None

    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir,
                                transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, test_loader


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    now = datetime.now()
    curtime = now.isoformat()

    # chk_unet = './logs/unet_dukemtmc_market1501_resnet50_0.001_0.4_D_2021-04-19T17:37/checkpoint_79.pth.tar'
    # chk_unet = './logs/unet_dukemtmc_market1501_resnet50_0.001_0.5_E_2021-04-20T14:54/checkpoint_79.pth.tar'
    # chk_unet = './logs/unet_dukemtmc_market1501_resnet50_0.001_0.5_E_2021-04-20T14:41/checkpoint_79.pth.tar'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(args.gpu)

    chk_dir = 'unet_dukemtmc_market1501_resnet50_0.010_0.5_0.0002_A_2021-04-29T10:59'
    # chk_unet = './logs/unet_dukemtmc_market1501_resnet50_0.010_0.5_0.0002_C_2021-04-29T10:58/checkpoint_74.pth.tar'
    chk_unet = './logs/{0}/checkpoint_74.pth.tar'.format(chk_dir)

    global start_epoch, best_mAP

    cudnn.benchmark = True

    if args.resume == '':
        args.resume = chk_unet

    chk_dir = args.resume.split('/')[0:-1]
    chk_dir = '/'.join(chk_dir)

    sys.stdout = Logger('{0}/log_d.txt'.format(chk_dir))
    # Load from checkpoint

    print("==========\nArgs:{}\n==========".format(args))

    if args.resume:
        # args.batch_size = 1
        # args.alpha = args.resume.split('_')[4]
        # args.beta = args.resume.split('_')[5]

        iters = args.iters if (args.iters > 0) else None

        dataset_source, num_classes, train_loader_source, test_loader_source = \
            get_data(args.dataset_source, args, args.num_instances, iters)

        dataset_target, _, train_loader_target, test_loader_target = \
            get_data(args.dataset_target, args, 0, iters)

        model_unet = models.create("UNetAuto", num_channels=3,
                                   batch_size=args.batch_size, max_features=1024)
        model_d = models.create("Discriminator")
        model_unet.cuda()
        model_d.cuda()

        checkpoint = load_checkpoint(args.resume)

        copy_state_dict(checkpoint['state_dict'], model_unet)
        copy_state_dict(checkpoint['state_dict2'], model_d)
        start_epoch = checkpoint['epoch']

        resultpath = args.image_path.split('/')
        filename = resultpath[-1]
        resultpath = '/'.join(resultpath[:-2])
        resultpath = resultpath + '/results/'

        foldername = chk_unet.split('/')[2]
        resultpath = resultpath + foldername
        os.makedirs(resultpath, exist_ok=True)
        print("==============================")

#########################
        # model2 = [model_unet, model_d]
        # trainer_unet = Trainer_Unet(model2, args)
        #
        # source_inputs = train_loader_source.next()
        # target_inputs = train_loader_target.next()
        #
        # s_inputs, targets = trainer_unet._parse_data(source_inputs)
        # t_inputs, _ = trainer_unet._parse_data(target_inputs)
# #########################
# #########################
#         t_outputs = model_unet(t_inputs)
#         t_inputs_val = model_d(t_inputs)
#         s_inputs_val = model_d(s_inputs)
#         t_outputs_val = model_d(t_outputs.detach())

        # StarGAN
        # t_inputs_val, _ = model_d(t_inputs)
        # s_inputs_val, _ = model_d(s_inputs)
        # t_outputs_val, _ = model_d(t_outputs.detach())

        # lambda_gp = 10
        # # Gradient penalty
        # gradient_penalty = compute_gradient_penalty(model_d, s_inputs, t_outputs)
        # # Adversarial loss
        # loss_d = -torch.mean(s_inputs_val) + torch.mean(t_outputs_val) + \
        #          lambda_gp * gradient_penalty

#########################

        itern = 10
        minin = 5
        with torch.no_grad():

            print('source')
            temp = 0
            temp2 = 0
            fulllen = 0
            full_iter = len(test_loader_source)
            if itern == 1:
                itern = full_iter
            iter_freq = int(full_iter / itern)

            for i, (imgs, fnames, pids, _) in enumerate(test_loader_source):
                s_inputs = imgs.cuda()
                # print(fnames)
                s_outputs = model_unet(s_inputs)

                s_inputs_val = model_d(s_inputs)
                s_outputs_val = model_d(s_outputs)

                if len(s_inputs_val)==2:
                    s_inputs_val = s_inputs_val[0]
                if len(s_outputs_val)==2:
                    s_outputs_val = s_outputs_val[0]

                score = torch.mean(s_inputs_val, dim=(2,3))
                temp += sum((score > 0).type(torch.float))

                score = torch.mean(s_outputs_val, dim=(2,3))
                temp2 += sum((score > 0).type(torch.float))

                fulllen += len(imgs)

                if i % iter_freq == 0 :
                    print('{0}/{1} ({2:02.2f}%)'.format(i, full_iter, i/full_iter*100), end='\r')
                if i > minin and args.minimode:
                    break
            print('                                  ', end='\r')
            print(' Completed')
            print("==============================")
            print("{0:d}/{1} ({2:2.2f}%)".format(int(temp[0]), fulllen, temp[0]/fulllen*100))
            print("{0:d}/{1} ({2:2.2f}%)".format(int(temp2[0]), fulllen, temp2[0]/fulllen*100))
            print("==============================")

            print('target')
            temp = 0
            temp2 = 0
            fulllen = 0
            full_iter = len(test_loader_target)
            if itern == 1:
                itern = full_iter
            iter_freq = int(full_iter / itern)

            for i, (imgs, fnames, pids, _) in enumerate(test_loader_target):
                t_inputs = imgs.cuda()
                # print(fnames)
                t_outputs = model_unet(t_inputs)

                t_inputs_val = model_d(t_inputs)
                t_outputs_val = model_d(t_outputs)

                if len(t_inputs_val)==2:
                    t_inputs_val = t_inputs_val[0]
                if len(t_outputs_val)==2:
                    t_outputs_val = t_outputs_val[0]

                score = torch.mean(t_inputs_val, dim=(2,3))
                temp += sum((score < 0).type(torch.float))

                score = torch.mean(t_outputs_val, dim=(2,3))
                temp2 += sum((score < 0).type(torch.float))

                fulllen += len(imgs)

                if i % iter_freq == 0 :
                    print('{0}/{1} ({2:02.2f}%)'.format(i, full_iter, i/full_iter*100), end='\r')
                if i > minin and args.minimode:
                    break
            print('                                  ', end='\r')
            print(' Completed')
            print("==============================")
            print("{0:d}/{1} ({2:2.2f}%)".format(int(temp[0]), fulllen, temp[0]/fulllen*100))
            print("{0:d}/{1} ({2:2.2f}%)".format(int(temp2[0]), fulllen, temp2[0]/fulllen*100))
            print("==============================")

        return






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-training on the source domain")
    # data
    parser.add_argument('-ds', '--dataset-source', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-dt', '--dataset-target', type=str, default='dukemtmc',
                        choices=datasets.names())
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--type', type=str, default='B')
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--alpha', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=0.5)

    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70],
                        help='milestones for the learning rate decay')
    # training configs
    parser.add_argument('--gpu', type=int, default='0',
                        help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--eval-step', type=int, default=5)
    parser.add_argument('--rerank', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--margin', type=float, default=0.0, help='margin for the triplet loss with batch hard')
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='/data2/syupoh/dataset/')
    parser.add_argument('--image-path', type=str, default='./gradcam/imgs/person1.jpg',
                        help='Input image path')
    # default=osp.join(os.getcwd(), 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(os.getcwd(), 'logs'))
    parser.add_argument('--minimode', action='store_true')
    main()
