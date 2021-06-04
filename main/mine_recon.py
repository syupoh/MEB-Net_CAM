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

import pdb

from datetime import datetime


start_epoch = best_mAP = 0

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

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(args.gpu)

    global start_epoch, best_mAP

    cudnn.benchmark = True

    print("==========\nArgs:{}\n==========".format(args))

    prefix = "{time}".format(time=curtime[0:16]
    )
    if args.prefix is not '':
        prefix = "{0}_{1}".format(
            args.prefix, prefix
        )

    args.logs_dir = "gradcam/results_recon"
    # args.logs_dir = "{0}/unet_recon_{1}_ongoing".format(args.logs_dir, prefix)
    sys.stdout = Logger(osp.join(args.logs_dir, '{0}_ongoing.txt'.format(prefix)))

    resume_set = os.listdir('logs/')

    resume_set='logs/unet_dukemtmc_market1501_resnet50_0.001_0.5_B_2021-04-08T03:54/checkpoint_79.pth.tar ' +\
    'logs/unet_dukemtmc_market1501_resnet50_0.001_0.5_B_2021-04-08T11:18/checkpoint_79.pth.tar ' +\
    'logs/unet_dukemtmc_market1501_resnet50_0.001_0.5_A_2021-04-08T03:46/checkpoint_79.pth.tar ' +\
    'logs/unet_dukemtmc_market1501_resnet50_0.001_0.5_C_2021-04-08T05:13/checkpoint_79.pth.tar ' +\
    'logs/unet_dukemtmc_market1501_resnet50_0.001_0.5_C_2021-04-08T11:18/checkpoint_79.pth.tar '

    # Load from checkpoint
    resume_set = resume_set.split(' ')
    # print(resume_set)
    for resume in resume_set:
        # print(resume)
        args.batch_size = 1
        args.alpha = resume.split('_')[4]
        args.beta = resume.split('_')[5]
        args.type = resume.split('_')[6]

        iters = args.iters if (args.iters > 0) else None

        dataset_source, num_classes, train_loader_source, test_loader_source = \
            get_data(args.dataset_source, args, args.num_instances, iters)

        dataset_target, _, train_loader_target, test_loader_target = \
            get_data(args.dataset_target, args, 0, iters)

        model_unet = models.create("UNetAuto", num_channels=3,
                                   batch_size=args.batch_size, max_features=1024)
        model_unet.cuda()
        checkpoint = load_checkpoint(resume)
        copy_state_dict(checkpoint['state_dict'], model_unet)
        start_epoch = checkpoint['epoch']

        resultpath = args.image_path.split('/')
        filename = resultpath[-1]
        resultpath = '/'.join(resultpath[:-2])
        resultpath = resultpath + '/results'

        print("==============================")

        # trainer_unet = Trainer_Unet(model_unet, args)
        #
        # source_inputs = train_loader_source.next()
        # target_inputs = train_loader_target.next()
        #
        # s_inputs, targets = trainer_unet._parse_data(source_inputs)
        # t_inputs, _ = trainer_unet._parse_data(target_inputs)

        with torch.no_grad():
            for i, (imgs, fnames, pids, _) in enumerate(test_loader_source):
                s_inputs = imgs.cuda()
                print(fnames)
                print(imgs)
                pdb.set_trace()
                break
            for i, (imgs, fnames, pids, _) in enumerate(test_loader_target):
                t_inputs = imgs.cuda()
                print(fnames)
                break

            pdb.set_trace()
            s_outputs, s_inputs_val = model_unet(s_inputs)
            t_outputs, t_inputs_val = model_unet(t_inputs)
            _, t_outputs_val = model_unet(t_outputs)

        s1 = torch.cat([s_inputs, s_outputs], 0)
        s2 = torch.cat([t_inputs, t_outputs], 0)
        image_tensor = torch.cat([s1, s2], 0)
        image_grid = v_utils.make_grid(image_tensor.data, nrow=2, padding=0, normalize=True, scale_each=True)

        resultname = '{resultpath}/recon_{alpha}_{beta}_{type}_{time}.jpg'\
            .format(
            resultpath=resultpath, filename=filename,
            alpha=args.alpha, beta=args.beta, type=args.type, time=curtime[0:16])
        v_utils.save_image(image_grid, resultname)
        print(resultname)
        print(model_unet.d(s_inputs_val))
        print(model_unet.d(t_inputs_val))
        print(model_unet.d(t_outputs_val))
        print("==============================")

    os.rename((osp.join(args.logs_dir, '{0}_ongoing.txt'.format(prefix))),
              (osp.join(args.logs_dir, '{0}.txt'.format(prefix))))

    return

        #################
        ### PIL resizing normalizing
        #################

        # print(args.image_path)
        # img = cv2.imread(args.image_path, 1)
        # img_clone = img.copy()
        #
        # color_converted = cv2.cvtColor(img_clone, cv2.COLOR_BGR2RGB)
        #
        # pil_image = Image.fromarray(color_converted)
        # input_img, resized_image = preprocess_image(pil_image, args)
        #
        #
        # # totensor = T.ToTensor()(resized_image) # 3 x 256 x 128
        # # v_utils.save_image(totensor, '{0}/totensor_{1}'.format(resultpath, filename))
        #
        # input_img = input_img.cuda() # 1 x 3 x 256 x 128
        # outputs, s_inputs_val = model_unet(input_img)
        #
        # pdb.set_trace()
        #
        # # totensor2 = T.ToTensor()(np.array(input_img.squeeze().cpu())) # 3 x 256 x 128
        # # v_utils.save_image(totensor2, '{0}/totensor2_{1}'.format(resultpath, filename))
        #
        #
        # v_utils.save_image(input_img.clone(), '{0}/input_{1}'.format(resultpath, filename))
        # # v_utils.save_image(outputs.clone(), '{0}/outputs_{2}_{3}_{1}'.format(resultpath, filename, args.alpha, args.beta))
        #
        # pdb.set_trace()
        #
        # # min = float(input_img.min())
        # # max = float(input_img.max())
        # # input_img.clamp_(min=min, max=max)
        # # input_img.add_(-min).div_(max - min + 1e-5)
        # # v_utils.save_image(input_img, '{0}/input2_{1}'.format(resultpath, filename))
        # # v_utils.save_image(outputs, '{0}/outputs_{1}_{2}_{3}'.format(resultpath, filename, args.alpha, args.beta))
        #
        #
        # # cv2.imwrite('{0}/s_outputs_{1}_{2}_{3}'.format(resultpath, filename, args.alpha, args.beta),
        # #             cv2.cvtColor(s_outputs, cv2.COLOR_RGB2BGR))


        # return



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-training on the source domain")
    # data
    parser.add_argument('-ds', '--dataset-source', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-dt', '--dataset-target', type=str, default='dukemtmc',
                        choices=datasets.names())
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    parser.add_argument('--gpu', type=int, default='0',
                        help='gpu_ids: e.g. 0  0,1,2  0,2')

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
    main()
