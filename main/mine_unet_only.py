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
sys.path.insert(0, '../')
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
from gradcam_id import show_cam_on_image, GradCam, FeatureExtractor, ModelOutputs

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
        # T.RandomHorizontalFlip(p=0.5),
        # T.Pad(10),
        # T.RandomCrop((height, width)),
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

    if args.dataset_source=='personx' or args.dataset_target=='personx':

        test_loader = DataLoader(
            Preprocessor(train_set,
                         root=dataset.images_dir, transform=test_transformer),
            batch_size=batch_size*2, num_workers=workers,
            shuffle=False, pin_memory=True)

    else:
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

    # from torchsummary import summary
    # summary(model_pre, input_size=(3, 256, 128))
    # summary(model_unet, input_size=(3, 256, 128))

    # Load from checkpoint
    if args.resume:
        args.batch_size = 1
        args.alpha = args.resume.split('_')[4]
        args.beta = args.resume.split('_')[5]

        iters = args.iters if (args.iters > 0) else None

        dataset_source, num_classes, train_loader_source, test_loader_source = \
            get_data(args.dataset_source, args, args.num_instances, iters)

        dataset_target, _, train_loader_target, test_loader_target = \
            get_data(args.dataset_target, args, 0, iters)

        model_unet = models.create("UNetAuto", num_channels=3,
                                   batch_size=args.batch_size, max_features=1024)
        model_unet.cuda()
        checkpoint = load_checkpoint(args.resume)
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
                break
            for i, (imgs, fnames, pids, _) in enumerate(test_loader_target):
                t_inputs = imgs.cuda()
                print(fnames)
                break

            s_outputs, s_inputs_val = model_unet(s_inputs)
            t_outputs, t_inputs_val = model_unet(t_inputs)

        s1 = torch.cat([s_inputs, s_outputs], 0)
        s2 = torch.cat([t_inputs, t_outputs], 0)
        image_tensor = torch.cat([s1, s2], 0)
        image_grid = v_utils.make_grid(image_tensor.data, nrow=2, padding=0, normalize=True, scale_each=True)

        resultname = '{resultpath}/recon_{alpha}_{beta}_{lr2}_{time}_{filename}'\
            .format(
            resultpath=resultpath, filename=filename,
            alpha=args.alpha, beta=args.beta, lr2=args.lr2, time=curtime[0:16])
        v_utils.save_image(image_grid, resultname)
        print(resultname)
        print(model_unet.d(s_inputs_val))
        print(model_unet.d(t_inputs_val))
        print("==============================")

        return

    prefix = "{src}_{tgt}_{arch}_{type}_{alpha:.3f}_{beta:.1f}_{delta:.4f}_{lr2:.4f}_{time}".format(
        src=args.dataset_source, tgt=args.dataset_target, type=args.type,
        arch=args.arch, alpha=args.alpha, beta=args.beta, delta=args.delta, lr2=args.lr2, time=curtime[0:16]
    )
    if args.prefix is not '':
        prefix = "{0}_{1}".format(
            args.prefix, prefix
        )

    args.logs_dir = "{0}/unet_{1}_ongoing".format(args.logs_dir, prefix)

    if not args.evaluate:
        # sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    else:
        log_dir = osp.dirname(args.resume)
        sys.stdout = Logger(osp.join(log_dir, 'log_test.txt'))

    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    iters = args.iters if (args.iters > 0) else None

    dataset_source, num_classes, train_loader_source, test_loader_source = \
        get_data(args.dataset_source, args, args.num_instances, iters)

    dataset_target, num_classes_t, train_loader_target, test_loader_target = \
        get_data(args.dataset_target, args, 0, iters)

    if args.arch_resume:
        # model_res = meb_models.create("resnet50", num_features=0, dropout=0, num_classes=num_classes)
        model_id = models.create(args.arch, num_features=0, dropout=0, num_classes=num_classes_t)

        checkpoint = load_checkpoint(args.arch_resume)
        copy_state_dict(checkpoint['state_dict'], model_id)
        model_id.cuda()

        if args.arch == "resnet50":
            grayscale_cam_id = GradCam(model=model_id, feature_module=model_id.base[6],
                                   target_layer_names=["2"], use_cuda=True, printly=args.printly)
        elif args.arch == "densenet":
            # model_id = GradCam(model=model_id, feature_module=model_id.base[0],
            #                        target_layer_names=["denseblock3"], name='dense', use_cuda=True,
            #                        printly=args.printly)

            grayscale_cam_id = GradCam(model=model_id, feature_module=model_id.base[0],
                                    target_layer_names=["denseblock4"], name='dense', use_cuda=True,
                                    printly=args.printly)
        elif args.arch == "inceptionv3":
            # model_id = GradCam(model=model_id, feature_module=model_id.base,
            #                          target_layer_names=["16"], name='incep', use_cuda=True, printly=args.printly)

            grayscale_cam_id = GradCam(model=model_id, feature_module=model_id.base,
                                      target_layer_names=["17"], name='incep', use_cuda=True, printly=args.printly)

            # model_id = GradCam(model=model_id, feature_module=model_id.base,
            #                           target_layer_names=["15"], name='incep', use_cuda=True, printly=args.printly)
            #
            # model_id = GradCam(model=model_id, feature_module=model_id.base,
            #                           target_layer_names=["14"], name='incep', use_cuda=True, printly=args.printly)
    else:
        grayscale_cam_id = None

    model_unet = models.create("UNetAuto", num_channels=3,
                               batch_size=args.batch_size, max_features=1024)
    model_unet.cuda()

    model_d = models.create("Discriminator")
    model_d.cuda()

    params_G = []
    params_D = []
    for key, value in model_unet.named_parameters():
        # print(key)

        if not value.requires_grad:
            continue
        params_G += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]

    for key, value in model_d.named_parameters():
        # print(key)

        if not value.requires_grad:
            continue
        params_D += [{"params": [value], "lr": args.lr2, "weight_decay": args.weight_decay}]

    # params_D[5]['params'][0].shape
    optimizer_G = torch.optim.Adam(params_G)
    lr_scheduler_G = WarmupMultiStepLR(optimizer_G, args.milestones, gamma=0.1, warmup_factor=0.01,
                                     warmup_iters=args.warmup_step)

    optimizer_D = torch.optim.Adam(params_D)
    lr_scheduler_D = WarmupMultiStepLR(optimizer_D, args.milestones, gamma=0.1, warmup_factor=0.01,
                                     warmup_iters=args.warmup_step)

    optimizer = [optimizer_G, optimizer_D]
    model2 = [model_unet, model_d]

    # Trainer
    trainer_unet = Trainer_Unet(model2, args, grayscale_cam_id)

    # from torchsummary import summary
    # summary(model_unet, input_size=(3, 224, 256))

    print('---------- Training Start ----------')

    # Start training

    print(args.logs_dir)
    os.makedirs('{path}/recon/'.format(
        path=args.logs_dir
    ), exist_ok=True
    )

    for epoch in range(start_epoch, args.epochs):
        train_loader_source.new_epoch()
        train_loader_target.new_epoch()
        start = time.time()

        # pdb.set_trace()
        # source_inputs = train_loader_target.next()
        # s_inputs, targets = trainer_unet._parse_data(source_inputs)
        # s_inputs2 = s_inputs[0].unsqueeze(0)

        # grayscale_cam_id(s_inputs, None)

        trainer_unet.train(epoch, train_loader_source, train_loader_target, optimizer,
                      train_iters=len(train_loader_source), print_freq=args.print_freq,
                           alpha=args.alpha, beta=args.beta)

        lr_scheduler_G.step()
        lr_scheduler_D.step()

        print(' {0:.3f} seconds \n{1}'.format(time.time() - start, args.logs_dir))

        with torch.no_grad():
            for i, (imgs, fnames, pids, _) in enumerate(test_loader_source):
                s_inputs = imgs.cuda()
                break
            for i, (imgs, fnames, pids, _) in enumerate(test_loader_target):
                t_inputs = imgs.cuda()
                break

            s_outputs = model_unet(s_inputs)
            t_outputs = model_unet(t_inputs)

        s1 = torch.cat([s_inputs, s_outputs], 0)
        s2 = torch.cat([t_inputs, t_outputs], 0)
        image_tensor = torch.cat([s1, s2], 0)
        image_grid = v_utils.make_grid(image_tensor.data, nrow=args.batch_size, padding=0, normalize=True, scale_each=True)

        resultname = '{resultpath}/recon/{epoch}.jpg' \
            .format(
            resultpath=args.logs_dir, epoch=epoch
        )
        v_utils.save_image(image_grid, resultname)

        if ((epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1)):
            start = time.time()

            save_checkpoint({
                'state_dict': model_unet.state_dict(),
                'state_dict2': model_d.state_dict(),
                'epoch': epoch + 1,
            }, False, fpath=osp.join(args.logs_dir, 'checkpoint_{0:02d}.pth.tar'.format(epoch)))

            print(' {0:.3f} seconds'.format(time.time() - start))

    print('---------------Training End------------')

    # print("Test on target domain:")
    # evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True,
    #                    rerank=args.rerank)

    # print('---------------Testing End------------')
    output_directory_complete = '{0}'.format(args.logs_dir[:-8])
    os.rename(args.logs_dir, output_directory_complete)
    print(output_directory_complete)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-training on the source domain")

    # data
    parser.add_argument('-ds', '--dataset-source', type=str, default='dukemtmc',
                        choices=datasets.names())
    parser.add_argument('-dt', '--dataset-target', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('-j', '--workers', type=int, default=1)
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
    parser.add_argument('--arch-resume', type=str, default=None, metavar='PATH')
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--type', type=str, default='F')

    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--lr2', type=float, default=0.0002,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--alpha', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--delta', type=float, default=0.5)

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
    parser.add_argument('--printly', action='store_true')
    main()
