from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import os
import sys
sys.path.insert(0, os.getcwd())
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

import time
from mebnet import datasets
from mebnet import models
from mebnet.trainers import PreTrainer
from mebnet.evaluators import Evaluator
from mebnet.utils.data import IterLoader
from mebnet.utils.data import transforms as T
from mebnet.utils.data.sampler import RandomMultipleGallerySampler
from mebnet.utils.data.preprocessor import Preprocessor
from mebnet.utils.logging import Logger
from mebnet.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from mebnet.utils.lr_scheduler import WarmupMultiStepLR

import pdb

from datetime import datetime
start_epoch = best_mAP = 0

def get_data(name, data_dir, height, width, batch_size, workers, num_instances, iters=200):
    # root = osp.join(data_dir, name)
    root = data_dir

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
    global start_epoch, best_mAP

    cudnn.benchmark = True

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(args.gpu)

    now = datetime.now()
    curtime = now.isoformat()

    prefix = "{src}_{tgt}_{arch}_{time}".format(
        src=args.dataset_source, tgt=args.dataset_target,
        arch=args.arch, time=curtime[0:16]
    )
    args.logs_dir = "{0}/pretrain_{1}_ongoing".format(args.logs_dir, prefix)
    if not args.evaluate:
        # sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    else:
        log_dir = osp.dirname(args.resume)
        sys.stdout = Logger(osp.join(log_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    iters = args.iters if (args.iters>0) else None
    dataset_source, num_classes, train_loader_source, test_loader_source = \
        get_data(args.dataset_source, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers, args.num_instances, iters)

    dataset_target, _, train_loader_target, test_loader_target = \
        get_data(args.dataset_target, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers, 0, iters)

    # Create model
    model = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=num_classes)
    model.cuda()
    # model = nn.DataParallel(model)

    # Load from checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model)
        start_epoch = checkpoint['epoch']
        best_mAP = checkpoint['best_mAP']
        print("=> Start epoch {}  best mAP {:.1%}"
              .format(start_epoch, best_mAP))


    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        print("Test on source domain:")
        evaluator.evaluate(test_loader_source, dataset_source.query, dataset_source.gallery, cmc_flag=True, rerank=args.rerank)
        print("Test on target domain:")
        evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True, rerank=args.rerank)
        return


    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(params)
    lr_scheduler = WarmupMultiStepLR(optimizer, args.milestones, gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)

    # Trainer
    #    model_set='./logs/_0505_typeF/unet_dukemtmc_market1501_resnet50_F_0.500_0.5_0.0700_2021-05-04T17:23/checkpoint_79.pth.tar \
    #    ./logs/_before_0501_typeF/unet_dukemtmc_market1501_resnet50_F_0.001_0.5_0.0010_2021-04-30T22:04/checkpoint_79.pth.tar \
    #    ./logs/_before_0501_typeF/unet_dukemtmc_market1501_resnet50_F_0.050_0.5_0.0100_2021-05-01T07:43/checkpoint_79.pth.tar \
    #    ./logs/_before_0501_typeF/unet_dukemtmc_market1501_resnet50_F_0.010_0.5_0.0200_2021-05-01T06:56/checkpoint_79.pth.tar'

    model_unet = None
    if args.AEtransfer:
        model_unet = models.create("UNetAuto", num_channels=3,
                                   batch_size=args.batch_size, max_features=1024)
        model_unet.cuda()
        checkpoint = load_checkpoint(args.AEtransfer)
        copy_state_dict(checkpoint['state_dict'], model_unet)
        print("=> Unet load \n -> {0}"
              .format(args.AEtransfer))

    trainer = PreTrainer(model, num_classes, margin=args.margin, AEtransfer=model_unet)

    print('---------- Training Start ----------')
    print(args.logs_dir)
    # Start training
    for epoch in range(start_epoch, args.epochs):
        
        train_loader_source.new_epoch()
        train_loader_target.new_epoch()

        start = time.time()
        trainer.train(epoch, train_loader_source, train_loader_target, optimizer,
                    train_iters=len(train_loader_source), print_freq=args.print_freq)
        lr_scheduler.step()
        print(' {0:.3f} seconds \n{1}'.format(time.time() - start, args.logs_dir))

        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
            start = time.time()

            _, mAP = evaluator.evaluate(test_loader_source, dataset_source.query, dataset_source.gallery, cmc_flag=True)

            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  source mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

            print(' {0:.3f} seconds'.format(time.time() - start))


    print('---------------Training End------------')

    print("Test on target domain:")
    evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True, rerank=args.rerank)

    print('---------------Testing End------------')
    output_directory_complete = '{0}_{1:.2f}'.format(args.logs_dir[:-8], best_mAP * 100)
    os.rename(args.logs_dir, output_directory_complete)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-training on the source domain")
    # data
    parser.add_argument('-ds', '--dataset-source', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-dt', '--dataset-target', type=str, default='dukemtmc',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=32)
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
    parser.add_argument('--AEtransfer', type=str, default=None)

    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70], help='milestones for the learning rate decay')
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
                        default ='/data2/syupoh/dataset/')
                        # default=osp.join(os.getcwd(), 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(os.getcwd(), 'logs'))
    # parser.add_argument('--logs-dir', type=str,
    #                     default=None)
    main()
