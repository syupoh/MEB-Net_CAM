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

from mebnet import datasets
from mebnet import models
from mebnet.evaluators import Evaluator
from mebnet.utils.data import transforms as T
from mebnet.utils.data.preprocessor import Preprocessor
from mebnet.utils.logging import Logger
from mebnet.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict


def get_data(name, data_dir, height, width, batch_size, workers):
    root = osp.join(data_dir)
    # root = osp.join(data_dir, name)

    dataset = datasets.create(name, root)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, test_loader


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    cudnn.benchmark = True

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(args.gpu)

    log_dir = osp.dirname(args.resume)
    sys.stdout = Logger(osp.join(log_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # args.resume = 'logs/dukemtmc_market1501/densenet_inceptionv3_resnet50_2021-06-03T19:26_ongoing'

    # Create data loaders
    dataset_target, test_loader_target = \
        get_data(args.dataset_target, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers)

    # Create model
    model = models.create(args.arch, pretrained=False, num_features=args.features, dropout=args.dropout, num_classes=0)
    model.cuda()
    # model = nn.DataParallel(model)

    # checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    # model_ema_list[0].load_state_dict(checkpoint['state_dict'])
    # evaluator_ema_list[0].evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)

    # Load from checkpoint
    checkpoint = load_checkpoint(osp.join(args.resume, 'model_best.pth.tar'))
    copy_state_dict(checkpoint['state_dict'], model)
    start_epoch = checkpoint['epoch']
    best_mAP = checkpoint['best_mAP']
    print("=> Checkpoint of epoch {}  best mAP {:.1%}".format(start_epoch, best_mAP))

    # Evaluator
    evaluator = Evaluator(model)
    print("Test on the target domain of {}:".format(args.dataset_target))
    evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True, rerank=args.rerank)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing the model")
    # data
    parser.add_argument('-dt', '--dataset-target', type=str, required=True,
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    # model
    parser.add_argument('-a', '--arch', type=str, required=True,
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # testing configs
    parser.add_argument('--gpu', type=int, default='0',
                        help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--resume', type=str, required=True, metavar='PATH')
    parser.add_argument('--rerank', action='store_true',
                        help="evaluation only")
    parser.add_argument('--seed', type=int, default=1)
    # path
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default ='/data2/syupoh/dataset/')
                        # default=osp.join(os.getcwd(), 'data'))
    main()

