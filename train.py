#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: cifar10-preact18-mixup.py
# Author: Tao Hu <taohu620@gmail.com>,  Yauheni Selivonchyk <y.selivonchyk@gmail.com>

import numpy as np
import argparse
import os
import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils.summary import *
from tensorpack.dataflow import dataset

from dataloader import ImageFromDir
from model import ResNet_Cifar

BATCH_SIZE = 32
CLASS_NUM = 10

LR_SCHEDULE = [(0, 0.01), (10, 0.001), (20, 0.0001)]
WEIGHT_DECAY = 1e-4

# FILTER_SIZES = [64, 128, 256, 512]
FILTER_SIZES = [32, 64, 128, 256]
MODULE_SIZES = [2, 2, 2, 2]


def get_data(train_or_test, isMixup, alpha):
    isTrain = train_or_test == 'train'
    if isTrain:
        ds = ImageFromDir(dir="D:/facedatasetzoo/fer2013/train",channel=1,resize=100, shuffle=isTrain)
    else:
        ds = ImageFromDir(dir="D:/facedatasetzoo/fer2013/test", channel=1, resize=100, shuffle=isTrain)
    if isTrain:
        augmentors = [
            imgaug.RandomCrop((96, 96)),
            imgaug.Flip(horiz=True),
        ]
    else:
        augmentors = [
            imgaug.CenterCrop((96, 96)),
        ]
    ds = AugmentImageComponent(ds, augmentors)

    batch = BATCH_SIZE
    ds = BatchData(ds, batch, remainder=not isTrain)

    def f(dp):
        images, labels = dp
        one_hot_labels = np.eye(CLASS_NUM)[labels]  # one hot coding
        if not isTrain or not isMixup:
            return [images, one_hot_labels]

        # mixup implementation:
        # Note that for larger images, it's more efficient to do mixup on GPUs (i.e. in the graph)
        weight = np.random.beta(alpha, alpha, BATCH_SIZE)
        x_weight = weight.reshape(BATCH_SIZE, 1, 1, 1)
        y_weight = weight.reshape(BATCH_SIZE, 1)
        index = np.random.permutation(BATCH_SIZE)

        x1, x2 = images, images[index]
        x = x1 * x_weight + x2 * (1 - x_weight)
        y1, y2 = one_hot_labels, one_hot_labels[index]
        y = y1 * y_weight + y2 * (1 - y_weight)
        return [x, y]

    ds = MapData(ds, f)
    #ds = PrefetchData(ds,2,4)
    return ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--mixup', help='enable mixup', action='store_true')
    parser.add_argument('--alpha', default=1, type=float, help='alpha in mixup')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    log_folder = 'train_log/fer-preact18%s' % ('-mixup' if args.mixup else '')
    logger.set_logger_dir(os.path.join(log_folder))

    dataset_train = get_data('train', args.mixup, args.alpha)
    dataset_test = get_data('test', args.mixup, args.alpha)

    config = TrainConfig(
        model=ResNet_Cifar(),
        data=QueueInput(dataset_train),
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            [ScalarStats('cost'), ClassificationError('wrong_vector')]),
            ScheduledHyperParamSetter('learning_rate', LR_SCHEDULE)
        ],
        max_epoch=200,
        steps_per_epoch=len(dataset_train),
        session_init=SaverRestore(args.load) if args.load else None
    )
    launch_train_with_config(config, SimpleTrainer())