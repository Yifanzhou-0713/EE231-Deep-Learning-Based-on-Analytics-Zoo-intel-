#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from model.LPRNET import CHARS
from model.STN import STNet
from data.load_data import LPRDataLoader, collate_fn
from Evaluation import eval, decode
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import argparse
import time

from zoo.orca import init_orca_context, stop_orca_context
from zoo.orca.learn.pytorch import Estimator
from zoo.orca.learn.metrics import Accuracy
from zoo.orca.learn.trigger import EveryEpoch


def model_creator(config):
    model = STNet()
    model.load_state_dict(torch.load('weights/STN_model_Init.pth', map_location=lambda storage, loc: storage))
    return model


def optimizer_creator(model, config):
    optimizer = torch.optim.Adam(model.parameters())
    return optimizer
    
    
def train_data_creator(config, batch_size):
    trainset = LPRDataLoader(["./data/train"], (94, 24))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=False,collate_fn=collate_fn)
    return trainloader


def validation_data_creator(config, batch_size):
    testset = LPRDataLoader(["./data/validation"], (94, 24))
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False,collate_fn=collate_fn)
    return testloader


def main():
    parser = argparse.ArgumentParser(description='SNT Training')
    parser.add_argument('--img_size', default=(94, 24), help='the image size')
    parser.add_argument('--img_dirs_train', default="./data/train", help='the training images path')
    parser.add_argument('--img_dirs_val', default="./data/validation", help='the validation images path')
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    parser.add_argument('--epoch', type=int, default=25, help='number of epoches for training')
    parser.add_argument('--batch_size', default=32, help='batch size')
    parser.add_argument('--cluster_mode',default='local')
    args = parser.parse_args()
    

    #criterion = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') # reduction: 'none' | 'mean' | 'sum' 
    criterion = nn.CrossEntropyLoss()
    
    if args.cluster_mode == "local":
        init_orca_context()
    elif args.cluster_mode == "standalone":
    	init_orca_context(cluster_mode=args.cluster_mode, cores=1,memory='512M', num_nodes=2, driver_cores=2, driver_memory="1g",python_location="/home/yifan/anaconda3/envs/zoo/bin/python")
    
    batch_size = 128
    epochs = 200
    orca_estimator = Estimator.from_torch(model=model_creator,
    					   optimizer=optimizer_creator,
    					   loss=criterion,
    					   metrics=[Accuracy()],
    					   use_tqdm=True,
    					   backend="torch_distributed")
    stats = orca_estimator.fit(train_data_creator, epochs=epochs, batch_size=batch_size)
    print("Train stats: {}".format(stats))
    orca_estimator.save('weights/model_STNet.pth')
    val_stats = orca_estimator.evaluate(validation_data_creator, batch_size=batch_size)
    print("Validation stats: {}".format(val_stats))
    orca_estimator.shutdown()


    stop_orca_context()
    


if __name__ == '__main__':
    main()
