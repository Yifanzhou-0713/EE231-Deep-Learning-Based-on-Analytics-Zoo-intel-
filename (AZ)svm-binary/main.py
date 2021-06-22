#!/usr/bin/env python

from __future__ import print_function
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import load_iris
import pandas as pd

from torch.utils.tensorboard import SummaryWriter

from zoo.orca import init_orca_context, stop_orca_context
from zoo.orca.learn.pytorch import Estimator
from zoo.orca.learn.metrics import Accuracy
from zoo.orca.learn.trigger import EveryEpoch


import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset,DataLoader,TensorDataset

class Dataset(Dataset):
    def __init__(self,x,y):
        # self.x = x.unsqueeze(dim=1)
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]



def train_data_creator(config, batchsize):
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[.5], std=[.5])])
    
    n_data = torch.ones(100, 2)
    x0 = torch.normal(2 * n_data, 1)
    y0 = torch.zeros(100)
    x1 = torch.normal(-2 * n_data, 1)
    y1 = torch.ones(100)
    x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
    y = torch.cat((y0, y1)).type(torch.LongTensor)
    
    
    train_dataset = TensorDataset(x,y)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batchsize,shuffle=True)

    return train_loader
    
    
def validation_data_creator(config, batchsize):
    val_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[.5], std=[.5])])

    val_dataset = DataClass(root='./data',split='test',transform=val_transform,download=False)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batchsize,shuffle=True)
    
    return val_loader


def optimizer_creator(model,config):
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    return optimizer

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
 
    def forward(self, x_input):
        x_hidden = F.relu(self.hidden(x_input))
        x_predict = self.predict(x_hidden)
        return x_predict

def model_creator(config):
    model = Net(2,10,2)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batchsize", type=int, default=5)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--cluster_mode", default="local")
    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    
    batchsize = args.batchsize
    epoch = args.epoch
    

    criterion = nn.CrossEntropyLoss()

    if args.cluster_mode == "local":
        init_orca_context()
    elif args.cluster_mode == "standalone":
    	init_orca_context(cluster_mode=args.cluster_mode, cores=1,memory='512M', num_nodes=2, driver_cores=2, driver_memory="1g",python_location="/home/yifan/anaconda3/envs/zoo/bin/python")
    
    orca_estimator = Estimator.from_torch(model=model_creator,
                                              optimizer=optimizer_creator,
                                              loss=criterion,
                                              metrics=[Accuracy()],
                                              use_tqdm=True,
                                              backend="torch_distributed")
    stats = orca_estimator.fit(train_data_creator, epochs=args.epoch, batch_size=args.batchsize)


    print("Train stats: {}".format(stats))

    orca_estimator.shutdown()
    stop_orca_context()

