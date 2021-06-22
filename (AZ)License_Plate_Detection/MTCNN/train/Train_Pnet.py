import sys
sys.path.append('..')
import os 
import torch
from torch.utils.data import Dataset
from Data_Loading import ListDataset
from model.MTCNN_nets import PNet
import time
import copy
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from collections import OrderedDict
import argparse

from zoo.orca import init_orca_context, stop_orca_context
from zoo.orca.learn.pytorch import Estimator
from zoo.orca.learn.metrics import Accuracy
from zoo.orca.learn.trigger import EveryEpoch

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0.1)


# load the model and weights for initialization


class Pnet(nn.Module):

    def __init__(self):
        super(Pnet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 10, 3, 1)),
            ('prelu1', nn.PReLU(10)),
            ('pool1', nn.MaxPool2d((2,5), ceil_mode=True)),

            ('conv2', nn.Conv2d(10, 16, (3,5), 1)),
            ('prelu2', nn.PReLU(16)),

            ('conv3', nn.Conv2d(16, 32, (3,5), 1)),
            ('prelu3', nn.PReLU(32))
        ]))

        self.conv4_1 = nn.Conv2d(32, 2, 1, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            b: a float tensor with shape [batch_size, 4, h', w'].
            a: a float tensor with shape [batch_size, 2, h', w'].
        """
        x = self.features(x)
        a = self.conv4_1(x)
        b = self.conv4_2(x)

        return b


def model_creator(config):
    model = Pnet()
    #print("Pnet loaded")
    return model


def optimizer_creator(model, config):
    optimizer = torch.optim.Adam(model.parameters())
    return optimizer

def train_data_creator(config, batch_size):

    trainloader = torch.utils.data.DataLoader(ListDataset('../data_preprocessing/anno_store/imglist_anno_12.txt'), batch_size=batch_size,
                                              shuffle=True)
    return trainloader


def validation_data_creator(config, batch_size):


    testloader = torch.utils.data.DataLoader(ListDataset('../data_preprocessing/anno_store/imglist_anno_12_val.txt'), batch_size=batch_size,
                                             shuffle=True)
    return testloader



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--cluster_mode", default="local")
    args = parser.parse_args()

    train_path = '../data_preprocessing/anno_store/imglist_anno_12.txt'
    val_path = '../data_preprocessing/anno_store/imglist_anno_12_val.txt'
 
    
    if args.cluster_mode == "local":
        init_orca_context()
    elif args.cluster_mode == "standalone":
    	init_orca_context(cluster_mode=args.cluster_mode, cores=1,memory='512M', num_nodes=2, driver_cores=2, driver_memory="1g",python_location="/home/yifan/anaconda3/envs/zoo/bin/python")
    	
    criterion = nn.MSELoss()
    batch_size = args.batchsize
    num_epochs = args.epoch
    orca_estimator = Estimator.from_torch(model=model_creator,
                                              optimizer=optimizer_creator,
                                              loss=criterion,
                                              metrics=[Accuracy()],
                                              use_tqdm=True,
                                              backend="torch_distributed",
                                              )
    stats = orca_estimator.fit(train_data_creator, epochs=num_epochs, batch_size=batch_size)
    orca_estimator.save('weights/pnet_Weights.pth')
    print("Train stats: {}".format(stats))
    val_stats = orca_estimator.evaluate(validation_data_creator, batch_size=batch_size)
    print("Validation stats: {}".format(val_stats))
    orca_estimator.shutdown()
    stop_orca_context()

if __name__ == '__main__':
    main()
