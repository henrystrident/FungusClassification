"""
@author: pgj
@time: 2020/9/22 4:44 下午
@usage: 进行一些想法的验证
"""

from dataset.generate_dataset import mushroom_dataset
from model.VGG import vgg19_bn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import random
import torch
import numpy as np
import torch.nn as nn


if __name__ == '__main__':
    dataset = mushroom_dataset("train")
    dataloader = DataLoader(dataset=dataset,
                            batch_size=16,
                            shuffle=True,
                            num_workers=16,
                            pin_memory=True)
    model = vgg19_bn(num_classes=4,
                     init_weights=False)
    model.to(torch.device("cuda"))
    criterion = nn.CrossEntropyLoss()
    for step, (imgs, labels) in enumerate(dataloader):
        imgs = imgs.to(torch.device("cuda"))
        labels = labels.to(torch.device("cuda"))
        loss = criterion(model(imgs), labels)
        print(step, loss)
