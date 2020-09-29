import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
import time
import os
from dataset.generate_dataset import mushroom_dataset
from utils.model import swich_model
from model.VGG import vgg19, vgg19_bn
from model.MobileNet import mobilenet_v2
import numpy as np

logging.getLogger().setLevel(logging.INFO)

default_save_dir = "/home/pgj/MushroomClassification/params"


class Trainer:
    def __init__(self, max_epoch:int, batch_size=32 ,count=2000, model="vgg" , lr=3e-4, weight_decay=1e-3, save_path=default_save_dir):

        self.__max_epoch = max_epoch
        self.__batch_size = batch_size
        self.__lr = lr
        self.__weight_decay = weight_decay

        self.__datasets = {x: mushroom_dataset(x, count) for x in ["train", "val"]}
        self.__dataloaders = {x: DataLoader(dataset=self.__datasets[x],
                                            batch_size=self.__batch_size if x == "train" else 1,
                                            shuffle=True,
                                            num_workers=16,
                                            pin_memory=True
                                            )
                       for x in ["train", "val"]}

        self.__train_loader = self.__dataloaders["train"]
        self.__val_loader = self.__dataloaders["val"]

        self.__net = swich_model(model)

        self.__device = torch.device("cuda")
        self.__net.to(self.__device)

        self.__criterion = nn.CrossEntropyLoss()

        # self.__optimizer = optim.Adam(params=self.__net.parameters(),
        #                               lr=self.__lr,
        #                               weight_decay=self.__weight_decay)

        self.__optimizer = optim.SGD(params=self.__net.parameters(),
                                     lr=self.__lr,
                                     momentum=0.9)

        self.__best_acc = -1

        self.__save_path = save_path

        self.__writer = SummaryWriter()

    def epoch_train(self, epoch):
        self.__net.train()
        start = time.time()
        with torch.set_grad_enabled(True):
            for step, (imgs, labels) in enumerate(self.__train_loader):
                self.__optimizer.zero_grad()
                imgs = imgs.to(self.__device)
                labels = labels.to(self.__device)
                outputs = self.__net(imgs)
                loss = self.__criterion(outputs, labels)
                loss.backward()
                self.__optimizer.step()
            self.__writer.add_scalar("Loss/train", np.array(loss.to(torch.device("cpu")).data).item(), epoch)
        return time.time()-start

    def epoch_val(self, epoch):
        with torch.set_grad_enabled(False):
            self.__net.eval()
            start = time.time()
            acc_count = 0
            with torch.set_grad_enabled(False):
                for step, (imgs, labels) in enumerate(self.__val_loader):
                    imgs = imgs.to(self.__device)
                    labels = labels.to(self.__device)
                    predict = torch.max(nn.functional.softmax(self.__net(imgs), dim=1), dim=1).indices.reshape(-1, 1)
                    acc_count += (predict == labels).sum().item()
            self.__writer.add_scalar("Val/train", acc_count, epoch)

            logging.info("epoch {}, eval_acc {:.3f}, eval cost {:.3f}s".format(epoch + 1, acc_count / self.__datasets["val"].__len__(),
                                                           time.time() - start))
            if acc_count > self.__best_acc:
                self.__best_acc = acc_count
                model_state_dic = self.__net.state_dict()
                torch.save(model_state_dic, os.path.join(self.__save_path, "epoch"+str(epoch+1)+".pth"))
                logging.info("epoch {} is best".format(epoch+1))



    def train(self):
        for epoch in range(self.__max_epoch):
            logging.info('-' * 20 + 'Epoch {}/{}'.format(epoch + 1, self.__max_epoch) + '-' * 20)
            train_time = self.epoch_train(epoch)
            logging.info("epoch {} costs {:.3f}s".format(epoch+1, train_time))
            self.epoch_val(epoch)
