import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import time
import os
from dataset.generate_dataset import mushroom_dataset
from model.VGG import vgg19, vgg19_bn

logging.getLogger().setLevel(logging.INFO)

default_save_dir = "/home/pgj/PycharmProjects/FungusClassification/params"


class Trainer:
    def __init__(self, max_epoch:int, batch_size=64 ,lr=1e-4, weight_decay=1e-3, save_path=default_save_dir):

        self.__max_epoch = max_epoch
        self.__batch_size = batch_size
        self.__lr = lr
        self.__weight_decay = weight_decay

        self.__datasets = {x: mushroom_dataset(x) for x in ["train", "val"]}
        self.__dataloaders = {x: DataLoader(dataset=self.__datasets[x],
                                            batch_size=self.__batch_size if x == "train" else 1,
                                            shuffle=True,
                                            num_workers=8,
                                            pin_memory=True
                                            )
                       for x in ["train", "val"]}

        self.__train_loader = self.__dataloaders["train"]
        self.__val_loader = self.__dataloaders["val"]

        self.__net = vgg19_bn(pretrained=True,
                              progress=True,
                              num_classes=4,
                              init_weights=False)

        self.__criterion = nn.CrossEntropyLoss()

        self.__optimizer = optim.Adam(params=self.__net.parameters(),
                                      lr=self.__lr,
                                      weight_decay=self.__weight_decay)

        self.__best_acc = -1

        self.__save_path = save_path

    def epoch_train(self):
        start = time.time()
        with torch.set_grad_enabled(True):
            for step, (imgs, labels) in enumerate(self.__train_loader):
                outputs = self.__net(imgs)
                loss = self.__criterion(outputs, labels)
                loss.backward()
                self.__optimizer.step()
        return time.time()-start

    def epoch_val(self, epoch):
        start = time.time()
        acc_count = 0
        with torch.set_grad_enabled(False):
            with torch.set_grad_enabled(False):
                for step, (imgs, labels) in enumerate(self.__val_loader):
                    predict = nn.functional.softmax(self.__net(imgs), dim=1)
                    acc_count += (predict == labels).sum().item()
            logging.info("epoch {}, eval_acc {:.3f}, eval cost {:.3f}s".format(epoch + 1, acc_count / self.__datasets["train"].__len__(),
                                                           time.time() - start))
            if acc_count > self.__best_acc:
                self.__best_acc = acc_count
                model_state_dic = self.__net.state_dict()
                torch.save(model_state_dic, os.path.join(self.__save_path, "epoch+1"+str(epoch)+".pth"))
                logging.info("epoch {} is best".format(epoch+1))



    def train(self):
        for epoch in range(self.__max_epoch):
            logging.info('-' * 20 + 'Epoch {}/{}'.format(epoch + 1, self.__max_epoch) + '-' * 20)
            train_time = self.epoch_train()
            logging.info("epoch {} costs {:.3f}s".format(epoch+1, train_time))
            self.epoch_val(epoch)
