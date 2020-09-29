"""
@author: pgj
@time: 2020/9/23 3:28 下午
@usage: 在测试集上进行测试
"""

from dataset.generate_dataset import mushroom_dataset
from model.VGG import vgg19_bn
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils.model import swich_model
import os


class Test:
    def __init__(self, count:int, model_category:str, model_name:str):
        self.__count = count
        self.__model = swich_model(model_category)
        self.__model_path = os.path.join("/home/pgj/MushroomClassification/best_model", model_name)
        self.__model.load_state_dict(torch.load(self.__model_path))
        self.__model.to(torch.device("cuda"))

        self.__data_set = mushroom_dataset(mode="val",
                                           count=self.__count)
        self.__data_loader = DataLoader(dataset=self.__data_set,
                                 batch_size=1,
                                 shuffle=True,
                                 num_workers=16,
                                 pin_memory=True)
        self.__acc_count = 0

    def inference_total(self):
        with torch.set_grad_enabled(False):
            self.__acc_count = 0
            self.__model.eval()
            for step, (img, label) in enumerate(self.__data_loader):
                img = img.to(torch.device("cuda"))
                label = label.to(torch.device("cuda"))
                predict = torch.max(torch.softmax(self.__model(img), dim=1), dim=1).indices
                self.__acc_count += (predict == label).sum().item()

            print("total acc:", self.__acc_count/self.__data_set.__len__())


    def category_inference(self):
        """
        统计每个类别的
        :return:
        """
        category_list = {0: "agaric",
                         1: "bolete",
                         2: "discomycete",
                         3: "lichen"}
        acc = {0: 0,
               1: 0,
               2: 0,
               3: 0}

        total = {0: 0,
                 1: 0,
                 2: 0,
                 3: 0}
        with torch.set_grad_enabled(False):
            for step, (img, label) in enumerate(self.__data_loader):
                img = img.to(torch.device("cuda"))
                label = label.to(torch.device("cuda")).item()
                predict = torch.max(torch.softmax(self.__model(img), dim=1), dim=1).indices.item()
                total[label] += 1
                if predict == label:
                    acc[predict] += 1
            for category in category_list:
                print(category_list[category]+":"+ (acc[category]/total[category]).__str__())




if __name__ == '__main__':
    test = Test(count=2000,
                model_category="mobileNet",
                model_name="mobileNet_2000.pth")
    test.inference_total()
    test.category_inference()
