"""
@author: pgj
@time: 2020/9/28 9:39 上午
@usage: 将训练集与测试集比例调整至7:3， 训练集467张，测试集200张
"""

from data_preprocess.train_val_list import generate_file_list
from utils.random import generate_random_index
import os
import cv2
import random


root_path = "/mnt/data/pgj/funguses"  # 所有训练图片的存放路径，下面有四个类别
save_root = "/mnt/data/pgj/funguses-467"  # 经过处理后的训练图片存放路径
category_dict = {"agaric": 0,
                 "bolete": 1,
                 "discomycete": 2,
                 "lichen": 3}


def generate_photo_dir(mode: str, category: str):
    """
    生成每个类别的训练和测试集
    :param mode: 训练还是测试
    :param category: 菌物种类
    :return:
    """
    origin_img_dir = os.path.join(root_path+"/"+mode, category)  # 该菌种的训练或测试原图片保存路径
    save_dir = os.path.join(save_root+"/"+mode, category)  # 该菌种经过预训练的训练或测试图片的保存路径


    # 如果保存文件夹不存在，则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for img_name in os.listdir(origin_img_dir):
        origin_img_path = os.path.join(origin_img_dir, img_name)
        origin_img = cv2.imread(origin_img_path)  # 从原文件夹进行读取
        if mode == "train":
            index = generate_random_index([467, len(os.listdir(os.path.join(root_path+"/"+mode, category)))-467])  # 从各个类别的原始文件夹读取照片总数，进行随机选取
            if index == 0:
                if random.random() > 0.5:
                    origin_img = cv2.flip(origin_img, 1)  # 进行随机翻转
                cv2.imwrite(os.path.join(save_dir, img_name), origin_img)
        else:
            cv2.imwrite(os.path.join(save_dir, img_name), origin_img)

    print(category+" "+mode+" has "+len(os.listdir(save_dir)).__str__())


def save_photo():
    for category in category_dict:
        generate_photo_dir("train", category)
        generate_photo_dir("val", category)




if __name__ == '__main__':
    pass
