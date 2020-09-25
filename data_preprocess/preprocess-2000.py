"""
@author: pgj
@time: 2020/9/25 9:57 上午
@usage: 将所有类别的训练集图片调整到2000张左右，其中agaric和bolete选取一半进行随机翻转，discomycete全部翻转，lichen随机翻转
"""


import os
import cv2
from utils.random import generate_random_index

root_path = "/mnt/data/pgj/funguses"  # 所有训练图片的存放路径，下面有四个类别
save_root = "/mnt/data/pgj/funguses-2000"  # 经过处理后的训练图片存放路径
category_dict = {"agaric": 0,
                 "bolete": 1,
                 "discomycete": 2,
                 "lichen": 3}


def save_photo():
    for category in category_dict:
        generate_photo_dir("train", category)
        generate_photo_dir("val", category)


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
        cv2.imwrite(os.path.join(save_dir, img_name), origin_img)  # 直接保存原图
        # 判断类别，不同类别进行不同的数据预处理

        if (category == "agaric" or category == "bolete") and mode == "train":
            index = generate_random_index([50, 50])  # 判断是否需要翻转，翻转与不翻转的比例是1:1
            if index == 0:
                fliped_img = cv2.flip(origin_img, 1)  # 水平翻转
                cv2.imwrite(os.path.join(save_dir, "fliped_"+img_name), fliped_img)  # 保存水平翻转的图片，名称为fliped_原名

        if category == "discomycete" and mode == "train":
            fliped_img = cv2.flip(origin_img, 1)  # 全部进行水平翻转
            cv2.imwrite(os.path.join(save_dir, "fliped_" + img_name), fliped_img)  # 保存水平翻转的图片，名称为fliped_原名

    print(category+" "+mode+" has "+len(os.listdir(save_dir)).__str__())


def generate_file_list(mode: str):
    """
    根据处理好的图片进行训练和预测列表的生成
    :return:
    """
    file_path = "/home/pgj/MushroomClassification/" + mode + "-2000.txt"
    with open(file_path, "w") as f:
        for category in category_dict:
            img_dir = os.path.join(save_root+"/"+mode, category)
            for img_name in os.listdir(img_dir):
                img_path = os.path.join(img_dir, img_name)
                f.write(img_path+" "+category_dict[category].__str__()+"\n")



if __name__ == '__main__':
    pass
