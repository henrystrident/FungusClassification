"""
@author: pgj
@time: 2020/9/21 10:22 上午
@usage: 从每个类别中随机挑选500张图片做为训练集，进行随机的翻转
"""


import os
import cv2
import random



root_path = "/mnt/data/pgj/funguses" #  所有训练图片的存放路径，下面有四个类别
save_root = "/mnt/data/pgj/funguses-500"
category_dict = {"agaric": 0,
                 "bolete": 1,
                 "discomycete": 2,
                 "lichen": 3}


def generate_random_index(rate: list):
    """
    以某个概率列表进行加权随机生成数字
    :param rate: 加权列表，元素为整数
    :return: 下标
    """
    random_num = random.randrange(0, sum(rate))
    split_index = 0 #  区间分隔符
    for index, scope in enumerate(rate):
        split_index += scope
        if random_num <= split_index:
            return index


def generate_data_dir(mode: str, count: int, category: str):
    """
    根据参数生成对应的图片集
    :param mode: 训练还是测试
    :param count: 图片集中图片的数量
    :param category: 菌物种类
    :return:
    """
    save_dir = os.path.join(save_root+"/"+mode, category)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    origin_dir = os.path.join(root_path+"/"+mode, category)
    for image_name in os.listdir(origin_dir):
        if generate_random_index([count ,len(os.listdir(origin_dir))-count]) == 0:
            save_path = os.path.join(save_dir, image_name)
            img = cv2.imread(os.path.join(origin_dir, image_name))
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
            cv2.imwrite(save_path, img)



def save_photo():
    """
    单个类别进行保存测试，生成新的训练集，500张
    opencv打开图片格式：h*w*c
    :return:
    """

    for category in category_dict:
        generate_data_dir("train", 500, category)
        generate_data_dir("val", 50 ,category)




def generate_file_list(mode: str):
    """
    根据训练和测试集生成训练和测试列表
    :param mode: 训练还是测试
    :return:
    """
    file_path = "/home/pgj/MushroomClassification/"+mode+".txt"
    with open(file_path, "w") as file:
        for category in category_dict:
            img_dir = os.path.join(save_root+"/"+mode, category)
            for img_name in os.listdir(img_dir):
                file.write(os.path.join(img_dir, img_name)+" "+category_dict[category].__str__()+"\n")


if __name__ == '__main__':
    generate_file_list("val")