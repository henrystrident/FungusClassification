"""
@author: pgj
@time: 2020/9/28 9:31 上午
@usage: 生成训练和测试列表
"""


import os


def generate_file_list(mode: str, count:int, save_root:str, category_dict:dict):
    """
    根据处理好的数据集生成训和测试列表
    :param mode: 训练/测试
    :param count: 训练集数量
    :param save_root: 图片保存位置
    :param category_dict: 种类字典
    :return:
    """
    file_path = "/home/pgj/MushroomClassification/" + mode + "-" +count.__str__() +".txt"
    with open(file_path, "w") as f:
        for category in category_dict:
            img_dir = os.path.join(save_root+"/"+mode, category)
            for img_name in os.listdir(img_dir):
                img_path = os.path.join(img_dir, img_name)
                f.write(img_path+" "+category_dict[category].__str__()+"\n")
