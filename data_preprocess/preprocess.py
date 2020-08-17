import os
from data_preprocess.category import mushroom
import cv2
import random


original_path = "/home/pgj/PycharmProjects/FungusClassification/original_img"
preprocessed_path = "/home/pgj/PycharmProjects/FungusClassification/preprocessed_img"

category_names = ["agaric", "bolete", "discomycete", "lichen"]


def preprocess_train_data():
    with open("/home/pgj/PycharmProjects/FungusClassification/train.txt", "w") as train_file:
        for category_name in category_names:
            category = mushroom(category_name, original_path)
            avg_height, avg_width = category.avg_height, category.avg_width  # 获取平均宽度，高度
            img_dir = os.path.join(original_path, "train/"+category_name)  # 原始数据存放文件夹
            save_dir = os.path.join(preprocessed_path, "train/"+category_name)  # 经过处理的数据存放文件夹
            if not os.path.exists(save_dir):  # 创建存放文件夹
                os.makedirs(save_dir)

            for file in os.listdir(img_dir):
                img_path = os.path.join(img_dir, file)  # 原图片存放路径
                img = cv2.imread(img_path)
                width, height = img.shape[0:2]

                if width < avg_width or height < avg_height:
                    resized_img = cv2.resize(img, (avg_width, avg_height))
                else:
                    resized_img = img
                save_path = os.path.join(save_dir, file)
                print(save_path)
                train_file.write(save_path+" "+category_names.index(category_name).__str__()+"\n")
                cv2.imwrite(save_path, resized_img)

                if category_name in ["agaric", "bolete"]:
                    if random.random()>0.5:
                        fliped_img = cv2.flip(resized_img, 1)
                        save_path = os.path.join(save_dir, "filped_"+file)
                        print(save_path)
                        train_file.write(save_path + " " + category_names.index(category_name).__str__() + "\n")
                        cv2.imwrite(save_path, fliped_img)

                elif category_name == "discomycete":
                    fliped_img = cv2.flip(resized_img, 1)
                    save_path = os.path.join(save_dir, "filped_"+file)
                    print(save_path)
                    train_file.write(save_path + " " + category_names.index(category_name).__str__() + "\n")
                    cv2.imwrite(save_path, fliped_img)


def generate_val_list():
    with open("/home/pgj/PycharmProjects/FungusClassification/val.txt", "w") as val_list:
        for category_name in category_names:
            img_dir = os.path.join(original_path, "val/" + category_name)
            save_dir = os.path.join(preprocessed_path+"/val", category_name)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            for val_img in os.listdir(img_dir):
                img_path = os.path.join(img_dir, val_img)
                save_path = os.path.join(save_dir, val_img)
                img = cv2.imread(img_path)
                cv2.imwrite(save_path, img)
                val_list.write(save_path+" "+category_names.index(category_name).__str__()+"\n")




if __name__ == '__main__':
    pass