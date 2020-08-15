import os
import cv2

class mushroom:
    def __init__(self, category: str, path: str):
        assert category in ["agaric", "bolete", "discomycete", "lichen"]

        self.__category = category
        self.__root_path = path
        self.__train_count = 0
        self.__val_count = 0
        self.__avg_height = 0
        self.__avg_width = 0
        self.__max_height = 0
        self.__max_width = 0
        self.__train_path = os.path.join(self.__root_path+"/train", self.__category+"/")
        self.__val_path = os.path.join(self.__root_path+"/val", self.__category+"/")

        self.update()

    @property
    def category(self):
        return self.__category


    @property
    def foot_path(self):
        return self.__root_path

    @property
    def train_count(self):
        return self.__train_count

    @property
    def val_count(self):
        return self.__val_count

    @property
    def avg_height(self):
        return self.__avg_height

    @property
    def avg_width(self):
        return self.__avg_width

    @property
    def max_height(self):
        return self.__max_height

    @property
    def max_width(self):
        return self.__max_width

    @property
    def train_path(self):
        return self.__train_path

    @property
    def val_path(self):
        return self.__val_path

    def update(self):
        """
        更新本类别目前训练集的情况，如图片数量，图片尺寸等。
        :return:
        """
        for train_img_path in os.listdir(self.train_path):
            self.__train_count += 1
            img = cv2.imread(os.path.join(self.train_path, train_img_path)) # cv2读取图片格式为(width, height, channel)
            width, height = img.shape[0], img.shape[1]
            self.__max_width = max(self.__max_width, width)
            self.__max_height = max(self.__max_height, height)
            self.__avg_width += width
            self.__avg_height += height
        self.__avg_height = round(self.__avg_height/self.__train_count)
        self.__avg_width = round(self.__avg_width/self.__train_count)

        for val_img in os.listdir(self.val_path):
            self.__val_count += 1

