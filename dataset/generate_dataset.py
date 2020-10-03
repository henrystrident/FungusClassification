import torch.utils.data as Data
from PIL import Image
from torchvision.transforms import transforms


class mushroom_dataset(Data.Dataset):
    """
    菌类数据集，返回图片和标签
    """
    def __init__(self, mode:str, count:int, img_size:int):
        """
        将文件中的所有路径和标签读到列表中
        :params mode: 模式
        """

        super(mushroom_dataset, self).__init__()
        assert mode in ["train", "val"]
        self.__mode=  mode
        self.__file_path = "/home/pgj/MushroomClassification/"+self.__mode+"-"+count.__str__()+".txt"
        self.__data_list = []
        self.__img_size = img_size

        with open(self.__file_path, "r") as f:
            for line in f.readlines():
                if line != "\n":
                    img_path, img_label = line.split(" ")
                    self.__data_list.append((img_path, int(img_label)))

    def __getitem__(self, item):
        data = self.__data_list[item]
        img_path, label = data
        img = Image.open(img_path).convert("RGB")
        transform = transforms.Compose([transforms.Resize((self.__img_size, self.__img_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        img = transform(img)
        return img, label

    def __len__(self):
        return len(self.__data_list)
