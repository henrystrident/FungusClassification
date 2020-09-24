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


if __name__ == '__main__':
    model = vgg19_bn(pretrained=False,
                     progress=False,
                     num_classes=4,
                     init_weights=False)
    model.load_state_dict(torch.load("/home/pgj/MushroomClassification/params/epoch4.pth"))


    acc_count=0

    with torch.set_grad_enabled(False):
        model.to(torch.device("cuda"))
        model.eval()

        dataset = mushroom_dataset("train")
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=1,
                                 shuffle=True,
                                 num_workers=16,
                                 pin_memory=True)
        for step, (img, label) in enumerate(data_loader):
            img = img.to(torch.device("cuda"))
            label = label.to(torch.device("cuda"))
            predict = torch.max(torch.softmax(model(img), dim=1), dim=1).indices
            acc_count += (predict == label).sum().item()
    print(acc_count/dataset.__len__())