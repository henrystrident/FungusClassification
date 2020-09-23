"""
@author: pgj
@time: 2020/9/22 4:44 下午
@usage: 进行一些想法的验证
"""

from dataset.generate_dataset import mushroom_dataset
from model.VGG import vgg19_bn
from torch.utils.data import DataLoader
import torch


if __name__ == '__main__':
    dataset = mushroom_dataset("train")
    dataloader = DataLoader(dataset=dataset,
                            batch_size=16,
                            shuffle=True,
                            num_workers=16,
                            pin_memory=True)
    model = vgg19_bn(num_classes=4,
                     init_weights=False)
    with torch.set_grad_enabled(False):
        model.eval()
        model.to(torch.device("cuda"))
        for step, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.to(torch.device("cuda"))
            print(imgs)
            print(labels)
            labels = labels.to(torch.device("cuda"))
            outputs = model(imgs)
            p = torch.softmax(outputs, dim=1)
            print(p)
            print(torch.max(p, dim=1).indices.reshape(-1, 1))
            break
