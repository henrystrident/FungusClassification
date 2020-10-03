"""
@author: pgj
@time: 2020/9/30 10:07 上午
@usage: 通过模型名称获取模型
"""

from model.MobileNet import mobilenet_v2
from model.VGG import vgg19_bn
from model.ResNet import resnet50
from model.Inception import inception_v3


def swich_model(model: str):
    assert model in ["vgg", "mobileNet", "resnet", "inception"]

    if model == "vgg":
        return vgg19_bn(pretrained=True,
                        progress=False,
                        num_classes=4,
                        init_weights=False)
    elif model == "mobileNet":
        return mobilenet_v2(pretrained=True,
                            progress=True,
                            num_classes=4,
                            width_mult=1.0,
                            inverted_residual_setting=None,
                            round_nearest=8,
                            block=None,
                            norm_layer=None)

    elif model == "resnet":
        return resnet50(pretrained=True,
                        progress=True,
                        num_classes=4)

    elif model == "inception":
        return inception_v3(pretrained=True,
                            progress=True,
                            num_classes=4,
                            aux_logits=True)
