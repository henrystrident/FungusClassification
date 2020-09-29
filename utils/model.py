"""
@author: pgj
@time: 2020/9/29 2:50 下午
@usage: 选择模型
"""

from model.MobileNet import mobilenet_v2
from model.VGG import vgg19_bn


def swich_model(model: str):
    assert model in ["vgg", "mobileNet"]

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
