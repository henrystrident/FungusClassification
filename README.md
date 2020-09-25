# READ ME  
本文件记录了菌物种类识别这个项目的大体流程，主要工作以及注意事项，是第一次尝试用Markdown对一个项目进行记录，后续有需要完善的地方还会继续改进。



# 目录
- [远程环境的部署](#远程环境的部署)
	- [frpc的重新连接](#frpc的重新连接) 
	- [远程环境部署](#远程环境部署)
- [数据预处理](#数据预处理)
- [训练](#训练)



# [远程环境的部署](#目录)

## frpc的重新连接
实验室暑假里由于安全问题关闭过所有服务器，我的电脑重启之后发现原来的frpc客户端我发正常开启，所以重新安装了一次。一开始下载成 `arm_64` 版本的了，无法启动，正确版本应当是 `amd_64`。当前版本的frpc需要先通过 `chmod+x <文件名>` 授予权限，开启后输入密钥，选择隧道IP即可使用。

## 远程环境部署
- 通过Pycharm的远程部署功能直接在服务器端新建项目，要注意的是 `tools -> development -> Configuraion` 中需要选择**mappings**有远程路径的那个连接。
- 本次项目的环境：Ubuntu18.04 + Pytorch1.6-GPU + Pycharm + CUDA10.0（Pytorch要求10.1，就目前来看10.0也能用）



# [数据预处理](#目录)
本次的原始数据共有两个部分，`train` 文件夹和 `val` 文件夹，里面各有四种类别的图片，数据分布如下：

|       | agaric | bolete | discomycete | lichen |
| :---: | :----: | :----: | :---------: | :----: |
| train |  1322  |  1327  |     852     |  1974  |
|  val  |  200   |  200   |     200     |  200   |

可以看到测试集数据分布很均衡，不用做调整，但是训练集存在的问题是前三个类别相较于第四个类别数据偏少，且第三个类别最少。可以看到 `agaric` 和 `bolete` 与 `lichen` 的比值约为1:1.5，`discomycete` 与 `lichen` 的比值约为1:2.3。数据调整的策略是将 `agaric` 和 `bolete` 类别中一半的数据进行随机翻转，`lichen` 中所有数据进行水平翻转。同时，生成训练和验证列表 `train.txt`，`val.txt`。经过调整后，各类数据如下：
|       | agaric | bolete | discomycete | lichen |
| :---: | :----: | :----: | :---------: | :----: |
| train |  1980  |  1975  |    1704     |  1974  |
|  val  |  200   |  200   |     200     |  200   |



# [训练](#目录)

## 数据集

本次实验数据集继承自 `torch.utils.data.Dataset`，从 `train.txt` 和 `val.txt` 读入图片路径和对应的标签，并将图片resize为 **224*224**。

## 训练过程
各项超参数：

|      | 学习率 | 正则化系数 | batch_size | 训练轮数 |
| :--: | :----: | :--------: | :--------: | :------: |
| 数据 |  1e-4  |    1e-5    |     64     |   100    |

本次训练采用了分类中常用的交叉熵损失函数，分类用softmax计算，优化器采用 `Adam`。在每轮训练完成后都会进行一次预测，并取当前最优模型进行保存。

## 训练结果
本次训练结果并不理想，由于数据较多，可能是读取数据的速度较低造成显卡占用率低，训练速度较慢，并且从前几轮的情况来看，训练的准确率很低。同时由于电源问题，电脑自动关机，训练并没有完成。