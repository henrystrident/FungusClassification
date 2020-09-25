"""
@author: pgj
@time: 2020/9/25 9:51 上午
@usage: 进行自定义的随机数生成
"""


import random


def generate_random_index(rate: list):
    """
    以某个概率列表进行加权随机生成数字
    :params rate: 加权列表，元素为整数，每个元素代表区间长度，或者概率比例
    :return: 下标
    """
    random_num = random.randrange(0, sum(rate))
    split_index = 0 #  区间分隔符
    for index, scope in enumerate(rate):
        split_index += scope
        if random_num <= split_index:
            return index
