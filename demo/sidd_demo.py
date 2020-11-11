import os
import sys
from os.path import dirname as p
sys.path.append(p(p(p(os.path.abspath(__file__)))))

from pysiddtools import SIDD, format_dataset


BASE_DIR = "/algo/data_office/SIDD/SIDD_Full_Dataset"


if __name__ == "__main__":
    sidd = SIDD(BASE_DIR)
    print(sidd[0])
    print(len(sidd))

    # 筛选符合条件的子集----手机：S6
    s6 = sidd.filter(smartphone="S6")
    print(len(s6))
    print(s6.img_num())

    # 筛选符合条件的子集----手机：S6并且为训练集
    # 由于SIDD隐藏了一部分数据集用于benchmark
    s6_visible = sidd.filter(visible=True, smartphone="S6")
    print(len(s6_visible))
    print(s6_visible.img_num())

    # 获取raw图像
    raw1 = s6[0].gt_raw(0)
    print(raw1.shape)

    # 获取raw图像，转换bayer pattern
    raw2 = s6[0].gt_raw(0, pattern="BGGR")
    print(raw2.shape)

    # 获取raw图像，转换bayer pattern，分割通道
    raw3 = s6[0].gt_raw(0, pattern="BGGR", split_channel=True)
    print(raw3.shape)
