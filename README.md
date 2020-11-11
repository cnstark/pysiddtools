# PySIDDTools

SIDD数据集工具集

## 简介

TODO

## 使用方法

* 实例化SIDD对象

```python
BASE_DIR = "<Path to SIDD Full Dataset>"
sidd = SIDD(BASE_DIR)
```

* 获取SceneInstance个数

```python
len(sidd)
```

* 获取SceneInstance

使用index获取

```python
sidd[0]
```

* 筛选SceneInstance

更多筛选条件见sidd.py代码

```python
s6_visible_L = sidd.filter(visible=True, smartphone="S6", luminance="L")
```

* 读取raw图像

```python
gt_0 = s6_visible_L[0].gt_raw(0)
gt_0_bggr = s6_visible_L[0].gt_raw(0, pattern="BGGR")

noisy_0 = s6_visible_L[0].noisy_raw(0)
noisy_0_rggb = s6_visible_L[0].noisy_raw(0, pattern="RGGB")
noisy_0_channels = s6_visible_L[0].noisy_raw(0, pattern="RGGB", split_channel=True)
```

* 读取rbg图像

```python
gt_0 = s6_visible_L[0].gt_srgb(0)
noisy_0 = s6_visible_L[0].noisy_srgb(0)
```
