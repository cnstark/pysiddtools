import math

import numpy as np
import torch
import torch.nn.functional as F


def get_raw_channel(src: torch.Tensor or np.ndarray, index: int):
    """
    获取raw单通道
    输入: (..., h, w)
    输出: (..., h / 2, w / 2)
    :param src: input
    :param index: channel index（0 - 3）
    """
    return src[..., index // 2::2, index % 2::2]


def pack_raw(src: torch.Tensor or np.ndarray):
    """
    分割raw通道
    输入: (..., h, w)
    输出: (..., 4, h / 2, w / 2)
    :param src: input
    """
    if isinstance(src, torch.Tensor):
        return torch.cat(
            [get_raw_channel(src, i).unsqueeze(src.ndim - 2) for i in range(4)],
            axis=src.ndim - 2
        )
    else:
        return np.concatenate(
            [np.expand_dims(get_raw_channel(src, i), src.ndim - 2) for i in range(4)],
            axis=src.ndim - 2
        )


def unpack_raw(src: torch.Tensor or np.ndarray):
    """
    合并raw通道
    输入: (..., 4, h / 2, w / 2)
    输出: (..., h, w)
    :param src: input
    """
    assert src.ndim >= 3
    assert src.shape[-3] == 4

    w = src.shape[-1]
    h = src.shape[-2]
    raw_shape = list(src.shape[0:-3])
    raw_shape.append(h * 2)
    raw_shape.append(w * 2)

    if isinstance(src, torch.Tensor):
        raw = torch.empty(raw_shape, dtype = src.dtype, device = src.device)
    else:
        raw = np.zeros(raw_shape, dtype = src.dtype)
    for i in range(4):
        raw[..., i // 2::2, i % 2::2] = src[..., i, :, :]
    return raw


def raw_align_up(src: torch.Tensor or np.ndarray, align_up_to: int):
    """
    将raw图像的h和w对齐为align_up_to的倍数
    输入: (..., h, w)
    输出: (..., h_align, w_align)
    :param src: input
    :param align_up_to: align up to (example 4)
    """
    assert 2 <= src.ndim <= 4

    pad_h = math.ceil(src.shape[-2] / align_up_to) * align_up_to - src.shape[-2]
    pad_w = math.ceil(src.shape[-1] / align_up_to) * align_up_to - src.shape[-1]

    assert pad_h % 2 == 0 and pad_w % 2 == 0

    if isinstance(src, torch.Tensor):
        ndim = src.ndim
        while src.ndim < 4:
            src = src.unsqueeze(0)
        pad = F.pad(src, pad=(0, pad_w, 0, pad_h), mode='reflect')
        while pad.ndim > ndim:
            pad = pad.squeeze(0)
        return pad
    else:
        pad_shape = [(0, 0) for _ in range(src.ndim - 2)]
        pad_shape.append((0, pad_h))
        pad_shape.append((0, pad_w))
        return np.pad(src, pad_shape, 'reflect')


def raw_crop(src: torch.Tensor or np.ndarray, \
        h_end: int, w_end:int, h_start: int=0, w_start: int=0):
    """
    对raw图像的h和w维进行裁剪
    输入: (..., h, w)
    输出: (..., h_end - h_start, w_end - w_start)
    :param src: input
    :param h_end: end of h
    :param w_end: end of w
    :param h_start: start of h (default 0)
    :param w_start: start of w (default 0)
    """
    return src[..., h_start:h_end, w_start:w_end]
