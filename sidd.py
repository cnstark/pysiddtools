import os
from typing import List

import numpy as np
import h5py
from scipy.io import loadmat
import cv2

from .bayer_unify_aug import bayer_unify


BAYER_PATTERN = {
    "GP": "BGGR",
    "IP": "RGGB",
    "S6": "GRBG",
    "N6": "BGGR",
    "G4": "BGGR",
}


CURRENT_DIR = os.path.dirname(__file__)
SCENE_INSTANCE_PATH = os.path.join(CURRENT_DIR, "scene_instance.txt")


def get_scene_instance(path: str) -> List[str]:
    with open(path, "r") as f:
        scene_instance_list = f.read().splitlines()
    return scene_instance_list


def read_raw(path: str) -> np.ndarray:
    assert path.split(".")[-1] in ["MAT", "mat"], \
        "Please give correct raw path"
    data = h5py.File(path)
    return np.array(data["x"])


def read_rgb(path: str) -> np.ndarray:
    assert path.split(".")[-1] in ["PNG", "png"], \
        "Please give correct img path"
    return cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)


class SIDDSceneInstance:
    def __init__(self, base_dir: str, scene_instance_name: str):
        self.scene_instance_name = scene_instance_name
        values = self.scene_instance_name.split("_")
        self.scene_instance_id = values[0]
        self.scene_id = values[1]
        self.smartphone = values[2]
        self.iso = int(values[3])
        self.shutter_speed = int(values[4])
        self.cct = int(values[5])
        self.luminance = values[6]
        self.bayer = BAYER_PATTERN[self.smartphone]

        self._noisy_raw_dir = os.path.join(base_dir, self.scene_instance_id + "_NOISY_RAW")
        self._gt_raw_dir = os.path.join(base_dir, self.scene_instance_id + "_GT_RAW")
        self._noisy_rgb_dir = os.path.join(base_dir, self.scene_instance_id + "_NOISY_SRGB")
        self._gt_rgb_dir = os.path.join(base_dir, self.scene_instance_id + "_GT_SRGB")
        self._metadata_dir = os.path.join(base_dir, self.scene_instance_id + "_METADATA_RAW")

        self.visible = os.path.isdir(self._metadata_dir)

        if self.visible:
            noisy_raw_num = len(os.listdir(self._noisy_raw_dir))
            gt_raw_num = len(os.listdir(self._gt_raw_dir))
            noisy_rgb_num = len(os.listdir(self._noisy_rgb_dir))
            gt_rgb_num = len(os.listdir(self._gt_rgb_dir))

            assert noisy_raw_num == gt_raw_num == noisy_rgb_num == gt_rgb_num, \
                "SIDD Dataset is not complete"
            
            self.img_num = noisy_raw_num
        else:
            self.img_num = 0

    def match(self, visible: bool=None, scene_id: str or List[str]=None, \
            smartphone: str or List[str]=None, iso: int or List[int]=None, \
            cct: int or List[int]=None, luminance: str or List[str]=None):
        if type(scene_id) == str:
            scene_id = [scene_id]

        if type(smartphone) == str:
            smartphone = [smartphone]

        if type(iso) == int:
            iso = [iso]

        if type(cct) == int:
            cct = [cct]

        if type(luminance) == str:
            luminance = [luminance]

        return (visible is None or self.visible == visible) \
            and (scene_id is None or self.scene_id in scene_id) \
            and (smartphone is None or self.smartphone in smartphone) \
            and (iso is None or self.iso in iso) \
            and (cct is None or self.cct in cct) \
            and (luminance is None or self.luminance in luminance)

    def noisy_raw(self, index: int, pattern: str=None, mode: str="crop"):
        assert self.visible, "This scene instance is held for benchmark"
        path = os.path.join(
            self._noisy_raw_dir,
            self.scene_instance_id + "_NOISY_RAW_" + str(index + 1).zfill(3) + ".MAT"
        )
        raw = read_raw(path)
        if pattern is not None and pattern != self.bayer:
            raw = bayer_unify(raw, self.bayer, pattern, mode)
        return raw

    def gt_raw(self, index: int, pattern: str=None, mode: str="crop"):
        assert self.visible, "This scene instance is held for benchmark"
        path = os.path.join(
            self._gt_raw_dir,
            self.scene_instance_id + "_GT_RAW_" + str(index + 1).zfill(3) + ".MAT"
        )
        raw = read_raw(path)
        if pattern is not None and pattern != self.bayer:
            raw = bayer_unify(raw, self.bayer, pattern, mode)
        return raw

    def noisy_rgb(self, index: int):
        assert self.visible, "This scene instance is held for benchmark"
        path = os.path.join(
            self._noisy_rgb_dir,
            self.scene_instance_id + "_NOISY_SRGB_" + str(index + 1).zfill(3) + ".PNG"
        )
        return read_rgb(path)

    def gt_rgb(self, index: int):
        assert self.visible, "This scene instance is held for benchmark"
        path = os.path.join(
            self._gt_rgb_dir,
            self.scene_instance_id + "_GT_SRGB_" + str(index + 1).zfill(3) + ".PNG"
        )
        return read_rgb(path)

    # TODO(wangyuhao): metadata

    def __str__(self):
        return "SIDDSceneInstance<" + self.scene_instance_name + ">"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return self.img_num


class SIDD(list):
    def __init__(self, base_dir: str, copy_from: list=None):
        list.__init__([])
        self.base_dir = base_dir

        if copy_from is None:
            scene_instance_name_list = get_scene_instance(SCENE_INSTANCE_PATH)

            self.extend([
                SIDDSceneInstance(self.base_dir, scene_instance_name) \
                for scene_instance_name in scene_instance_name_list
            ])
        else:
            for i in copy_from:
                assert type(i) == SIDDSceneInstance, \
                    "Param copy_from must be a list of SIDDSceneInstance"
            self.extend(copy_from)

    def filter(self, visible: bool=None, scene_id: str or List[str]=None, \
            smartphone: str or List[str]=None, iso: int or List[int]=None, \
            cct: int or List[int]=None, luminance: str or List[str]=None):
        return SIDD(self.base_dir, copy_from=list(filter(
            lambda scene_instance: scene_instance.match(visible, scene_id, smartphone, iso, cct, luminance),
            self
        )))

    def img_num(self):
        return sum(map(len, self))
