import os
import re
import json

import torch
from torch.utils.data import Dataset
import numpy as np

from PIL import Image
import PIL.Image
try:
    import pyspng
except ImportError:
    pyspng = None


class CustomDataset(Dataset):
    def __init__(self, data_dir):
        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}

        self.images_dir = os.path.join(data_dir, 'images')
        self.features_dir = os.path.join(data_dir, 'vae-sd')

        # images
        image_json_path=os.path.join(self.images_dir, 'dataset.json')
        with open( image_json_path, 'r') as f:
            image_data = json.load(f)
        all_labels = image_data['labels']
        self.image_fnames, self.labels = [], []

        for item in all_labels:
            fname, label = item
            if self._file_ext(fname) in supported_ext:  # 检查文件扩展名是否受支持
                self.image_fnames.append(fname)  # 相对路径
                self.labels.append(label)
        # 转换为绝对路径并排序
        self.image_fnames = [os.path.join(self.images_dir, fname) for fname in self.image_fnames]
        self.labels = np.array(self.labels).astype({1: np.int64, 2: np.float32}[np.array(self.labels).ndim])

        # features
        feature_json_path=os.path.join(self.features_dir, 'dataset.json')
        with open( feature_json_path, 'r') as f:
            feature_data = json.load(f)
        all_labels = feature_data['labels']
        self.feature_fnames = []
        for item in all_labels:
            fname, label = item
            if self._file_ext(fname) in supported_ext:
                self.feature_fnames.append(fname)
        # 转换为绝对路径并排序
        self.feature_fnames = [os.path.join(self.features_dir, fname) for fname in self.feature_fnames]
        assert len(self.image_fnames) == len(self.feature_fnames), "Mismatch between images and features"


    
    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def __len__(self):
        assert len(self.image_fnames) == len(self.feature_fnames), \
            "Number of feature files and label files should be same"
        return len(self.feature_fnames)

    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        feature_fname = self.feature_fnames[idx]
        image_ext = self._file_ext(image_fname)
        with open(image_fname, 'rb') as f:
            if image_ext == '.npy':
                image = np.load(f)
                image = image.reshape(-1, *image.shape[-2:])
            elif image_ext == '.png' and pyspng is not None:
                image = pyspng.load(f.read())
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
            else:
                image = np.array(PIL.Image.open(f))
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)

        features = np.load(os.path.join(self.features_dir, feature_fname))
        return torch.from_numpy(image), torch.from_numpy(features), torch.tensor(self.labels[idx])