import os
import re
import json

from PIL import Image
import PIL.Image
try:
    import pyspng
except ImportError:
    pyspng = None

def extract_number(filename,flag=0):
    """提取文件名中的数字部分"""
    if flag==0:
        match = re.search(r'img(\d+)', filename)
    else:
        match = re.search(r'-std-(\d+)', filename)
    return match.group(1) if match else None

def check_correspondence(image_fnames, feature_fnames):
    """检查 image 和 feature 的文件是否一一对应"""
    # 提取数字部分
    image_numbers = [extract_number(fname,0) for fname in image_fnames]
    feature_numbers = [extract_number(fname,1) for fname in feature_fnames]
    # 检查是否一一对应
    for i in range(len(image_numbers)):
        if image_numbers[i] != feature_numbers[i]:
            print(image_fnames[i], feature_fnames[i])
    return image_numbers == feature_numbers

def _file_ext(fname):
    return os.path.splitext(fname)[1].lower()
if __name__=="__main__":
    data_dir="/home/t2vg-a100-G4-42/t2vgusw2/videos/imagenet/sd_latents/REPA_256"
    PIL.Image.init()
    supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}

    images_dir = os.path.join(data_dir, 'images')
    features_dir = os.path.join(data_dir, 'vae-sd')
    # images

    image_json_path=os.path.join(images_dir, 'dataset.json')
    print("image_json_path",image_json_path)
    with open( image_json_path, 'r') as f:
        image_data = json.load(f)
    all_labels = image_data['labels']
    image_fnames, labels = [], []

    for item in all_labels:
        fname, label = item
        if _file_ext(fname) in supported_ext:  # 检查文件扩展名是否受支持
            image_fnames.append(fname)  # 相对路径
            labels.append(label)
    # 转换为绝对路径并排序
    image_fnames = [os.path.join(images_dir, fname) for fname in image_fnames]

    # features
    feature_json_path=os.path.join(features_dir, 'dataset.json')
    with open( feature_json_path, 'r') as f:
        feature_data = json.load(f)
    all_labels = feature_data['labels']
    feature_fnames = []
    for item in all_labels:
        fname, label = item
        if _file_ext(fname) in supported_ext:
            feature_fnames.append(fname)
    # 转换为绝对路径并排序
    feature_fnames = [os.path.join(features_dir, fname) for fname in feature_fnames]
    print(1)
    check_correspondence(image_fnames, feature_fnames)