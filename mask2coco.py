#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File        : mask2coco.py
@Date        : 2020/02/21
@Author      : shine young
@Contact     : x0601y@126.com
@License     : Copyright(C), Wukong-Inc.
@Description : Convert camelyon to binary mask ,then convert them into coco format json file.
'''
import os
import re
import glob
import json
import fnmatch
import datetime
import cv2
import numpy as np
import random
import collections
from PIL import Image
from pycococreatortools import pycococreatortools
from pycocotools import mask

from tqdm import tqdm

import multiprocess
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import itertools
'''COCO 公共信息

'''
INFO = {
    "description": "Camelyon 16 Dataset",
    "url": "https://github.com/nipeone/data-preparation/mask2coco.py",
    "version": "0.0.1",
    "year": 2020,
    "contributor": "nipeone",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": ""
    }
]

# 根据自己的需要添加种类
CATEGORIES = [
    # {
    #     'id': 0,
    #     'name': '_background',
    #     'supercategory': '_background',
    # },
    {
        'id': 1,
        'name': 'tumor',
        'supercategory': 'tumor',
    },
        {
        'id': 2,
        'name': 'normal',
        'supercategory': 'normal',
    }
]


# def rgb2binary(label_name):
#     # convert one rgb-mask to multiple binary masks
#     lbl_id = os.path.split(label_name)[-1].split('.')[0]
#     lbl = cv2.imread(label_name, 1)
#     h, w = lbl.shape[:2]
#     leaf_dict = {}
#     idx = 0
#     white_mask = np.ones((h, w, 3), dtype=np.uint8) * 255
#     for i in range(h):
#         for j in range(w):
#             if tuple(lbl[i][j]) in leaf_dict or tuple(lbl[i][j]) == (0, 0, 0):
#                 continue
#             leaf_dict[tuple(lbl[i][j])] = idx
#             mask = (lbl == lbl[i][j]).all(-1)
#             # leaf = lbl * mask[..., None]      # rgb-mask with black background
#             # np.repeat(mask[...,None],3,axis=2)    # 3D mask
#             leaf = np.where(mask[..., None], white_mask, 0)
#             mask_name = './shapes/train/annotations/' + lbl_id + '_leaf_' + str(idx) + '.png'
#             cv2.imwrite(mask_name, leaf)
#             idx += 1


def filter_for_image(root, files):
    file_types = ['*.jpeg', '*.jpg', '*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    return files


def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
    return files

class MultiProcessCounter():
    ''' 多进程 计数器
        param: ini 
            计数器的初始值
    '''
    def __init__(self, ini=0):
        self.val = multiprocess.Value('i', ini)

    def increment(self, delta=1):
        with self.val.get_lock():
            self.val.value += delta
            return self.val.value
    
    @property
    def value(self):
        with self.val.get_lock():
            return self.val.value

def worker_create_annotation_info(image_filename, image_id):
    image = Image.open(image_filename)
    image_info = pycococreatortools.create_image_info(image_id, os.path.basename(image_filename), image.size)
    #多进程计数器必须标明global
    global ann_counter
    # filter for associated png annotations
    annotation_infos=[]
    for root, _, files in os.walk(MASK_DIR):
        annotation_files = filter_for_annotations(root, files, image_filename)
        # go through each associated annotation
        for annotation_filename in annotation_files:
            class_id = [x['id'] for x in CATEGORIES if x['name'] in os.path.basename(annotation_filename)][0]                
            binary_mask = np.asarray(Image.open(annotation_filename).convert('1')).astype(np.uint8)

            category_info = {'id': class_id, 'is_crowd': 0}
            annotation_info = pycococreatortools.create_annotation_info(
                    ann_counter.increment(), image_id, category_info, binary_mask,
                    image.size, tolerance=2)

            if annotation_info is not None:
                annotation_infos.append(annotation_info)

    return image_info, annotation_infos


def parallel_gen_coco(num_workers):
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    #负样本比例太大，所以减小负样本数量
    normal_image_infos={}
    normal_annotation_infos={}
    # ann_counter=itertools.count().__next__
    
    with ProcessPoolExecutor(num_workers) as pool:
        for root, _, files in os.walk(IMAGE_DIR):
            image_files = filter_for_image(root, files)
            # 如果是肿瘤图片，直接添加。
            # 如果是正常图片，先收拢一起，然后随机采样和肿瘤图片一样数量的，保正负样本均衡。
            tumor_image_files = [image_filename for image_filename in image_files if 'tumor' in os.path.basename(image_filename)]
            normal_image_files = random.sample([image_filename for image_filename in image_files if 'normal' in os.path.basename(image_filename)],len(tumor_image_files))

            print(f"tumor nums:{len(tumor_image_files)},normal nums:{len(normal_image_files)}")

            tasks = [pool.submit(worker_create_annotation_info, *(image_filename, image_id+1)) for image_id, image_filename in enumerate(tumor_image_files+normal_image_files)]
            for future in tqdm(as_completed(tasks), total=len(tasks)):
                try:
                    image_info, annotation_infos = future.result()
                    coco_output["images"].append(image_info)
                    coco_output["annotations"].extend(annotation_infos)
                except Exception as e:
                    print(f"{e}")

    with open(f'{ROOT_DIR}/instances_camelyon16_train2020.json', 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


def gen_coco():
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    # filter for jpeg images
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_image(root, files)

            # 如果是肿瘤图片，直接添加。
            # 如果是正常图片，先收拢一起，然后随机采样和肿瘤图片一样数量的，保正负样本均衡。
        tumor_image_files = [image_filename for image_filename in image_files if 'tumor' in os.path.basename(image_filename)]
        normal_image_files = random.sample([image_filename for image_filename in image_files if 'normal' in os.path.basename(image_filename)],len(tumor_image_files))


        # go through each image
        for image_filename in tqdm(tumor_image_files+normal_image_files):
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                    image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(MASK_DIR):
                annotation_files = filter_for_annotations(root, files, image_filename)

                # go through each associated annotation
                for annotation_filename in annotation_files:

                    class_id = [x['id'] for x in CATEGORIES if x['name'] in os.path.basename(annotation_filename)][0]                
                    binary_mask = np.asarray(Image.open(annotation_filename).convert('1')).astype(np.uint8)

                    # class_id = 1 if binary_mask.sum() > 0 else 0
                    category_info = {'id': class_id, 'is_crowd': 0}
                    annotation_info = pycococreatortools.create_annotation_info(
                            segmentation_id, image_id, category_info, binary_mask,
                            image.size, tolerance=2)

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1        
            image_id = image_id + 1

    with open('{}/instances_camelyon16_train2020.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    # mask_dir = '/media/disk1/camelyon/train/masks'
    # label_list = glob.glob(os.path.join(mask_dir, '*.png'))
    # for label_name in label_list:
    #     rgb2binary(label_name)

    # ROOT_DIR = '/media/disk1/camelyon/train'
    ann_counter = MultiProcessCounter(0)
    ROOT_DIR = ''
    IMAGE_DIR = os.path.join(ROOT_DIR, "images")
    MASK_DIR = os.path.join(ROOT_DIR, "masks")

    parallel_gen_coco(8)
    # gen_coco()
