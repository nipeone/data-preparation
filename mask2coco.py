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
from PIL import Image
from pycococreatortools import pycococreatortools
from pycocotools import mask

from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

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

# def create_annotation_info(annotation_id, image_id, category_info, binary_mask, 
#                            image_size=None, tolerance=2, bounding_box=None):

#     if image_size is not None:
#         binary_mask = pycococreatortools.resize_binary_mask(binary_mask, image_size)

#     binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

#     area = mask.area(binary_mask_encoded)
#     #此处需要改造，针对camelyon normal类图片，
#     if area < 1:
#         return None

#     if bounding_box is None:
#         bounding_box = mask.toBbox(binary_mask_encoded)

#     if category_info["is_crowd"]:
#         is_crowd = 1
#         segmentation = binary_mask_to_rle(binary_mask)
#     else :
#         is_crowd = 0
#         segmentation = binary_mask_to_polygon(binary_mask, tolerance)
#         if not segmentation:
#             return None

#     annotation_info = {
#         "id": annotation_id,
#         "image_id": image_id,
#         "category_id": category_info["id"],
#         "iscrowd": is_crowd,
#         "area": area.tolist(),
#         "bbox": bounding_box.tolist(),
#         "segmentation": segmentation,
#         "width": binary_mask.shape[1],
#         "height": binary_mask.shape[0],
#     } 

#     return annotation_info

def worker_create_annotation_info(image_filename, image_id)
    image = Image.open(image_filename)
    image_info = pycococreatortools.create_image_info(image_id, os.path.basename(image_filename), image.size)

    # filter for associated png annotations
    annotation_infos=[]
    for root, _, files in os.walk(MASK_DIR):
        annotation_files = filter_for_annotations(root, files, image_filename)
        # go through each associated annotation
        for segmentation_id, annotation_filename in enumerate(annotation_files):
            class_id = [x['id'] for x in CATEGORIES if x['name'] in os.path.basename(annotation_filename)][0]                
            binary_mask = np.asarray(Image.open(annotation_filename).convert('1')).astype(np.uint8)

            category_info = {'id': class_id, 'is_crowd': 0}
            annotation_info = pycococreatortools.create_annotation_info(
                    segmentation_id+1, image_id, category_info, binary_mask,
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
    with ThreadPoolExecutor(num_workers) as pool:
        for root, _, files in os.walk(IMAGE_DIR):
            image_files = filter_for_image(root, files)
            tasks = [pool.submit(worker_create_annotation_info,(image_filename, image_id+1) for image_id, image_filename in enumerate(image_files)]
            for future in tqdm(as_completed(tasks), len(tasks)):
                try:
                    image_info, annotation_infos = future.result()
                    coco_output["images"].append(image_info)
                    coco_output["annotations"].extend(annotation_infos)
                except Exception as e:
                    print(f"{e}")

    with open(f'{ROOT_DIR}/instances_camelyon16_train2020.json', 'w') as output_json_file:
        json.dump(coco_output, output_json_file)

if __name__ == "__main__":
    # mask_dir = '/media/disk1/camelyon/train/masks'
    # label_list = glob.glob(os.path.join(mask_dir, '*.png'))
    # for label_name in label_list:
    #     rgb2binary(label_name)

    ROOT_DIR = ''
    IMAGE_DIR = os.path.join(ROOT_DIR, "images")
    MASK_DIR = os.path.join(ROOT_DIR, "masks")

    parallel_gen_coco(8)
