#!/usr/bin/env python
#coding=utf-8
'''
@File        : compute_mean.py
@Date        : 2020/02/21
@Author      : shine young
@Contact     : x0601y@126.com
@License     : Copyright(C), Wukong-Inc.
'''
import skimage.io as skio
import os
import numpy as np 
from time import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, as_completed

#图片根目录
img_root = ''

def mean_std_per_img(img_path):
    img = skio.imread(img_path)
    if len(img.shape)==3 and img.shape[2] ==4:
        #如果是RGBA通道，取前3个通道
        img=img[..., :3]
    mean = [img[..., i].mean() for i in range(3)]
    std = [img[..., i].std() for i in range(3)]
    return mean, std

def parallel_compute_mean_std(img_root, num_workers):
    with ThreadPoolExecutor(num_workers) as pool:
        if os.path.isdir(img_root):
            tasks = [pool.submit(mean_std_per_img,(d.path)) for d in os.scandir(img_root)]
            # dones, not_dones = wait(tasks, return_when=ALL_COMPLETED)
            #注意设置axis
            try:
                mean, std = np.mean([future.result() for future in tqdm(as_completed(tasks),total=len(tasks))], axis=0)
            except Exception as e:
                print(f'{e}')
            return mean, std
        elif os.path.isfile(img_root):
            return mean_std_per_img(img_root)
        else:
            return [0., 0., 0.], [0., 0., 0.]

if __name__ == "__main__":
    t1=time()
    mean,std = parallel_compute_mean_std(img_root, 8)
    print(f"cost time:{time()-t1}")
    print(f"MEAN_PIXEL = {mean}")
    print(f"STD_PIXEL = {std}")
