#coding=utf-8

import skimage.io as skio
import os
import numpy as np 
from time import time
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

#图片根目录
img_root=''

def mean_std_per_img(img_path):
    img = skio.imread(img_path)
    if len(img.shape)==3 and img.shape[2] ==4:
        #如果是RGBA通道，取前3个通道
        img=img[..., :3]
    mean = [img[..., i].mean() for i in range(3)]
    std = [img[..., i].std() for i in range(3)]
    return mean, std

def parallel_compute_mean_std(img_root, num_workers):
    pool=ThreadPoolExecutor(num_workers)

    if os.path.isdir(img_root):
        tasks = [pool.submit(mean_std_per_img,(d.path)) for d in os.scandir(img_root)]
        dones, not_dones = wait(tasks, return_when=ALL_COMPLETED)
        #注意设置axis
        mean, std = np.mean([future.result() for future in dones], axis=0)
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
