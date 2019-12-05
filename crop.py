# coding=utf-8

import os
import math
import skimage.io as skio


'''
    裁剪图片索引    x轴起始位置    
    0               0
    1               width-overlap
    2               (width-overlap)+width-overlap
    .
    .
    .
    m               (width-overlap)*m

    (width-overlap)*m + width <= w
    m <= (w-width)/(width-overlap)
    n <= (h-height)/(height-overlap)
'''
def crop(filepath, width, height, overlap):
    # 获取文件所在路径
    file_dir = os.path.dirname(filepath)
    filename,ext = os.path.splitext(os.path.basename(filepath))
    # 创建裁剪图片存储路径
    crop_save_dir = os.path.join(file_dir, "crops",filename)
    if not os.path.exists(crop_save_dir):
        os.makedirs(crop_save_dir)

    img = skio.imread(filepath)
    h, w = img.shape[:2]

    m = math.ceil((w-width)/(width-overlap))+1
    n = math.ceil((h-height)/(height-overlap))+1
    for i in range(m):
        for j in range(n):
            x_ = (width-overlap)*i
            y_ = (height-overlap)*j
            if(x_+width) > w:
                x_ = w-width
            if(y_+height) > h:
                y_ = h-height
            crop_img = img[y_:y_+height, x_:x_+width, ...]
            skio.imsave(os.path.join(
                crop_save_dir, f"{m}_{n}_{i}_{j}_{x_}_{y_}_{width}_{height}{ext}"), crop_img, quality=95, dpi=(300, 300))


def main():
    img_dir = "./images"
    crop("./nuclei_seg_pred/images/1900714B16_x_32768_y_24576.jpg",1024,1024,128)


if __name__ == '__main__':
    main()
