#coding=utf-8

import os
import skimage.io as skio
import skimage.transform as sktrans

IMG_SIZE=1024
IMG_FOLDERS=["1908216B"]
# IMG_ROOT="1908923A\\jpg"

def main():
    for IMG_FOLDER in IMG_FOLDERS:
        IMG_ROOT = os.path.join(IMG_FOLDER,"jpg")
        if not os.path.exists(os.path.join(os.path.dirname(IMG_ROOT),"resized")):
            os.mkdir(os.path.join(os.path.dirname(IMG_ROOT),"resized"))

        for filename in os.listdir(IMG_ROOT):
            img=skio.imread(os.path.join(IMG_ROOT, filename))
            dst=sktrans.resize(img, (IMG_SIZE, IMG_SIZE),mode='constant')
            skio.imsave(os.path.join(os.path.dirname(IMG_ROOT),"resized",filename),dst,quality=95,dpi=(300,300))


if __name__ == '__main__':
    main()
    

