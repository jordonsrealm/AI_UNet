import os
import numpy as np
from PIL import Image
from natsort import natsorted


def convert_one_channel(img):
    # some images have 3 channels , although they are grayscale image
    if len(img.shape) > 2:
        img = img[:, :, 0]
        return img
    else:
        return img


def pre_masks(resize_shape, path):
    dirs = natsorted(os.listdir(path))
    masks = img = Image.open(path+dirs[0])
    masks = (masks.resize(resize_shape, Image.ANTIALIAS))
    masks = convert_one_channel(np.asarray(masks))
    for i in range(1, len(dirs)):
        img = Image.open(path+dirs[i])
        img = img.resize(resize_shape, Image.ANTIALIAS)
        img = convert_one_channel(np.asarray(img))
        masks = np.concatenate((masks, img))
    masks = np.reshape(masks, (len(dirs), resize_shape[0], resize_shape[1], 1))
    return masks


# CustomMasks 512x512
def pre_splitted_masks(path):
    dirs = natsorted(os.listdir(path))
    masks = img = Image.open(path+dirs[0])
    masks = convert_one_channel(np.asarray(masks))
    for i in range(1, len(dirs)):
        img = Image.open(path+dirs[i])
        img = convert_one_channel(np.asarray(img))
        masks = np.concatenate((masks, img))
    masks = np.reshape(masks, (len(dirs), 512, 512, 1))
    return masks
    




    
