from skimage.io import imsave
import numpy as np
from config import *
import os

def save_png(images, col_size, path):

    img = (images + 1.0) / 2.0
    h, w = img.shape[1], img.shape[2]
    merge_img = np.zeros((h * col_size[0], w * col_size[1], 3))
    for idx, image in enumerate(images):
        i = idx % col_size[1]
        j = idx // col_size[1]
        merge_img[j * h:j * h + h, i * w:i * w + w, :] = image

    root, file_name = os.path.split(path)
    if not os.path.exists(root):
        os.makedirs(root)
    #print(np.max(merge_img), np.min(merge_img))
    merge_img = (merge_img * 255 / 2 + 255 / 2).clip(0, 255).astype(np.uint8)
    imsave(path, merge_img)
    #print(np.max(merge_img), np.min(merge_img))

def save_tiff(images, col_size, path):

    img = (images + 1.0) / 2.0
    h, w = img.shape[1], img.shape[2]
    merge_img = np.zeros((h * col_size[0], w * col_size[1], 1))
    for idx, image in enumerate(images):
        i = idx % col_size[1]
        j = idx // col_size[1]
        merge_img[j * h:j * h + h, i * w:i * w + w, :] = image

    root, file_name = os.path.split(path)
    if not os.path.exists(root):
        os.makedirs(root)

    imsave(path, np.reshape(merge_img[:,:,0], [merge_img.shape[0],merge_img.shape[1]]).astype(np.float32))

def save_one_tiff(images, path):

    img = (images + 1.0) / 2.0

    root, file_name = os.path.split(path)
    if not os.path.exists(root):
        os.makedirs(root)

    imsave(path, img.astype(np.float32))

def save_one_png(images, path):
    root, file_name = os.path.split(path)
    if not os.path.exists(root):
        os.makedirs(root)

    imsave(path, images)