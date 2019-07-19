from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
from config import *

def generate_img():
    while 1:
        img = np.zeros([img_h, img_w])

        curr_w = random.randint(10, 30)
        curr_h = curr_w
        # curr_h = random.randint(10,50)

        x_offset = random.randint(0, img_w - curr_w)
        y_offset = random.randint(0, img_h - curr_h)

        img[y_offset:y_offset + curr_h, x_offset:x_offset + curr_w] = 1
        label = [x_offset, y_offset, curr_w, curr_h]

        yield img, label


img_h = 100
img_w = img_h
if __name__ == '__main__':
    plt.ion()
    for _ in range(100):
        img, label = generate_img(img_h, img_w)
        plt.title('x:{}, y:{}, w:{}, h:{}'.format(label[0], label[1], label[2], label[3]))
        plt.imshow(img, cmap='gray')
        plt.pause(0.5)