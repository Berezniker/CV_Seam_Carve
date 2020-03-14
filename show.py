#! /usr/bin/env python3

from pickle import load
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np

path = 'tests/01_'

gt = []
output = []
with open(path + 'test_img_check/gt/seams', 'rb') as current_input:
    for i in range(8):
        gt.append(load(current_input))

with open(path + 'test_img_check/output/output_seams', 'rb') as current_input:
    for i in range(8):
        output.append(load(current_input))

tests_type = [
    "horizontal shrink",
    "vertical shrink",
    "horizontal expand",
    "vertical expand",
    "horizontal shrink with mask",
    "vertical shrink with mask",
    "horizontal expand with mask",
    "vertical expand with mask",
]

img = imread(path + 'test_img_input/img.png')
width, height, _ = img.shape
for i in range(8):
    print(tests_type[i], gt[i] == output[i])

    _img = img.copy()

    for idx, jdx in output[i]:
        _img[idx, jdx] = [255, 0, 0]

    for idx, jdx in gt[i]:
        _img[idx, jdx] = [0, 255, 0]

    if output[i] != gt[i]:

        # for j in range(len(output)):
        #     if output[i][j] != gt[i][j]:
        #         print(output[i][j])
        #         print(gt[i][j])

        plt.imshow(_img)
        plt.show()
