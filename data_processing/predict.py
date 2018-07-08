#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Predict result for Inaturalist competition on Kaggle
"""

import csv
import json
import operator
import os

import numpy as np
from PIL import Image, ImageFilter
from keras.models import load_model

MDL = 'VGG16'
IMG_SIZE = 128, 128, 3

if __name__ == '__main__':
    model = load_model(MDL + '.h5')
    with open('test2018.json') as in_test:
        in_test_dict = json.load(in_test)
    with open(MDL + '_result.csv', mode='w') as res_csv:
        csv_file = csv.writer(res_csv)
        csv_file.writerow(['id', 'predicted'])
        num_imgs = len(in_test_dict['images'])
        p = 1
        for img in in_test_dict['images']:
            timg = Image.open(os.path.join('data', 'test2018', img['file_name']))
            if timg.mode != 'RGB':
                timg = timg.convert('RGB')
            timg = timg.resize(IMG_SIZE, Image.LANCZOS).filter(ImageFilter.SHARPEN)
            np_timg = np.asarray(timg, dtype=np.float32) / 255.0
            predictions = list(model.predict(np_timg, batch_size=1))[0]
            results = list()
            for _ in range(3):
                index, value = max(enumerate(predictions), key=operator.itemgetter(1))
                results.append(index)
                predictions.pop(index)
            csv_file.writerow([str(img['id']), ' '.join(results)])
            print('Prediction progression :', str(p / num_imgs))
