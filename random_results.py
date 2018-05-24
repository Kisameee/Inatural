#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Random Result for Inaturalist competition on Kaggle
"""


import os
import random
import csv
import json


if __name__ == '__main__':
    with open('test2018.json') as in_test:
        in_test_dict = json.load(in_test)
    with open('random_result.csv', mode='w') as res_csv:
        csv_file = csv.writer(res_csv)
        csv_file.writerow(['id', 'predicted'])
        for img in in_test_dict['images']:
            random_classes = list()
            for _ in range(3):
                random_classes.append(str(random.randint(0, 8141)))
            csv_file.writerow([str(img['id']), ' '.join(random_classes)])
