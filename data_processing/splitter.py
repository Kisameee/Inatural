#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Move all files in data to inaturalist data in train and val data
"""

import json
import os
import shutil
from glob import glob

with open(os.path.join('ptrain2018.json')) as ptj, \
    open(os.path.join('pval2018.json')) as pvj:
    pTrain = json.load(ptj)
    PVAL = json.load(pvj)

dirlist = list(glob('data/*/*'))

for d in dirlist:
    dirName = d.split('//')[-2:]
    pathTrain = os.path.join('train', *dirName)
    pathVal = os.path.join('val', *dirName)
    if not os.path.exists(pathTrain) and not os.path.exists(pathVal):
        try:
            os.makedirs(pathTrain)
            os.makedirs(pathVal)
        except OSError as ose:
            print(str(ose))
            exit(-1)
i = 1
print('creating a train and validation dataset')

for pathFile in pTrain:
    try:
        shutil.move(pathFile['file_name'],
                    os.path.join('train', *pathFile['file_name'].split('/')[1:]))
    except Exception as e:
        print(str(e))
    print(str(i / len(pTrain)))
    i += 1

i = 1
for pathFile in PVAL:
    try:
        shutil.move(pathFile['file_name'],
                    os.path.join('val', *pathFile['file_name'].split('/')[1:]))
    except Exception as e:
        print(str(e))
    print(str(i / len(PVAL)))
    i += 1
