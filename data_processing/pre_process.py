#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Process image for iNaturalist 2018
"""

import logging
import multiprocessing as mp
import os
import shutil
from datetime import datetime
from glob import glob

from PIL import Image, ImageFilter

# Get path of the current dir to work with it
DIRPATH = os.path.dirname(os.path.realpath(__file__)).split(os.sep)
PTV = 'processed_train_val2018'
fh = logging.FileHandler(__file__ + '.log')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(fh)
mplogger = mp.log_to_stderr()
mplogger.setLevel(logging.INFO)
mplogger.addHandler(fh)


def process_images(f):
    mplogger.info('Processing : ' + os.path.join(os.sep, *DIRPATH, *f))
    # logger.info(str(count / len(filelist)) + "% done")
    newpath = f[1:]
    newpath.insert(0, PTV)
    im = Image.open(os.path.join(os.sep, *DIRPATH, *f))
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.resize((128, 128), Image.LANCZOS) \
        .filter(ImageFilter.SHARPEN) \
        .save(os.path.join(os.sep, *DIRPATH, *newpath), 'JPEG', optimize=True)


if __name__ == '__main__':
    # Log facility
    logger.info('Starting the processing of images for iNaturalist 2018')
    start = datetime.now()

    # Creating the dir which will receive all the files
    ptv_path = os.path.join(os.sep, *DIRPATH, PTV)
    if not os.path.exists(ptv_path):
        try:
            os.makedirs(ptv_path)
            logger.info('Created root dir : ' + ptv_path)
        except OSError as ose:
            logger.error(str(ose))
            exit(-1)

    # Get needed list of files path to process
    logger.info('Scanning files')
    dirlist = list(glob('train_val2018/*/*'))
    filelist = [[*root.split(os.sep), name] for root, dirs, files in os.walk('train_val2018')
                for name in files]

    logger.info('Creating needed directories ...')
    # Create all needed dirs
    for d in dirlist:
        subdir = d.split(os.sep)[1:]
        subdir.insert(0, PTV)
        path = os.path.join(os.sep, *DIRPATH, *subdir)
        if not os.path.exists(path):
            try:
                os.makedirs(path)
                logger.info('Created : ' + path)
            except OSError as ose:
                logger.error(str(ose))
                exit(-1)

    logger.info('Processing files in parallel')
    with mp.Pool(processes=5) as pool:
        pool.map(process_images, filelist)

    logger.info('Zipping everything !')
    shutil.make_archive(ptv_path, 'zip', ptv_path)

    logger.info('Deleting processed images directory')
    shutil.rmtree(ptv_path)

    end = datetime.now()
    logger.info('Process done in {} exiting !'.format(end - start))
