#!/usr/bin/env python3

__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"

import os

import numpy as np
import SimpleITK as sitk
from typing import Iterable, Optional, Sequence, Union
import sys
import pandas as pd
from pathlib import Path
import cv2


def getslice(df: pd.DataFrame,
    odir: Union[str, Path], imtype: str):
    odir.mkdir(parents=True, exist_ok=True)
    imfile = Path(df[imtype])
    #maskfile = Path(df['MASK'])
    print('-----------------')
    #print('Processing image: ', maskfile)
    image = sitk.ReadImage(str(imfile))
    im_arr = sitk.GetArrayFromImage(image)
    #get the slice with more pixels corresponding to the tumor
    for n in range(int(np.shape(im_arr)[0])):
        plane_z = np.array(im_arr[n,:,:])
        final_filename= imfile.stem+'_{}.jpg'.format(n)
        (odir/imfile.parent.name).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(odir/imfile.parent.name/final_filename),plane_z)

odir = Path('/mnt/sda1/swarm-learning/radiology-dataset/slices')
dir_path = Path('/mnt/sda1/swarm-learning/radiology-dataset/converted-niix')
# get all the files in the directory ends with .nii
files = [f for f in os.listdir(dir_path) if f.endswith('.nii')]
# get rid of the files that the name ends with a.nii
files = [f for f in files if not f.endswith('a.nii')]
print(files)
for imfile in files:
    imfile = Path(imfile)

    image = sitk.ReadImage(dir_path/imfile)
    im_arr = sitk.GetArrayFromImage(image)
    patient_id = imfile.stem[:14]
    # get the slice with more pixels corresponding to the tumor
    for n in range(int(np.shape(im_arr)[0])):
        plane_z = np.array(im_arr[n, :, :])
        final_filename = imfile.stem + '_{}.jpg'.format(n)
        #print(odir / patient_id / final_filename)
        (odir / patient_id).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(odir / patient_id / final_filename), plane_z)