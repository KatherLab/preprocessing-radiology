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
import tqdm
from PIL import Image
import glob, os
odir = Path('/mnt/sda1/swarm-learning/radiology-dataset/odelia_slices_sub_try')
dir_path = Path('/mnt/sda1/swarm-learning/radiology-dataset/examine_nii/')
sub_files = [Path(f) for f in dir_path.glob('**/*') if f.is_file() and Path(f).name == 'sub.nii.gz']
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
def plot_image(image):
    plt.imshow(image, cmap='gray')
    #plt.show()
    plt.savefig('test.png')


for imfile in (sub_files):
    imfile = Path(imfile)
    save_name = imfile.name
    print(imfile)
    patient_id = str(imfile.parent).split('/')[-1]
    #print(patient_id)
    image = sitk.ReadImage(imfile)
    im_arr = sitk.GetArrayFromImage(image)
    #print(im_arr.dtype)

    #print(im_arr.shape)
    #print(im_arr[0, :, :])
    plot_image(im_arr[0, :, :])

    full_patient_id = Path('Breast_MRI_' + patient_id)

    for n in range(int(np.shape(im_arr)[0])):
        plane_z = np.array(im_arr[n, :, :])
        #print(np.shape(plane_z))
        #print(plane_z)
        final_filename = imfile.stem.split('.')[0] + '_{}.jpg'.format(n)
        #print(odir / patient_id / final_filename)
        (odir / full_patient_id).mkdir(parents=True, exist_ok=True)
        #cv2.read(plane_z)

        #plt.imshow(plane_z, cmap='gray')
        #plt.axis('off')
        #plt.show()

        # save image as 256x256

        plt.imsave(fname=str(odir / full_patient_id / final_filename), arr=plane_z, cmap='gray')

        #plt.savefig(str(odir / full_patient_id / final_filename), bbox_inches='tight', pad_inches=0,transparent=True)
        #plt.savefig(str(odir / full_patient_id / final_filename) + '.png', bbox_inches='tight', pad_inches=0)
        #plt.savefig(str(odir / full_patient_id / final_filename) + '.png')
        #cv2.imwrite(str(odir / full_patient_id / final_filename), plane_z)
