#!/usr/bin/env python3

__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"
import numpy as np
import matplotlib.pyplot as plt

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from typing import Iterable, Optional, Sequence, Union
import sys
import pandas as pd
from pathlib import Path
import cv2
import tqdm
def plot_image(image):
    plt.imshow(image, cmap='gray')
    plt.show()

odir = Path('/mnt/sda1/swarm-learning/radiology-dataset/odelia_slices_sub_try')
dir_path = Path('/mnt/sda1/swarm-learning/radiology-dataset/examine_nii/')
sub_files = [Path(f) for f in dir_path.glob('**/*') if f.is_file() and Path(f).name == 'sub.nii.gz']
for imfile in (sub_files):
    imfile = Path(imfile)
    print(imfile)
    patient_id = str(imfile.parent).split('/')[-1]
    #print(patient_id)
    image = nib.load(imfile)
    data = image.get_fdata()

    print(data.shape)
    print(data[0, :, :])
    plot_image(data[0, :, :])
    full_patient_id = Path('Breast_MRI_' + patient_id)
