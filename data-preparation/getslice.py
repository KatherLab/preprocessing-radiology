
import os

import numpy as np
import SimpleITK as sitk
from typing import Iterable, Optional, Sequence, Union
import sys
import pandas as pd
from pathlib import Path
import cv2

PathLike = Union[str, Path]

def getslice(df: pd.DataFrame, 
    odir: Union[str, Path], imtype: str):    
    odir.mkdir(parents=True, exist_ok=True)
    imfile = Path(df[imtype])
    maskfile = Path(df['MASK'])       
    print('-----------------')
    print('Processing image: ', maskfile)                              
    image = sitk.ReadImage(str(imfile))  
    im_arr = sitk.GetArrayFromImage(image)
    #get the slice with more pixels corresponding to the tumor
    for n in range(int(np.shape(im_arr)[0])):
        plane_z = np.array(im_arr[n,:,:])        
        final_filename= imfile.stem+'_{}.jpg'.format(n)
        (odir/imfile.parent.name).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(odir/imfile.parent.name/final_filename),plane_z)

   
if __name__ == '__main__':
    csv_path = Path(sys.argv[1])
    odir = Path(sys.argv[2])
    df = pd.read_csv(csv_path)
    imtype = sys.argv[3]
    import logging
    LOG = odir.parents[0]/'log_file_crop.txt'  # Here you can ount n the log name a 'log_{}'.format(params) with the params for this run    
    logger = logging.getLogger()
    logHandler = logging.FileHandler(filename=LOG, mode='w')
    logHandler.setLevel(logging.INFO)
    logHandler.setFormatter(logging.Formatter('%(levelname)-.1s: (%(threadName)s) %(name)s: %(message)s'))
    logger.addHandler(logHandler)
    for index, row in df.iterrows():
        getslice(row,odir, imtype)
