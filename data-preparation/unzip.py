#!/usr/bin/env python3

__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"
# Script to unzip all .zip files in a specified directory and place the unzipped files in subdirectories with names according to the original .zip file names.

import os
import zipfile
import logging
from multiprocessing import Pool

input_dir = "/mnt/sda1/ukbiobank/20204/"  # The directory containing the zip files
output_dir = "/mnt/sda1/ukbiobank/20204_extracted/"  # The directory where the unzipped files will be placed


# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# logging
log_filename = os.path.join(output_dir, "unzip_log.txt")
logging.basicConfig(filename=log_filename, level=logging.INFO)

# Define a function to unzip a single file
def unzip_file(zip_path):
    zip_filename = os.path.basename(zip_path)
    output_subdir = os.path.join(output_dir, os.path.splitext(zip_filename)[0])
    os.makedirs(output_subdir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_subdir)
    logging.info(f"Unzipped {zip_filename}")

# Loop through all files in the input directory
zip_files = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir) if filename.endswith(".zip")]
with Pool() as pool:
    pool.map(unzip_file, zip_files)

