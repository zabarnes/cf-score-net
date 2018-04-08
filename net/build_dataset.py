"""
Process DICOM images and write to DATA directory.
We will also extract a csv of supplemental patient data. 
"""

import argparse
import random
import os
import pydicom

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--orig_dir', default='data/ORIG', help="Directory with the DICOM images")
parser.add_argument('--out_dir', default='data/IMGS', help="Where to write the new data")

FIELDS = ['AccessionNumber', 'AcquisitionDate', 'AcquisitionTime', 'BodyPartExamined', 'ExposuresOnPlate', 'ImageComments','ImageType', 'PatientSex', 'PatientBirthDate']

def process_dicom(fn, dataset, dest):
    # Extract/write the metadata from dicom
    dicom_data = pydicom.dcmread(fn)
    metadata = [str(dicom_data.get(field)).replace(",","|") for field in FIELDS]
    dataset.write(",".join(metadata)+"\n")

    # Build/write the image
    I = dicom_data.pixel_array
    I8 = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8)
    Image.fromarray(I8).save(os.path.join(dest,metadata[0]+".jpg"))

if __name__ == '__main__':
    args = parser.parse_args()

    # Get in/out DIRS
    assert os.path.isdir(args.orig_dir), "Couldn't find the original data at {}".format(args.orig_dir)
    if not os.path.exists(args.out_dir): os.makedirs(args.out_dir)

    # Get the filenames in the orig dataset
    filenames = os.listdir(args.orig_dir)
    
    # Open a file to store the image dataset
    dataset_dir = os.path.join(args.out_dir, "dataset.csv")
    with open(dataset_dir, 'w') as dataset:
        # Process DICOM
        dataset.write(",".join(FIELDS)+"\n")
        for filename in tqdm(filenames):
            fn = os.path.join(args.orig_dir, filename)
            process_dicom(fn, dataset, args.out_dir)

    print("Done building dataset")
