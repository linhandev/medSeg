import os
import argparse

from tqdm import tqdm
import nibabel as nib
import numpy as np

from util import filter_largest_volume

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--in_dir", type=str, required=True)
parser.add_argument("-o", "--out_dir", type=str, required=True)
args = parser.parse_args()

names = os.listdir(args.in_dir)

for name in tqdm(names):
    segf = nib.load(os.path.join(args.in_dir, name))
    header = segf.header
    data = segf.get_fdata()
    data = filter_largest_volume(data, mode="hard")
    newf = nib.Nifti1Image(data, segf.affine, header)
    nib.save(newf, os.path.join(args.out_dir, name))
