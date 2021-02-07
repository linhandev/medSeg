import os
import argparse
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import scipy.ndimage

parser = argparse.ArgumentParser()
parser.add_argument("--scan_dir", type=str, default="/home/lin/Desktop/data/aorta/external/nii_raw/")
parser.add_argument("--out_dir", type=str, default="/home/lin/Desktop/data/aorta/external/nii_512/")
args = parser.parse_args()


def to_512(inf_path):
    name = os.path.basename(inf_path)
    scanf = nib.load(inf_path)
    header = scanf.header.copy()
    scan_data = scanf.get_fdata()

    if scan_data.shape[0] != 512:
        print("------------------------------------------------------------")
        print(inf_path)
        d = header["pixdim"]
        scale = 512 / scan_data.shape[0]
        print(header["pixdim"])
        print(scale)
        header["pixdim"] = [-1, d[1] / scale, d[2] / scale, d[3], 0, 0, 0, 0]
        print(header["pixdim"])
        scan_data = scipy.ndimage.interpolation.zoom(scan_data, (scale, scale, 1), order=3)
        print(scan_data.shape)

    newf = nib.Nifti1Image(scan_data.astype(np.float32), scanf.affine, header)
    nib.save(newf, os.path.join(args.out_dir, name))


if __name__ == "__main__":
    names = os.listdir(args.scan_dir)
    names = [os.path.join(args.scan_dir, n) for n in names if n.endswith("nii") or n.endswith("nii.gz")]
    print(names)

    p = Pool(8)
    p.map(to_512, names)

    # for name in names:
    #     to_512(name)
