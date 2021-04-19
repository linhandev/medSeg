import os
import argparse
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import scipy.ndimage

parser = argparse.ArgumentParser()
parser.add_argument("--scan_dir", type=str, required=True, help="扫描或标签路径")
parser.add_argument("--out_dir", type=str, required=True, help="reisize后输出路径")
parser.add_argument("--size", nargs=2, required=True, help="resize目标大小")
paresr.add_argument(
    "-l",
    "--is_label",
    type=bool,
    default=False,
    help="是否是标签，如果是标签会使用一阶插值并转成uint8",
)
args = parser.parse_args()

# TODO: 区分标签和扫描
# TODO: 实现指定目标大小
def resize(inf_path):
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
    names = [
        os.path.join(args.scan_dir, n) for n in names if n.endswith("nii") or n.endswith("nii.gz")
    ]

    p = Pool(8)
    p.map(resize, names)

    # for name in names:
    #     to_512(name)
