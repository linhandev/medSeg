import os
import os.path as osp
import argparse
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import scipy.ndimage

import util

parser = argparse.ArgumentParser()
parser.add_argument("--scan_dir", type=str, required=True, help="扫描或标签路径")
parser.add_argument("--out_dir", type=str, required=True, help="resize后输出路径")
parser.add_argument("--size", nargs=2, default=[512, 512], help="resize目标大小")
parser.add_argument("-t", "--thickness", type=float, default=None, help="统一的层间距")
parser.add_argument(
    "-l",
    "--is_label",
    default=False,
    action="store_true",
    help="是否是标签，如果是标签会使用零阶插值",
)
args = parser.parse_args()

# TODO: 区分标签和扫描
def resize(path):
    name = osp.basename(path)
    scanf = nib.load(path)
    header = scanf.header.copy()
    scan_data = scanf.get_fdata()
    old_pixdim = scanf.header.copy()["pixdim"]
    old_shape = scan_data.shape
    if scan_data.shape[:2] != args.size:
        scale = [t / c for c, t in zip(scan_data.shape[:2], args.size)]
        s = scale
        header["pixdim"][1] /= s[0]
        header["pixdim"][2] /= s[1]
    else:
        scale = [1, 1]

    if args.thickness and header["pixdim"][3] != args.thickness:
        scale.append(header["pixdim"][3] / args.thickness)
        header["pixdim"][3] = args.thickness
    else:
        scale.append(1)

    if scale != [1, 1, 1]:
        s = scale
        scan_data = scipy.ndimage.interpolation.zoom(
            scan_data, (s[0], s[1], s[2]), order=0 if args.is_label else 3
        )
        # if args.is_label:
        #     scan_data = scan_data.astype("uint8")

    newf = nib.Nifti1Image(scan_data.astype(np.float32), scanf.affine, header)
    nib.save(newf, osp.join(args.out_dir, name))
    print(
        name,
        ":",
        old_pixdim[1:4],
        old_shape,
        scale,
        header["pixdim"][1:4],
        scan_data.shape,
    )


if __name__ == "__main__":
    args.size = util.toint(args.size)
    names = os.listdir(args.scan_dir)
    names = [
        osp.join(args.scan_dir, n)
        for n in names
        if n.endswith("nii") or n.endswith("nii.gz")
    ]
    if not osp.exists(args.out_dir):
        os.makedirs(args.out_dir)

    p = Pool(8)
    p.map(resize, names)

    # for name in names:
    #     to_512(name)
