# 将肝脏和肿瘤的分割标签合并
import nibabel as nib
import os

liver_path = "/home/aistudio/liver_seg"
tumor_path = "/home/aistudio/tumor_seg"
merge_path = "/home/aistudio/lab_merge"


assert len(os.listdir(liver_path)) == len(os.listdir(tumor_pathr)), "肝脏和肿瘤的分割标签数量不想等"

for liver_fname in tadm(os.listdir(liver_path)):
    liverf = nib.load(os.path.join(liver_path, liver_fname))
    tumorf = nib.load(os.path.join(tumor_path, liver_fname))

    liver = liverf.get_fdata()
    tumor = tumorf.get_fdata()

    assert len(np.where(tumor == 1)[0]) != 0, "肿瘤中包含标签为1的前景，是不是肝脏和肿瘤数据弄反了"
    assert len(np.where(liver == 2)[0]) != 0, "肝脏中包含标签为2的前景，是不是肝脏和肿瘤数据弄反了"

    print("肝体{},肿瘤{}".format(liver.sum(), tumor.sum()))

    tumor = tumor / 2
    liver += tumor

    merge_file = nib.Nifti1Image(liver, volf.affine)
    nib.save(merge_file, os.path.join(merge_path, liver_fnamem))
