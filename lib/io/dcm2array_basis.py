# coding=utf-8
# @Time	  : 2018-10-31 20:07
# @Author   : Monolith
# @FileName : dcm2array_basis.py
# @License  : (C)LINKINGMED,2018
# @Contact  : baibaidj@126.com, 18600369643
from lib.io import re, os, np, nb, time, json, pydicom, Counter, store_info_dict
from lib.info_dict_module import InfoDict
from lib.error import Error, ErrorCode
from lib.io.file_operation import is_dcm_file


def load_dcm_scan(info_dict):
    """
    读取单套CT序列，筛选出有效的dicom图像文件，提取序列信息并得到图像

    :param info_dict: 带有data_path, include_series 指定序列等字段
    :return: image_3d（按ipp_z从小到大排列的扫描图像）, info_dict（添加sop_list和ipp_list）
    """

    # 1. 筛选出可用pydicom读取成功的文件
    # 扫描数据根目录，筛选出dcm数据
    # 存两个字典（key=路径，value=pydicom object）
    # path_slices_dict存断层扫描，path_rts_dict存rs
    path_slices_dicts, path_rts_dicts = scan4image_rt(info_dict.data_path)
    series_list = list(path_slices_dicts.keys())
    most_series = None
    if len(series_list) == 0:
        Error.exit(ErrorCode.ld_ct_load_fail)
    elif len(series_list) ==1:
        most_series = series_list[0]
    else:
        nb_slices_in_series = [len(path_slices_dicts[s]) for s in series_list]
        most_series = series_list[nb_slices_in_series.index(max(nb_slices_in_series))]

    # 2. 将pydicom类转换成自定义类，筛选出带有效图像数据的层
    slice_data_dict, info_dict.slice_path = data_in_image_scans(path_slices_dicts[most_series])

    # 3. 提取断层扫描数据基本信息，
    info_dict = get_case_info(info_dict, slice_data_dict)

    # 筛选图像数据并根据参数排序
    order_slice_list, info_dict = sort_filter_slices(info_dict, slice_data_dict)

    # 4. 提取扫描图像数据
    image_3d = np.stack([s.image for s in order_slice_list], axis=0)

    return image_3d, info_dict


def data_in_image_scans(raw_slices_dict):
    """
    将pydicom类转换成自定义类，筛选出带有效图像数据的层
    :param raw_slices_dict:
    :return: slice_data_dict 键值是 每张slice的唯一识别码和数据信息， sop:scan_info
    """
    slice_data_dict = {} # key=sop; value = scan_info
    path, new_f_name = None, None
    series_uid_list = []
    for path, scan in raw_slices_dict.items():
        # print(path)
        slice_obj = SliceInfo_new(scan)
        if slice_obj.image is None:
            continue
        series_uid_list.append(slice_obj.series_uid)
        slice_sop = slice_obj.sop_uid
        slice_data_dict[slice_sop] = slice_obj

        # std_f_name = slice_sop + '.dcm'
        # if path.split(os.sep)[-1] != std_f_name:
        #     new_f_name = os.sep.join(path.split(os.sep)[:-1] + [std_f_name])
        #     os.rename(path, new_f_name)

    # series_unique = set(series_uid_list)
    some_slice_path = new_f_name if new_f_name is not None else path

    return slice_data_dict, some_slice_path



def sort_filter_slices(info_dict, slice_data_dict):
    # 获取医院名称
    # 显示有初始有多少张ct
    num_slice_total = len(slice_data_dict)
    print('     %s %s contains %d slices'
          % (info_dict.hospital, info_dict.pid, num_slice_total))

    #  筛选有效层并排序
    order_slice_list = _sort_slices(slice_data_dict, info_dict.ipp_order_reverse)
    order_slice_list = _filter_series(order_slice_list, info_dict.include_series)
    info_dict.image_shape_raw[0] = len(order_slice_list)

    # 提取单张图片的信息
    # 获取每层的唯一标识号
    info_dict.sop_list = [str(x.sop_uid) for x in order_slice_list]  # SOPInstanceUID
    # 获取每层的屋里坐标
    info_dict.ipp_list = [list(x.ipp) for x in order_slice_list]  # ImagePositionPatient

    # 患者增强的识别号
    info_dict.pid_aug = '_'.join([str(info_dict.pid),
                                  str(info_dict.series_uid[-10:]),
                                  str(info_dict.image_shape_raw[0])])
    # 显示有效层的数量
    num_slice_valid = len(order_slice_list)
    print('     Valid imaging slices: %d' % len(order_slice_list))
    if num_slice_total == 0 or num_slice_valid == 0:
        raise Error(ErrorCode.ld_ct_load_fail)
    return order_slice_list, info_dict



def get_case_info(info_dict, slice_data_dict):
    """
    提取一套序列图的基本信息
    :param info_dict:
    :param slice_data_dict:
    :return:
    """
    slice_obj = None
    for path, v in slice_data_dict.items():
        slice_obj = v
        if slice_obj is not None:
            break

    info_dict.hospital = 'somehosp'

    if 'hosp_in_root_ind' in info_dict.keys():
        if info_dict.hosp_in_root_ind is not None:
            info_dict.hospital = info_dict.data_path.split(os.sep)[info_dict.hosp_in_root_ind]
    info_dict.hospital = re.sub(r'[ \t\r\n\0./]', '', slice_obj.hospital) \
        if info_dict.hospital is 'somehosp' else info_dict.hospital

    # re.sub是为了剔除字符串中存在的异常字符，比如空格和斜杠等
    info_dict.pid = re.sub(r'[ \t\r\n\0./]', '', slice_obj.pid)
    info_dict.gender = slice_obj.gender
    info_dict.birthdate = slice_obj.birthdate
    info_dict.studydate = slice_obj.studydate
    info_dict.studyid = slice_obj.studyid
    info_dict.bodypart = slice_obj.bodypart
    info_dict.series_uid = str(slice_obj.series_uid)


    # necessary info
    info_dict.spacing_list = slice_obj.spacing_list
    info_dict.iop = [float(i) for i in slice_obj.iop]
    info_dict.image_shape_raw = [None,
                                 slice_obj.image.shape[0],
                                 slice_obj.image.shape[1]]




    return info_dict


def image_in_scan(scan):
    try:
        image = scan.pixel_array
    except AttributeError:
        try:
            image = scan.PixelData
        except AttributeError:
            try:
                image = np.frombuffer(scan.PixelData, dtype=np.int32, count=-1)
            except:
                return None

    intercept = scan.RescaleIntercept
    slope = scan.RescaleSlope

    if slope != 1:
        image = (slope * image.astype(np.float32)).astype(np.int16)
    image = np.array(image + intercept, dtype=np.int16)

    # # 防止有些数据有金属伪影
    # if hu_range is not None:
    #     image[image < hu_range[0]] = hu_range[0]
    #     image[image > hu_range[1]] = hu_range[1]

    return np.array(image, dtype=np.int16)


def read_slice_w_filter(dcm_path):
    try:
        if not os.path.exists(dcm_path):
            Error.exit(ErrorCode.ld_ct_path_not_exist)
        scan = pydicom.dcmread(dcm_path, force=True)
    except PermissionError:
        return None

    return is_valid_image(scan)


def is_valid_image(scan):
    if not bool(scan):
        return None

    if hasattr(scan, 'ImageType'):
        include_type = {'ORIGINAL', 'PRIMARY', 'AXIAL', 'REFORMATTED'}
        scan = scan if scan.Modality == 'CT' and include_type.intersection(scan.ImageType) else None
    else:
        print('.. modality is not CT')
        return None

    if not ((hasattr(scan, 'SliceLocation') or
             hasattr(scan, 'ImagePositionPatient')) and
            hasattr(scan, 'PixelSpacing')):
        print('...no valid spacing')
        return None

    if not bool(scan.file_meta.TransferSyntaxUID):
        print("Unknown Transfer Syntax, try to use  TransferSyntax: Little Endian Implicit")
        scan.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    return scan


class SliceInfo_new(object):
    def __init__(self, scan):

        # 图像信息
        self.series_uid = None
        self.sop_uid = None
        self.ipp = None
        self.iop = None
        self.spacing_list = None

        # 基本信息
        self.hospital = None
        self.pid = None
        self.studydate = None
        self.gender = None
        self.birthdate = None

        # 图像
        self.read(scan)
        self.image = image_in_scan(scan)

    def read(self, scan):
        """ 人口学信息"""
        # 医院信息
        self.hospital = re.sub(r'[ \t\r\n\0./]', '', str(scan.InstitutionName)) if hasattr(scan, 'InstitutionName') else None
        # self.hospital = re.sub('[/ ]', '', self.hospital) if self.hospital is not None else None
        self.hospital = 'somehosp' if self.hospital in ['', None] else self.hospital
        # 获取序列日期
        self.studydate = scan.StudyDate if hasattr(scan, 'StudyDate') else None
        # 获取性别
        self.gender = scan.PatientSex if hasattr(scan, 'PatientSex') else None
        # 获取出生年月
        self.birthdate = scan.PatientBirthDate if hasattr(scan, 'PatientBirthDate') else None
        # 获取病人id
        self.pid = re.sub(r'[ \t\r\n\0./]', '', str(scan.PatientID)) if hasattr(scan, 'PatientID') else None

        # 研究序列号
        self.studyid = scan.StudyID if hasattr(scan, 'StudyID') else None

        # 研究部位
        self.bodypart = scan.StudyDescription if hasattr(scan, 'StudyDescription') else None
        # 获取这套图的spacing，z-axis, rows and cols


        """扫描参数信息, spacing, iop, ipp"""
        try:
            rc_spacing = [a for a in scan.PixelSpacing]
            spacing = map(float, ([scan.SliceThickness] + rc_spacing))  # z x y
            self.spacing_list = list(spacing)
        except AttributeError:
            print('this case %s does not contain spacing')
            self.spacing_list = None
        # 获取患者体位朝向信息
        self.iop = list(np.array(scan.ImageOrientationPatient)) if \
            hasattr(scan, 'ImageOrientationPatient') else None
        if self.iop is None:
            print('this case %s does not contain IOP')
        # 获取层的物理坐标信息 image position patient IPP
        self.ipp = scan.ImagePositionPatient

        """断层扫描的识别信息"""
        # 研究序列号
        self.study_uid = scan.StudyInstanceUID
        self.series_uid = scan.SeriesInstanceUID
        self.refer_uid = scan.FrameOfReferenceUID
        self.class_uid = scan.SOPClassUID
        self.sop_uid = scan.SOPInstanceUID

def is_meet_condition(obj, attr = '',  type = str):
    """
    对象是否含有某个属性，属性的数据类型是否符合指定标准
    :param obj:
    :param attr:
    :param type:
    :return:
    """
    status = True
    status &= hasattr(obj, attr)
    if status:
        status &= type(obj.attr) == str
    return status

def _filter_series(slices, include_series='all'):
    """
    根据指定序列 纳入筛选图像；如果没有指定，则提取层数最多的那个序列
    :param slices:
    :param include_series:
    :return:
    """
    if len(slices) != 0:
        if include_series == 'most':
            # unique_series = list(set([s.series_uid for s in slices]))
            most_series = Counter([s.series_uid for s in slices]).most_common(1)
            slices = [s for s in slices if s.series_uid == most_series[0][0]]
        elif bool(include_series) and (include_series not in ['most', 'all']):
            # slices = [s for s in slices if s.SeriesDescription == include_series]
            slices = [s for s in slices if s.series_uid == include_series]

    return slices


def _sort_slices(slice_data_dict, ipp_order_reverse=False):
    """
    按IPP_z从小到大对slices进行排序
    :param slice_data_dict:
    :param ipp_order_reverse:
    :return:
    """

    # a = [print(x.ipp) for x in slices]
    order_slice_list = sorted(slice_data_dict.items(),
                              key=lambda d: d[1].ipp[-1],
                              reverse=bool(ipp_order_reverse))
    order_slice_list = [v for k, v in order_slice_list]
    # else:
    #     slices.sort(key=lambda x: int(x.ImagePositionPatient[2]), reverse=True)

    return order_slice_list



def scan4image_rt(path):
    """
    1. 提取跟路径下是dcm后缀的文件
    2. 筛选出可用pydicom读取成功的文件
    3. 判断pydicom是断层扫描还是rtstruct
     Get contour file from a given path by searching for ROIContourSequence
     inside dicom data structure.
     More information on ROIContourSequence available here:
     http://dicom.nema.org/medical/dicom/2016c/output/chtml/part03/sect_C.8.8.6.html
    """
    print('     searching %s' % path)
    tic = time.time()
    dcm_file_paths = []
    for subroot, subdir, subfiles in os.walk(path):
        if len(subfiles) > 0:
            files_path = [os.path.join(subroot, f) for f in subfiles if is_dcm_file(f)]
            dcm_file_paths.extend(files_path)

    path_slice_dict, path_rs_dict, series_ids = load_dcm_w_uids(dcm_file_paths)
    # nb_series = len(series_ids)
    path_slice_dicts = {}
    path_rs_dicts = {}

    for p, ct_s in path_slice_dict.items():
        ct_series = ct_s.SeriesInstanceUID
        if ct_series not in path_slice_dicts.keys():
            path_slice_dicts[ct_series] = {}
        if ct_series in series_ids:
            path_slice_dicts[ct_series][p] = ct_s

    for p, rs_s in path_rs_dict.items():
        rs_series_id = series_id_in_rtstruct(rs_s)

        if rs_series_id not in path_rs_dicts.keys():
            path_rs_dicts[rs_series_id] = {}
        if rs_series_id in series_ids:
            path_rs_dicts[rs_series_id][p] = rs_s

        # if rs_series_id in series_ids:
        #     # s_ix = series_ids.index(rs_series_id)
        #     path_rs_dicts[rs_series_id][p] = rs_s

    return path_slice_dicts, path_rs_dicts



def load_dcm_w_uids(dcm_file_paths):
    """
    逐个读取dcm文件并筛选出有效的图像和RS文件，分别放入相应的字典中
    :param dcm_file_paths:
    :return:
    """
    series_ids = []
    path_slice_dicts = {}
    path_rs_dicts = {}
    for f in dcm_file_paths:
        try:
            # print(f)
            scan = pydicom.dcmread(f, force=True)
        except PermissionError:
            print(f'     read {f} failed')
            continue
        except OSError:
            continue
        judge_scan = is_valid_image(scan)
        if judge_scan is not None:
            path_slice_dicts[f] = judge_scan
            series_ids.append(judge_scan.SeriesInstanceUID)
        else:
            if 'ROIContourSequence' in dir(scan):
                path_rs_dicts[f] = scan
                # rs_refer_series_id = series_id_in_rtstruct(scan)
                # series_ids.append(rs_refer_series_id)

    series_ids = list(set(series_ids))

    return path_slice_dicts, path_rs_dicts, series_ids


def series_id_in_rtstruct(rs_obj):
    """
    提取RS的序列号
    :param rs_obj: pydicom的类
    :return:
    """
    rs_refer = rs_obj.ReferencedFrameOfReferenceSequence[0]
    # study_uid
    rs_refer_study = rs_refer.RTReferencedStudySequence[0]
    # series_uid
    rs_refer_series = rs_refer_study.RTReferencedSeriesSequence[0]
    rs_refer_series_uid = rs_refer_series.SeriesInstanceUID
    return rs_refer_series_uid

def load_bin_json(info_dict):
    n_bytes4pixel = 4
    bin_dtype = np.float32

    # bin_file_path = os.path.join(info_dict.bin_path, 'ct.bin')
    # json_file_path = os.path.join(info_dict.bin_info_path, 'info.json')
    tic = time.time()
    with open(info_dict.bin_info_path, 'r') as jf:
        info = json.load(jf)
    toc = time.time()
    print(' load info takes %.3f' %(toc-tic))

    info_dict = store_info_dict(info_dict, info)
    with open(info_dict.bin_path, 'rb') as bf:
        buffer_data = bf.read()
    tic = time.time()
    print(' load bin takes %.3f' % (tic - toc))
    z, rows, cols = info_dict.image_shape_raw
    print(' image shape is %s' %str(info_dict.image_shape_raw))

    if info_dict.include_range is not None:
        z_range = info_dict.include_range
        info_dict.ipp_list = info_dict.ipp_list[z_range[0]:z_range[1]]
        info_dict.sop_list = info_dict.sop_list[z_range[0]:z_range[1]]
        info_dict.spacing_list = info_dict.spacing_list[z_range[0]:z_range[1]]
        info_dict.image_shape_raw = z_range[1] - z_range[0], rows, cols
    else:
        z_range = (0, z - 1)

    num_char_slice = rows * cols * n_bytes4pixel
    start_pos = z_range[0] * num_char_slice
    end_pos = (z_range[1] + 1) * num_char_slice
    target_bin_data = buffer_data[start_pos:end_pos]  # start_pos+len_pixel*512

    pixel_flatten = np.frombuffer(target_bin_data, dtype=bin_dtype, count=-1)
    image_target = pixel_flatten.reshape((z_range[1] - z_range[0] + 1), rows, cols)
    toc = time.time()
    print(' transform and slicing takes %.3f' %(toc-tic))
    return np.array(image_target, dtype=np.int16), info_dict


def load_npz_json(info_dict):
    """
    加载数据，支持两种格式，npz+json 和 纯的nii.gz
    :param info_dict:
    :return:
    """

    image_3d = None

    npz_file = [f for f in os.listdir(info_dict.npz_path) if 'npz' in f]
    json_file = [f for f in os.listdir(info_dict.npz_path) if 'info.json' in f]

    if 'img_file' in info_dict.keys():
        nii_file = [f for f in os.listdir(info_dict.npz_path) if info_dict.img_file in f]
        if len(nii_file) > 0:
            ct_file_path = os.path.join(info_dict.npz_path, nii_file[0])
            print(ct_file_path)
            nii_obj = nb.load(ct_file_path)
            # print(nii_obj.header)
            image_3d = np.swapaxes(nii_obj.get_data(), 2, 0)
            info_dict.spacing_list = nii_obj.header.get_zooms()[::-1]
            info_dict.image_shape_raw = image_3d.shape

    if len(npz_file) >0:
        ct_file_path = os.path.join(info_dict.npz_path, npz_file[0])
        print(ct_file_path)
        image_3d = np.load(ct_file_path)['img']
        # from lkm_lib.utlis.visualization import plot2Image
        # plot2Image(image_3d[1], image_3d[2])
    if len(json_file) > 0:
        json_file_path = os.path.join(info_dict.npz_path, json_file[0])
        with open(json_file_path, 'r') as jf:
            info = json.load(jf)
        info_dict = store_info_dict(info_dict, info)


    return np.array(image_3d, dtype=np.int16), info_dict


# if __name__ == '__main__':
#
#     ct_path = r'/media/dejun/holder/Data/sparse_testing/ZYi_eca1null_iop_except/Id283263'
#     info_dict = InfoDict()
#     info_dict.include_series = 'most'
#     info_dict.data_path = ct_path
#     info_dict = case_dicom_info(info_dict)
#     image_raw, info_dict = load_dcm_scan(info_dict)


