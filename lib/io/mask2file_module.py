# coding=utf-8
# @Time	  : 2018-12-11 14:15
# @Author   : Monolith
# @FileName : mask2file_module.py
# @License  : (C)LINKINGMED,2018
# @Contact  : baibaidj@126.com, 18600369643

from lib.io import *
from skimage.measure import subdivide_polygon
from skimage.filters import threshold_otsu
from concurrent.futures import ProcessPoolExecutor
from lib.error import ErrorCode, Error


def mask2file(mask_3d, info_dict, organ_name=None):
    """
    逐层计算单个roi的物理坐标，并存储为json文件。
    :param mask_3d:
    :param info_dict:
    :param roi_ind:
    :return:
    """
    # 增加organ_names的适用性---by YY

    if not os.path.exists(info_dict.goal_path):
        Error.exit(ErrorCode.tofile_ouput_path_not_exist)
    if not organ_name:
        Error.exit(ErrorCode.tofile_json_name_is_none)

    response_ind = [a for a in range(mask_3d.shape[0]) if np.amax(mask_3d[a]) > 0]
    # sop_list = [info_dict.sop_list[i] for i in response_ind]
    # ipp_list = [info_dict.ipp_list[i] for i in response_ind]
    ###提取含有分割结果的各层的信息 #J就是slice的层号
    label_slice_list = []
    transform_matrix = grid2world_matrix(info_dict.iop, info_dict.spacing_list)

    info_dict = contour_type_mode_by_roi(organ_name, info_dict)

    for r in response_ind:
        slice_obj = slice_roi_contours(mask_3d[r], info_dict.sop_list[r], info_dict.ipp_list[r],
                                       transform_matrix, organ_name,
                                       contour_type=info_dict.contour_type,
                                       chain_mode=info_dict.chain_mode,
                                       smooth_polygon_times=info_dict.smooth_polygon_times,
                                       smooth_polgygon_degree=info_dict.smooth_polygon_degree)
        label_slice_list.extend(slice_obj)
    # with ProcessPoolExecutor() as executor:
    #     future_obj = executor.map(slice_roi_contours, mask_3d[response_ind],
    #                               sop_list, ipp_list,
    #                               itertools.repeat(transform_matrix, len(response_ind)),
    #                               itertools.repeat(organ_names[roi_ind], len(response_ind)),
    #                               itertools.repeat(info_dict.contour_type, len(response_ind)),
    #                               itertools.repeat(info_dict.chain_mode, len(response_ind))
    #                               )
    #     for ll_slice in future_obj:
    #         label_slice_list.extend(ll_slice)

    one_roi2json(label_slice_list, info_dict.goal_path)


def mask2csv(mask_3d, info_dict, roi_ind=0, organ_names=None):
    """
    逐层计算单个roi的物理坐标，并存储为json文件。
    :param mask_3d:
    :param info_dict:
    :param roi_ind:
    :return:
    """
    # 增加organ_names的适用性---by YY
    if organ_names is None:
        organ_names = info_dict.organ_names

    if not os.path.exists(info_dict.goal_path):
        raise Error(ErrorCode.tofile_ouput_path_not_exist)
    if not organ_names[roi_ind]:
        raise Error(ErrorCode.tofile_json_name_is_none)

    response_ind = [a for a in range(mask_3d.shape[0]) if np.amax(mask_3d[a]) > 0]
    sop_list = [info_dict.sop_list[i] for i in response_ind]
    ipp_list = [info_dict.ipp_list[i] for i in response_ind]
    ###提取含有分割结果的各层的信息 #J就是slice的层号
    label_slice_list = []

    # for r in response_ind:
    #     slice_obj = slice_roi_contours_csv(mask_3d[r], info_dict.sop_list[r], info_dict.ipp_list[r],
    #                                     organ_names[roi_ind],
    #                                    contour_type=info_dict.contour_type,
    #                                    chain_mode = info_dict.chain_mode)
    #     label_slice_list.extend(slice_obj)
    with ProcessPoolExecutor(max_workers=16) as executor:
        future_obj = executor.map(slice_roi_contours_csv, mask_3d[response_ind],
                                  sop_list, ipp_list,
                                  itertools.repeat(organ_names[roi_ind], len(response_ind)),
                                  itertools.repeat(info_dict.contour_type, len(response_ind)),
                                  itertools.repeat(info_dict.chain_mode, len(response_ind))
                                  )
        for ll_slice in future_obj:
            label_slice_list.extend(ll_slice)

    to_csv(label_slice_list, info_dict.goal_path)


def mask2nii(mask_3d, store_root, file_name, nii_obj= None, is_compress = False):
    """
    将输入图像存储为nii
    输入维度是z, r=y, c=x
    输出维度是x=c, y=r, z
    :param mask_3d:
    :param store_root:
    :param file_name:
    :param nii_obj:
    :return:
    """

    extension = 'nii.gz' if is_compress else 'nii'
    mask_3d = np.swapaxes(mask_3d, 0, 2)
    store_path = os.path.join(store_root, '.'.join([file_name, extension]))
    if nii_obj is None:
        nb_ojb = nb.Nifti1Image(mask_3d, np.eye(4))
    else:
        nb_ojb = nb.Nifti1Image(mask_3d, nii_obj.affine, nii_obj.header)

    nb.save(nb_ojb, store_path)


def contour_type_mode_by_roi(organ_name, info_dict):
    if organ_name == 'Heart':
        info_dict.contour_type = cv2.RETR_EXTERNAL
        info_dict.chain_mode = cv2.CHAIN_APPROX_SIMPLE
        return info_dict
    else:
        return info_dict

def find_contours(mask_2d_ip, contour_type=cv2.RETR_TREE, chain_mode=cv2.CHAIN_APPROX_SIMPLE,
                  is_otsu = False):
    """
    在图像中找出label的边界坐标点
    :param mask_2d:
    :param contour_type: cv2.RETR_TREE= 3, cv2.RETR_EXTERNAL = 0
    :param chain_mode: cv2.CHAIN_APPROX_SIMPLE =2, cv2.CHAIN_APPROX_NONE =1
    :return: Numpy array of (x/col,y) coordinates of boundary points of the object
    """
    # mask_3d = mask_2d.reshape(tuple([1])+mask_2d.shape)
    # mask_2d = np.array(mask_2d, dtype=np.int16)
    # from lkm_lib.utlis.visualization import plot2Image
    # plot2Image(mask_2d_ip, mask_2d)
    # print(f'cv2 rescale range {np.amin(mask_2d)}-{np.amax(mask_2d)}')

    if is_otsu:
        thresh_value = threshold_otsu(mask_2d_ip)
        mask_bi = np.array(mask_2d_ip > thresh_value, dtype= np.uint8)
        # ret, mask_bi = cv2.threshold(mask_2d, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU) #
        # thresh = cv2.adaptiveThreshold(mask_2d, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    else:
        mask_2d_ip = cv2.convertScaleAbs(mask_2d_ip)  # rescale mask_2d to 0-255
        ret, mask_bi = cv2.threshold(mask_2d_ip, 0, 1, 0)
    # plot2Image(mask_2d, thresh)
    mask_bi, contours_list, hierarchy = cv2.findContours(mask_bi, contour_type, chain_mode)
    contours_n2 = []
    for k in range(len(contours_list)):
        obj_k_points = contours_list[k][:, 0, :]
        """如果没有如下操作，前端显示中将出现重大缺口"""
        # 在n*2的坐标点最后，添加初始的点，让整个连通区域闭合。
        obj_k_points = np.concatenate([obj_k_points, obj_k_points[0, :].reshape(1, -1)], axis=0)
        contours_n2.append(obj_k_points)

    return contours_n2


def grid2world4slice(trans_matrix, ipp):
    trans_matrix[3, :3] = ipp
    return trans_matrix


def grid2world_matrix(IOP, spacing):
    """
    计算像素坐标和物理坐标的变换矩阵
    :param IOP:患者的体位朝向信息，ImageOrientationPatient, 注意和IPP区别
    :param IPP_start:整套图头张slice的物理坐标，IPP
    :param spacing:
    :return: 4*4的转换矩阵
    """
    if IOP is None:
        IOP = [1, 0, 0, 0, 1, 0]
    IOP = [float(i) for i in IOP]
    trans_matrix = np.zeros((4, 4), dtype=np.float32)
    trans_matrix[-1, -1] = 1
    x_axis_direction_cosine = np.array(IOP[:3]) * spacing[2]
    y_axis_direction_cosine = np.array(IOP[3:]) * spacing[1]

    z_axis_direction_cosine = np.array([IOP[1] * IOP[5] - IOP[2] * IOP[4],
                                        IOP[2] * IOP[3] - IOP[0] * IOP[5],
                                        IOP[0] * IOP[4] - IOP[1] * IOP[3]]) * spacing[0]

    trans_matrix[:3, 0] = x_axis_direction_cosine
    trans_matrix[:3, 1] = y_axis_direction_cosine
    trans_matrix[:3, 2] = z_axis_direction_cosine

    return trans_matrix


def slice_roi_contours(mask_2d, sop, ipp, trans_matrix, roi_name,
                       contour_type=cv2.RETR_TREE,
                       chain_mode=cv2.CHAIN_APPROX_SIMPLE,
                       smooth_polygon_times=2,
                       smooth_polgygon_degree=3
                       ):
    """

    针对每张mask，基于相应的转换矩阵，获得轮廓点信息。
    :param mask_2d:
    :param sop: slice的唯一标识号
    :param ipp: 每层的起点物理坐标
    :param trans_matrix: 变换矩阵，从像素坐标到物理坐标
    :param roi_name:
    :param contour_type:
    :param chain_mode: 存储轮廓的复杂度，每个点都存，还是存几何形状的关键点
    :param smooth_polygon_times: 轮廓点平滑次数
    :param smooth_polgygon_degree: 轮廓点平滑种类
    :return:
    """
    label_list = []
    # plt.imshow(mask_2d)
    points_list = find_contours(mask_2d, contour_type=contour_type, chain_mode=chain_mode)
    ##正确办法：将每个连通区域的轮廓点，逐一存储，才能防止viewer上多个连通区域被杂乱链接来来。
    trans_matrix = grid2world4slice(trans_matrix, ipp)
    for k in range(len(points_list)):
        # obj_k_points = points_list[k][:, 0, :]
        """如果没有如下操作，前端显示中将出现重大缺口"""
        # 在n*2的坐标点最后，添加初始的点，让整个连通区域闭合。
        # obj_k_points = np.concatenate([obj_k_points, obj_k_points[0,:].reshape(1,-1)], axis=0)
        obj_k_points = points_list[k]
        if obj_k_points.shape[0] > 5:
            for _ in range(smooth_polygon_times):
                obj_k_points = subdivide_polygon(obj_k_points,
                                                 degree=smooth_polgygon_degree
                                                 )
        obj_contour_n4 = np.ones((obj_k_points.shape[0], 4))  # 建立一个n*4的矩阵，用于后续做变换
        obj_contour_n4[:, :2] = obj_k_points
        obj_contour_n4[:, 2] = 0
        obj_contour_n4_world_coor = np.array(np.dot(obj_contour_n4, trans_matrix))
        obj_contour_n4_world_coor = np.round(obj_contour_n4_world_coor, decimals=4)
        # obj_contour_n4_world_coor[:,2] = ipp
        slice_output = SliceContours(obj_contour_n4_world_coor[:, :3], sop, roi_name)
        label_list.append(slice_output)
    return label_list


def slice_roi_contours_csv(mask_2d, sop, ipp, roi_name,
                           contour_type=cv2.RETR_TREE,
                           chain_mode=cv2.CHAIN_APPROX_NONE):
    """
    针对每张mask，基于相应的转换矩阵，获得轮廓点信息。
    :param mask_2d:
    :param sop: slice的唯一标识号
    :param z_ind: slice的z轴序号
    :param trans_matrix: 变换矩阵
    :param roi_name: roi名
    :return:
    """
    label_list = []
    # plt.imshow(mask_2d)
    points_list = find_contours(mask_2d, contour_type=contour_type, chain_mode=chain_mode)
    ##正确办法：将每个连通区域的轮廓点，逐一存储，才能防止viewer上多个连通区域被杂乱链接来来。
    # trans_matrix = grid2world4slice(trans_matrix, ipp)
    for k in range(len(points_list)):
        # obj_k_points = points_list[k][:, 0, :]
        """如果没有如下操作，前端显示中将出现重大缺口"""
        # 在n*2的坐标点最后，添加初始的点，让整个连通区域闭合。
        # obj_k_points = np.concatenate([obj_k_points, obj_k_points[0,:].reshape(1,-1)], axis=0)
        obj_k_points = points_list[k]
        # if obj_k_points.shape[0] > 5:
        #     for _ in range(smooth_polygon_times):
        #         obj_k_points = subdivide_polygon(obj_k_points,
        #                                          degree=smooth_polgygon_degree
        #                                          )
        obj_contour_n4 = np.zeros((obj_k_points.shape[0], 3), dtype=np.int16)  # 建立一个n*4的矩阵，用于后续做变换
        obj_contour_n4[:, :2] = obj_k_points
        obj_contour_n4[:, 2] = ipp
        # obj_contour_n4_world_coor = np.array(np.dot(obj_contour_n4, trans_matrix))
        # obj_contour_n4_world_coor = np.round(obj_contour_n4_world_coor, decimals=4)
        # obj_contour_n4_world_coor[:,2] = ipp
        slice_output = SliceContours(obj_contour_n4, sop, roi_name)
        label_list.append(slice_output)
    return label_list


class SliceContours(object):
    """
    汇总一张CT slice的身份信息以及自动勾靶轮廓的坐标（物理坐标，需要用spacing计算的得到）

    """

    def __init__(self, obj_contours, slice_SOP, target_name):
        """
        :param obj_contours: 目标区域的像素坐标, N*3的矩阵, x, y, z
        :param slice_SOP: 对应slice的识别号
        :param IPP: 对应slice的物理起始坐标 ImagePositionPatient
        :param IOP: 患者的体位朝向信息，ImageOrientationPatient, 注意和IPP区别
        :param trans_matrix: 4*4的转换矩阵
        :param target_name:目标的名称
        """
        self.contourGeometricType = "CLOSED_PLANAR"
        self.contour = obj_contours
        self.target_name = target_name
        self.data = self.mask_coord2vector()
        if isinstance(slice_SOP, pydicom.uid.UID):
            self.slice_SOP = slice_SOP
        else:
            self.slice_SOP = pydicom.uid.UID(slice_SOP)

    def mask_coord2vector(self):
        pointArray = []

        count = self.contour.shape[0] - 1
        for k in range(count):
            pointArray.append(round((self.contour[k, 0]), 2))  # 这里spacing 包括了z,y,x三个值
            pointArray.append(round((self.contour[k, 1]), 2))  #
            pointArray.append(round((self.contour[k, 2]), 2))
        pointArray.append('')
        # coord = []
        self.count = count
        return np.array(pointArray)


def one_roi2json(slicedata_list, file_path):
    """
    将一个兴趣区的分割结果存储到Json文件
    :param slicedata_list: 包含 每张CT图基本信息和分割结果的集合
    :param file_path: 存储路径
    :return:无
    """
    type_name = slicedata_list[0].target_name
    file_name = os.path.join(file_path, type_name + '.json')
    json_data = []
    for slice in slicedata_list:
        points = list(filter(lambda x: x != '', slice.data))
        for i in range(len(points)):
            points[i] = float(points[i])
        jsondata = {}
        jsondata["referringSOPInstanceUID"] = slice.slice_SOP
        jsondata["contourGeometricType"] = slice.contourGeometricType
        jsondata["numofContourPoints"] = slice.count
        jsondata["contourData"] = points
        json_data.append(jsondata)
    with open(file_name, "w") as file:
        json.dump(json_data, file)

    print('     saved %s' % file_name)


def to_csv(slicedata_list, file_path):
    """
    parameters:
        slicedata_list: data to write
        file_path: path to save file
    """
    type_name = slicedata_list[0].target_name
    file_name = os.path.join(file_path, type_name + '.csv')
    data = []
    for slice in slicedata_list:
        data.extend(slice.data)
    data = np.array(data[0:-1]).reshape(1, -1)
    df = pd.DataFrame(data)
    df.to_csv(file_name, header=False, index=False)
