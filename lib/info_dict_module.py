#coding=utf-8

import cv2

# 时间相关
time_dict_default = {
    'load_data_begin': -1.,
    'load_data_end': -1.,
    'load_data_diff': -1.,
    'classification_begin': -1.,
    'classification_end': -1.,
    'classification_diff': -1.,
    'preprocess_begin': -1.,
    'preprocess_end': -1.,
    'preprocess_diff': -1.,
    'load_model_begin': -1.,
    'load_model_end': -1.,
    'load_model_diff': -1.,
    'prediction_begin': -1.,
    'prediction_end': -1.,
    'prediction_diff': -1.,
    'postprocess_begin': -1.,
    'postprocess_end': -1.,
    'postprocess_diff': -1.,
    'to_file_begin': -1.,
    'to_file_end': -1.,
    'to_file_diff': -1.,
}


class InfoDict(dict):

    def __init__(self, time_dict = time_dict_default.copy(), **kwargs):
        # 版本相关
        self.release_version = '1.1.9'
        self.alg_name = None

        # 数据路径信息 #成都调用时传入的参数
        self.data_path = None # ct图像路径
        self.classes_path = None # 全身分类的路径
        self.goal_path = None # 最终存储json的路径
        self.model_path = None #模型路径
        self.bin_path = None
        self.bin_info_path = None
        self.npz_path = None

        # 器官相关信息
        self.organ_names = []
        self.organ_version = None
        self.organ_classify_pos = None #包含这个器官的全身分类
        self.crop_x_range = None
        self.crop_y_range = None
        self.model_input_size = None ##这个啥意思？
        self.model_name = None
        self.model_names = None
        self.is_print_logo = None
        self.is_save_csv = None
        self.include_range = None  # 提取的上下界，两个元素
        self.united_rois_dict = None #合并多个器官成一个的字典

        # 轮廓存储参数
        self.contour_type = cv2.RETR_EXTERNAL #cv2.RETR_TREE #  cv2.RETR_EXTERNAL
        self.chain_mode = cv2.CHAIN_APPROX_SIMPLE #cv2.CHAIN_APPROX_NONE # cv2.CHAIN_APPROX_SIMPLE
        self.interp_kind = cv2.INTER_LINEAR

        self.smooth_polygon_times = 1 # 逼近次数越多，生成的文件越大。
        self.smooth_polygon_degree = 7 # 采用B样条逼近多边形曲线，参考如下
        # http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.subdivide_polygon

        # 分类相关
        self.classify_file_name = 'series_classes.json'
        self.classes_value = None
        self.head2feet = None

        # 体位矫正相关
        self.head_adjust_angle = None
        self.head_adjust_center = None
        self.body_adjust_angle = None
        self.body_adjust_center = None
        self.use_head_adjust = True
        self.use_body_adjust = True

        # 大图像矫正相关
        self.large_image_shape = None
        self.large_raw_spacing = None
        self.use_large_image_check = True

        # 图像dicom信息
        # 以下三个列表以及图像，层数必须一样，顺序也必须一一对应
        self.sop_list = [] # SOP instance UID
        self.ipp_list = [] # image patient position
        self.image_shape_raw = [] # z rows cols

        self.spacing_list = []
        self.iop = None # image orientation patient
        self.hospital = 'Linkingmed'
        self.pid = 'John^Doe'
        self.gender = None
        self.include_series = 'all' # most or series description
        self.series_uid = None
        self.ipp_order_reverse = False

        # 前处理相关
        self.target_spacing = None
        self.hu_range = (-1000,1000) # CT图像的取值范围，可去伪影
        self.absolute_x_range = None
        self.absolute_y_range = None
        self.image_shape_postinterp = None
        self.window_width = 2000
        self.window_center = 0

        # 预测相关
        self.predict_batch_size = 1
        self.allow_soft_placement = True
        self.allow_growth = True
        self.tf_log_level = 2
        self.gpu_memory_fraction = None
        self.verbose = 1
        self.multigpu_model = False

        # 时间相关
        self.time_dict = time_dict

        # 单元测试相关
        self.is_unit_testing = False

        # 存储传入字典
        for key, value in kwargs.items():
            self[key] = value

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __call__(self, key):
        return self[key]

    def __getstate__(self):
        pass

    def copy(self):
        info_dict = InfoDict()
        for temp_dict_key, temp_dict_value in self.items():
            info_dict[temp_dict_key] = temp_dict_value
        return info_dict

    def get_para(self, key, default_value):
        if (key in self.keys()) and (self[key] != None):
            rst_value = self[key]
        else:
            rst_value = default_value
        return rst_value


# if __name__ == "__main__":
#     import json
#     data_info = InfoDict()
#     data_info.data_path = 'lol'
#     print(json.dumps(data_info, indent=4))
#
#     info_dict_new = data_info.copy()
#     info_dict_new.data_path = 'lllll'
#     print(data_info.data_path)
#     print(info_dict_new.data_path)
#
#
#     # data_info['test'] = 'lol'
#     # print(json.dumps(data_info, indent=4))
#     # print(data_info.test)
#     #
#     # data_info.test1 = 'lol1'
#     # print(data_info)
#     # print(data_info['test1'])
#
#     print(type(data_info))