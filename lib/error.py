# coding:utf-8

import sys
import traceback
import logging

class ErrorCode(object):
    '''环境资源错误码'''
    env_compu_res_insufficient = 2  # 服务器计算资源不足
    env_mem_res_insufficient = 3    # 服务器内存资源不足
    env_storage_res_insufficient = 4  # 服务器存储资源不足
    """  数据读取错误码  """
    ld_ct_load_module_error = 10  # 加载数据模块失败
    ld_ct_path_not_exist = 11  # CT数据路径不存在
    ld_ct_load_fail = 12  # CT图像读取失败
    ld_position_unsupported = 13  # CT图像体位不支持
    ld_thick_zero = 14  # CT图像层厚为0
    ld_spacing_zero = 15  # CT的spacing为0
    ld_no_target_layer = 16  # 图像中没有目标器官的层，基于分类结果
    ld_no_classify_json = 17  # 没有全身分类json
    ld_classify_code_error = 18  # 分类结果错误

    """  加载模型和权重错误码  """
    lm_model_load_module_error = 20  # 加载模型模块失败
    lm_model_path_not_exist = 21  # 模型路径不存在
    lm_weight_path_not_exist = 22  # 权重路径不存在
    lm_load_model_fail = 23  # 模型载入失败
    lm_load_weight_fail = 24  # 权重读取失败

    """  预测错误码  """
    pred_input_is_none = 31  # 预测模块输入数据为空
    pred_fail = 32  # 数据预测失败(指调用Tensorflow的预测函数失败)
    pred_ouput_is_none = 33  # 预测输出结果为空
    pred_input_shape_error = 34  # 输入数据的shape和模型的输入不符

    """  图像处理错误码  """
    process_module_error = 40  # 图像处理模块失败
    process_input_shape_error = 41  # 输入数据的维度不符合要求
    process_data_type_error = 42  # 数据类型不符合标准 (cv2对数据类型要求严格,类型不对会有奇怪的报错)
    process_clips_out_of_range = 43  # 裁剪尺寸越界
    process_postinterp_mismatch = 44  # 反插值后维度与原图不一致

    process_output_is_empty = 45   # 预测输出数据处理后为空


    """  to_file 错误码  """
    tofile_module_error = 50  # tofile模块失败
    tofile_ouput_path_not_exist = 51  # 输出路径不存在
    tofile_label_post_empty = 52  # 输出的 Json 文件数量和预测结果类别数不一致
    tofile_json_name_is_none = 53  # json name 为空

    """  保存log信息的字典  """
    description_4_code = {
        #服务器资源错误码
        env_compu_res_insufficient: 'Server computing resources are insufficient',
        env_mem_res_insufficient: 'Server memory resources are insufficient',
        env_storage_res_insufficient: 'Server storage resources are insufficient',
        # 数据读取错误码
        ld_ct_load_module_error:'load data: CT data load module error',
        ld_ct_path_not_exist: 'load data: CT path does not exist',
        ld_ct_load_fail: 'load data: load CT Failed',
        ld_position_unsupported: 'load data: CT position is unsupported',
        ld_thick_zero: 'load data: CT layer thickness is zero',
        ld_spacing_zero: 'load data: CT spacing is zero',
        ld_no_target_layer:'no target organ layer',
        ld_no_classify_json: 'no classify json',
        ld_classify_code_error:'classify code error',

        # 加载模型和权重错误码
        lm_model_load_module_error: 'load model: model data load module error',
        lm_model_path_not_exist: 'load model: model path does not exist',
        lm_weight_path_not_exist: 'load model: weight path does not exist',
        lm_load_model_fail: 'load model: load model fail',
        lm_load_weight_fail: 'load model: load weight fail',

        # 预测错误码
        pred_input_is_none: 'prediction: input data is none',
        pred_fail: 'prediction: predict failed',
        pred_ouput_is_none: 'prediction: ouput is none or the result is not marked',
        pred_input_shape_error: 'prediction: input shape is no match the model',

        # 图像处理错误码
        process_module_error: 'process: process module error',
        process_input_shape_error: 'process: input data shape error',
        process_data_type_error: 'process: data type error',
        process_clips_out_of_range: 'process: clips out of range',
        process_postinterp_mismatch: 'process: shape of postinterp data is not match the origin',
        process_output_is_empty:'process: the data after process is empty',

        # to_file 错误码
        tofile_module_error: 'to_file: to_file module error',
        tofile_ouput_path_not_exist: 'to_file: output path does not exist',
        tofile_label_post_empty: 'to_file: no valid values in the label post',
        tofile_json_name_is_none: 'to_file: json name is none',
    }


class Error(Exception):
    EXIT_2_RAISE = False
    def __init__(self, code):
        self.code = code
        self.description = ''

        self._get_description()

        # 拼接信息，在打印堆栈信息的最后一行显示
        self.args = ('\n' + self.description + '\nErrorCode: ' + str(self.code),)

    def print_log(self, is_exit=True):
        """
        打印log,在excption中调用该函数自动打印log
        """
        print('ErrorCode:', self.code, '    description: ', self.description)
        if is_exit:
            traceback.print_exc()
            sys.exit(self.code)

    def _get_description(self):
        description_4_code = ErrorCode.description_4_code
        keys_list = list(description_4_code.keys())

        self.description = description_4_code[self.code] if self.code in keys_list else str("Unknown error")

    @classmethod
    def get_description(cls,error_code):
        """
        获取错误消息
        """
        description_4_code = ErrorCode.description_4_code
        keys_list = list(description_4_code.keys())
        desc_str = description_4_code[error_code] if error_code in keys_list else str("Unknown error")
        return desc_str

    @classmethod
    def exit(cls,error_code):
        """
        打印出错信息,退出
        param1: error_code - 出错代码
        """
        print('ErrorCode:', error_code, '    description: ', Error.get_description(error_code))
        traceback.print_exc()
        if cls.EXIT_2_RAISE:
            raise ValueError(Error.get_description(error_code))
        else:
            sys.exit ( error_code )

    @staticmethod
    def warn(str_warn=None):
        """
        输出警告信息
        """
        logging.warning(str_warn)
        return None


# if __name__ == '__main__':
#     import numpy as np
#     #import sys
#     #import traceback
#
#     def process_data(data):
#         try:
#             shape = data.shape  # 获取data的shape
#
#             if shape != (2, 2, 2):  # 判断shape是否符合要求,不符合抛出shape异常的错误
#                 raise Error(ErrorCode.process_input_shape_error)
#                 #Error.exit(ErrorCode.process_input_shape_error)
#
#             pass  # 这里pass表示省略其他代码
#
#             if not data.dtype == np.uint8:  # 判断data的类型是否符合要求,不符合抛出相应的错误
#                 raise Error(ErrorCode.process_data_type_error)
#         except Error as e:
#             e.print_log()  # 打印log
#
#
#     #Error.warn('warning message.')
#     Error.warn('warning message！')
#     Error.exit( ErrorCode.env_storage_res_insufficient )  # ErrorCode.ld_no_target_layer
#     # 测试1: 输入维度不符合的数据
#     #test_data1 = np.zeros(shape=(3, 3, 3), dtype=np.uint8)
#     #process_data(test_data1)
#
#     # # 测试2: 输入数据类型不符合的数据
#     # test_data2 = np.zeros(shape=(2, 2, 2), dtype=np.float32)
#     # process_data(test_data2)
#     #
#     # # 测试3: 输入符合要求的数据的
#     # test_data2 = np.zeros(shape=(2, 2, 2), dtype=np.uint8)
#     # process_data(test_data2)

