# coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: __init__.py
@Time: 2018-12-04 13:22
@Last_update: 2018-12-04 13:22
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""

import os
import sys
import re
import cv2
import time
import json
import shutil
import numpy as np
import pandas as pd
import pydicom
from collections import Counter
import itertools
import datetime
import nibabel as nb
from concurrent.futures import ProcessPoolExecutor
np.set_printoptions(precision=4, suppress=True)
from lib.io.store_info_dict_module import store_info_dict
from lib.io.parser_argv_as_dict_module import parser_argv_as_dict, parser_classify_argv_as_dict
from lib.io.dcm2array_basis import SliceInfo_new, load_dcm_scan, load_bin_json, load_npz_json
from lib.io.mask2file_module import mask2file, mask2nii, grid2world_matrix, slice_roi_contours, one_roi2json, find_contours
from lib.io.parser_keyvalue_argv_as_dict_module import parser_keyvalue_argv_as_dict
from lib.io.parser_argv_to_info_dict_module import parser_argv_to_info_dict




class DcmInfo(dict):


    def __init__(self, **kwargs):
        self.include_series = 'all' # most or series description
        self.series_uid = None
        self.ipp_order_reverse = False
        self.refer_doc = None

        for temp_dict_key, temp_dict_value in kwargs.items():
            self[temp_dict_key] = temp_dict_value

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __call__(self, key):
        return self[key]

    def __getstate__(self):
        pass

# def file_size(file_path, unit='kb'):
#     size = os.path.getsize(file_path)
#     param = 1024
#     if unit == 'kb':
#         return round(size / param, 3)
#     elif unit == 'mb':
#         return round(size / param / param, 3)
#     else:
#         raise ('unit can only be kb or mb')


def generate_number():
    """
    生成时间戳
    :return: 时间戳序列
    """
    now_time = datetime.datetime.now().strftime("%m%d%H%M%S%f")
    unique_num = str(now_time)
    return unique_num


def get_key(dict, value):
    key = [k for k, v in dict.items() if value in v]
    if key:
        return key[0]
    else:
        return 'others'



rois_by_bodypart = {
    "hnead": ["Eye_L", "Eye_R","Lens_L", "Lens_R",
              "OpticNerve_L", "OpticNerve_R", "OpticChiasm",
              "TemporalLobe_L", "TemporalLobe_R",
              "Cochlea_L", "Cochlea_R", 'BrainStem',
              "Brain", "Cerebrum", "Cerebellum", "Body",
              "Parotid", "Parotid_L", "Parotid_R",
              "Pituitary", "Thyroid", "Mandible","SMG_L", "SMG_R", 'SMG',
              "Mandible_L", "Mandible_R", "Tongue", "Larynx", "OralCavity",
              "TMJ_L", "TMJ_R", "Scleido_M", "Sternohyoid_M"
                             ],
                    # "neck": ["Trachea", "Thyroid", "Larynx", "Scleido_M", "Sternohyoid_M"],

    "chest": [ "Lung_L", "Lung_R",'Bronchus',
               'Bronchus_R', 'Bronchus_L',"Trachea",
               "Heart", "Atrium_L", "Atrium_R",
               "Ventricle_L", "Ventricle_R",
               "Breast_L", "Breast_R",
               "Esophagus", "SpinalCord"
                              ],

    "abdomen": ["Liver", "Stomach", "Pancreas", "Spleen",'Kidney'
                "Kidney_L", "Kidney_R", "SmallIntestine", "BowelBag"],

    "pelvis": ["Bladder", "PelvicBone", "Rectum", "Sigmoid",'Femur',
               "FemoralHead_L", "FemoralHead_R", "Femur_L", "Femur_R",
               'Ovary_R', 'Ovary-L', 'Ovary-r'],

    'vessels':["Vertebral.A_L", "Vertebral.A_R",
               'CCA',"CCA_L", "CCA_R",
               "Aorta", "IJV_L", "IJV_R",
               "BCV_L", "BCV_R", "SVC", "IMA_L", "IMA_R",
               "IVC", "Subclavian.A_L", "Subclavian.A_R",
               "Pulmonary.A", "Pulmonary.V", "IMA", 'vein', 'Vertebral.A'],

    'vertebra': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8',
                 'T9', 'T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4']
                    # "longorgan": ["Body", "SpinalCord"]
                    }