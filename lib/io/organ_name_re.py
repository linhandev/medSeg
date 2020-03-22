# coding=utf-8
# @Time	  : r2018-11-21 15:04
# @Author   : rMonolith
# @FileName : rfile_operation_module.py
# @License  : r(C)LINKINGMED,2018
# @Contact  : rbaibaidj@126.com, 18600369643

from lib.io import re, np

organ_name_re = {
    # 器官标准命名的正则化表达式，不考虑左右
    # 各ROI的先后顺序有意义。如果A字符串包含B，那么A应该在放在前面。
    # 比如BrainStem就应该在Brain之前检索。T10和T11应该在T1前面。
    "A_Aorta_Asc": r"a?[\W_]?ascen\W?a?",
    "A_Aorta_Desc":r"a?[\W_]?de?s?c?enden?d?s?\W?a?", #dsendensA.json
    "A_AorticArch": r"a?[\W_]?arch\W?a?",
    'Aorta': r'^aorta(?!(asc|de))$',
    # "Aorta": r"(a\W)?((de|a)scen\w*|arch)(\Wa)?",
    "Atrium": r"H?.?atrium",
    "BCV": r"V?.?BCV",
    "Bladder": r"bladd?er",
    "Body": r"skin|body",
    "BowelBag": "^(small)?[\W_]?(ins?[\W_]?(tes?tio?ne?s?)?|bowe?l)[\W_]*(bag)?",
    "BrainStem": r"^br(ai|ia)n[\W_]?stea?m$",
    "Brain": r"(brain|daxiaonao)$(?![\W]?stem)",
    "Breast": r"Breast",
    "Bronchus":r'pb|(bronch(ia|us))',
    "CCA": r"a?[\W]?CCA[\W]?",
    "Cerebellum": r"Cerebellum",
    "Cerebrum": r"Cerebrum",
    "Cochlea": r"^(cochlea|innea?r|ie|ei|ear)",
    "Colon":r'colon',
    "Duodenum":r'Duodenum',
    "Esophagus": r"eso[\W_]*[\W_]*",
    "Eye": r"((eye[\W_]*(ball)?)|(retina))",
    "FemoralHead": r"(((fem(ur|or?a?l?))?[\W_]*head[\W_]*(of)?[\W_]*(fem(ur|or?a?l?))?)|(hip joint))",
    "Femur": r"fei?m(ur|or?a?l?)",
    "Heart": r"heart[\W_]*",
    "IJV": r"V?.?IJV",
    "IMA": r"A?.?IMA",
    "IVC": r"V?.?IVC",
    "Kidney": r"(k(id?ne?y?)?|renal)",
    "Larynx": r"(larynx|(oro?ph(a?r?)?y?nx(ary)?))",
    "Lens": r"(len|crystal)s?",
    "Liver": r"^l[ie]ver[\W_]*(ai)?",
    "Lung": r"lung",
    "Mandible": r"mandible(?!(j(oi|io)nt))",
    "OpticChiasm": r"(optic(al)?[\W_]*(nerv)?)?[\W_]*chi?as?r?ma?",
    "OpticNerve": r"^(o(ptic(al)?)?)[\W_]*(n(erve)?)",
    "OralCavity": r"^(oral[\W_]*cavit?y|oc|mouth)",
    "Pancreas": r"pan((creas)|[\W_]*)?",
    "Parotid": r"(par[io]t?id([\W_]?g(land)?)?|pg)",
    "PelvicBone": r"(pelvic)?(bones?)\W?(of)?\W?(pelvi[sc])?",
    "Pituitary": r"(chui[\W_]*ti)|(pit?ui?t(ar|ra)(y|ium)[\W_]*(gland)?)",
    "Pulmonary.A": r"^a?.?pulmonary.?a?",
    "Pulmonary.V": r"^v?.?pulmonary.?v?",
    "Rectum": r"reca?tum",
    "Scleido.M": r"m?.?scleido.?m?",
    "SMG": r"^(SMG|she[\W]?xiaxian|submand)",
    "SpinalCord": r"(?!ex)(sp(ina|ian)l)?[\W_]*(cord|canal)$",
    "Spleen": r"^Sp(leen)?(?!inal)",
    "Sternohyoid.M": r"M.anterior cervi",
    "Stomach": r"St[ao]mach[\W_]*(ai)?",
    "Subclavian.A": r"A?.?subclavian.?a?",
    "SVC": r"V?.?SVC",
    "TemporalLobe": r"^(((lobe)?[\W_]*(temp(o?r?a?l?))[\W_]*(lobe)?)|tem|tml)",
    "Thyroid": r"thy?r(io|oi)d[\W_]*(gland)?(ai)?",
    "TMJ": r"(mandible[\W_]?(j(oi|io)nt)|tmj)",
    "Tongue": r"Tongue",
    "Trachea": r"t[rh][ea]chea?", #((principal)\W*bro(nch(ia|us))?|
    "Ventricle": r"h?[\W_]?(ventricle)",
    "Vertebral.A": r"A?.?vertebral.?a?",

    # 'ThoracicVertebra': r'^t[0-9]+$',
    # 'LumbarVertebra' :r'^l[0-9]+$',
    'T2': '^t2$', 'T3': '^t3$', 'T4': '^t4$', 'T5': '^t5$', 'T6': '^t6$',
    'T7': '^t7$', 'T8': '^t8$', 'T9': '^t9$', 'T10': '^t10$', 'T11': '^t11$', 'T12': '^t12$',
    'L1': '^l1$', 'L2': '^l2$', 'L3': '^l3$', 'L4': '^l4$', 'L5': '^l5$', 'T1': '^t1$',


}

rois_by_bodypart = {"head": ["Brain", "TemporalLobe_R", "TemporalLobe_L",
                             "Eye_L", "Eye_R", "OpticNerve_L", "OpticNerve_R",
                             "Lens_L", "Lens_R", "Len_L", "Len_R", "BrainStem",
                             "Cerebellum", "Cochlea_L", "Cochlea_R", "Pituitary",
                             "Parotid_L", "Parotid_R", "Mandible_L", "Mandible_R",
                             "TMJ_L", "TMJ_R"],

                    "neck": ["Trachea", "Thyroid", "Larynx", "Scleido_M", "Sternohyoid_M"],

                    "chest": ["Esophagus", "Heart", "Lung_L", "Lung_R",
                              'Bronchus', 'Bronchus_R', 'Bronchus_L', "Trachea"],

                    "abdomen": ["Liver", "Stomach", "Pancreas", "Spleen",
                                "Kidney_L", "Kidney_R", "SmallIntestine"],
                    "pelvis": ["Bladder", "BowelBag", "PelvicBone", "Rectum", "Sigmoid",
                               "FemoralHead_L", "FemoralHead_R", "Femur_L", "Femur_R",
                               'Ovary_R', 'Ovary-L', 'Ovary-r'],

                    "longorgan": ["Body", "SpinalCord"]
                    }

def if_not_none(input_obj):
    if input_obj == None:
        rst = False
    else:
        rst = True

    return rst

#   if bool(re.search(r'[\s\-\_\.\,]', words)):
#   prefix_template = re.compile(r'^%s[\s\-\_\.\,]?(.*)' % left_right, re.M | re.I)

def re_search_left_right(left_right, words, ignore_white_space = False):
    """
     必须以l/r开头和结尾的才匹配
    有的可能会议AI结尾
    :param left_right:
    :param words:
    :param ignore_white_space:
    :return:
    """

    white_space = '[\s\-\_\.\,]'
    suffix_template = re.compile(r'(.*)%s{1,3}(%s|%s%s{1,3}ai)$' #{1,3}表示某个字符重复多少次
                                 % (white_space, left_right, left_right,
                                    white_space), re.M | re.I)

    if ignore_white_space:
        prefix_template = re.compile(r'^%s%s*(.*)' % (left_right, white_space), re.M | re.I)
    else:
        prefix_template = re.compile(r'^%s%s{1,3}(.*)' % (left_right, white_space), re.M | re.I)

    res1 = bool(suffix_template.match(words))
    res2 = bool(prefix_template.match(words))
    # res1 = if_not_none(res1)
    # res2 = if_not_none(res2)
    res = res1 or res2
    return res


def search_left_right(words, ignore_white_space = False, orient_root=("l", "left", "lef")):
    """
    判断roi命名中是否存在左右
    :param words: roi的字符
    :param ignore_white_space:
    :param orient_root: 检索的方位词根
    :return: 1/0
    """
    if_exist = False
    for r in orient_root:
        if_exist = if_exist or re_search_left_right(r, words, ignore_white_space)
    return if_exist


def judge_direction(target_name, ignore_white_space):
    """
    判断目标roi是否包含左右，范围标准左右
    :param target_name:
    :param ignore_white_space:
    :return:
    """
    if_left = search_left_right(target_name, ignore_white_space, orient_root=("l", "left", "lef"), )
    if_right = search_left_right(target_name, ignore_white_space, orient_root=("r", "right", "rig"))

    if (if_left and if_right) or (not if_left and not if_right):
        return None
    elif if_left and not if_right:
        return 'L'
    else:
        return 'R'


def match_std_names(target_name, organ_name_re = organ_name_re, exclude_string=r'tv|pv|time'):
    """
    找到医生命名roi的标准命名
    :param target_name: 医生命名roi
    :param organ_name_re: 器官标准命名和正则化表达式的对应集合
    :param exclude_string: 排除字段
    :return: 标准命名
    """
    # print(target_name)
    for std_name, re_name in organ_name_re.items():
        # if not bool(re.match(r'bronchus', std_name, re.I)):
        #     continue
        # print(std_name, re_name)
        inc = bool(re.search(re_name, target_name, re.I)) or bool(re.search(std_name, target_name, re.I))
        exc = bool(re.search(exclude_string, target_name, re.I))

        if bool(re.match(r'bronchus', std_name, re.I)):
            rl_status = judge_direction(target_name, ignore_white_space=True)
        else:
            rl_status = judge_direction(target_name, ignore_white_space=False)

        if rl_status is not None:
            std_name = '_'.join([std_name, rl_status])
        if inc and not exc:
            return std_name
    return None


def rois_in_rt_w_target(rois_in_rt, info_dict):
    """
    匹配RS中roi的标准命名，并且筛选出需要的ROI
    :param rois_in_rt:
    :param info_dict:
    :return:
    """
    info_dict.roi_names, info_dict.std_names, info_dict.target_roi_ind = None, None, None
    if rois_in_rt is not None:
        print('     rt contains the following rois %s' % rois_in_rt)
        # 读取RT中存在的所有ROI，及其标准命名
        std_names = [match_std_names(n, organ_name_re, exclude_string='(pv|tv|help)') for
                     n in rois_in_rt]
        oar_index = [ix for ix in range(len(std_names)) if std_names[ix] is not None]
        oar_raw_names = [rois_in_rt[ix] for ix in oar_index]
        oar_std_names = [std_names[ix] for ix in oar_index]
        info_dict.roi_names, info_dict.std_names = oar_raw_names, oar_std_names

        # 获取指定ROI的索引号
        std_names = np.array([n.lower() if n is not None else None for n in std_names])
        # RS中可能存在多个指向同一标准命名的roi
        target_roi_ind = [list(np.where(std_names == r.lower())[0]) for r in info_dict.roi_include]
        info_dict.target_rois_index = target_roi_ind
        print('     index of target rois %s are %s' % (info_dict.roi_include, str(target_roi_ind)))

    return info_dict


def scan_rois4organ_at_risk(rois_in_rt, info_dict):
    """
    找到RS中roi的标准命名
    :param rois_in_rt:
    :param info_dict:
    :return:
    """
    oar_raw_names, oar_std_names = ['na'], ['na']
    if rois_in_rt is not None:
        std_names = [match_std_names(n, organ_name_re, exclude_string='(pv|tv|help)') for
                     n in rois_in_rt]
        oar_index = [ix for ix in range(len(std_names)) if std_names[ix] is not None]

        oar_raw_names = [rois_in_rt[ix] for ix in oar_index]
        oar_std_names = [std_names[ix] for ix in oar_index]

    info_dict.roi_names, info_dict.std_names = oar_raw_names, oar_std_names
    return info_dict

if __name__ == '__main__':
    rois = ['T10', 'T11', 'T12', 'T1']
    std_names = [match_std_names(r, organ_name_re) for r in rois]
    print(std_names)

    # a = judge_direction('Lung_R', ignore_white_space=True)