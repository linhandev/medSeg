#导入BosClient配置文件
import bos_conf

#导入BOS相关模块
from baidubce import exception
from baidubce.services import bos
from baidubce.services.bos import canned_acl
from baidubce.services.bos.bos_client import BosClient

from tqdm import tqdm
import os

#新建BosClient
bos_client = BosClient(bos_conf.config)

bucket_name = 'lits'

local_dir = '/tmp/data/preprocess/'
bos_dir = 'data/prep_raw/'

for file_name in tqdm(os.listdir(local_dir)):
	# print(file_name)
	bos_client.put_object_from_file(bucket_name, bos_dir + file_name, os.path.join(local_dir, file_name) )
	# input('pause')
