import numpy as np
import pickle as pkl
from batchgenerators.utilities.file_and_folder_operations import *

path  = r'nnUNetPlansv2.1_plans_3D.pkl'

with (open(path, 'rb')) as f:
    s = pkl.load(f)
    print(s['plans_per_stage'][0]['batch_size'])
    print(s['plans_per_stage'][0]['patch_size'])

    plans = load_pickle(path)
    plans['plans_per_stage'][0]['batch_size'] = 4
    plans['plans_per_stage'][1]['batch_size'] = 4


    save_pickle(plans, join(path))  # 路径的保存必须以_plans_xD.pkl结尾才能被识别