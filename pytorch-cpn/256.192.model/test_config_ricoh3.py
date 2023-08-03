import os
import os.path
import sys
import numpy as np

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        
class Config:
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir_name = cur_dir.split('/')[-1]
    root_dir = os.path.join(cur_dir, '..')

    model = 'LSBN_CPN50' 

    num_class = 14
    img_path = os.path.join(root_dir, '/data01/', 'PoseInTheDark/PID_OTHER')
    symmetry = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
    bbox_extend_factor = (0.1, 0.15) # x, y

    pixel_means = np.array([122.7717, 115.9465, 102.9801]) # RGB
    data_shape = (256, 192)
    output_shape = (64, 48)


    use_GT_bbox = True
    # if use_GT_bbox:
    gt_path = os.path.join(root_dir, '256.192.model/Annotations/ExLPose-OCN','ExLPose-OC_test_RICOH3_trans.json')
    ori_gt_path = os.path.join(root_dir, '256.192.model/Annotations/ExLPose-OCN','ExLPose-OC_test_RICOH3.json')

cfg_test = Config()
add_pypath(cfg_test.root_dir)
add_pypath(os.path.join(cfg_test.root_dir, 'cocoapi/PythonAPI'))