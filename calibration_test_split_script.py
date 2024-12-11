import os
import shutil

from dataset_splitter import copy_n_random_files

src_path = "./data/chest_xray/test"
calib_path="./data/calibration"
test_path = "./data/test"

class1= "PNEUMONIA"
class2 = "NORMAL"
test_calib_split_ratio = 0.5

if os.path.exists(calib_path):
    shutil.rmtree(calib_path)

if os.path.exists(test_path):
    shutil.rmtree(test_path)

copy_n_random_files(230, os.path.join(src_path, class1), os.path.join(test_path, class1), os.path.join(calib_path, class1), test_calib_split_ratio)
copy_n_random_files(230, os.path.join(src_path, class2), os.path.join(test_path, class2), os.path.join(calib_path, class2), test_calib_split_ratio)