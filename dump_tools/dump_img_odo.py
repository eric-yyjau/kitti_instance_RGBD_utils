# NAME: loader.py

"""
# DESCRIPTION: data loader for raw kitti data
For other datasets (apollo, euroc, tum), please refer to dump_img_odo_tum.py
"""

import os
import sys
# sys.path.append('/home/ruizhu/Documents/Projects/kitti_instance_RGBD_utils/deepSfm_ori/FME')

import numpy as np 
import scipy.misc
import os
import cv2
from glob import glob
import time

from path import Path
from tqdm import tqdm
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from torch.utils.data import Dataset

# for test
# from config import get_config
# config, unparsed = get_config()

import argparse
from pebble import ProcessPool
import multiprocessing as mp
# ratio_CPU = 0.5
# default_number_of_process = int(ratio_CPU * mp.cpu_count())
default_number_of_process = 1 # to prevent congestion; SIFT and matrix operations in recfity points already takes advantage of multi-cores
import time

parser = argparse.ArgumentParser(description='Foo')
parser.add_argument("--dataset_dir", type=str, default="/data/KITTI/raw_meta/", help="path to dataset")   
parser.add_argument("--num_threads", type=int, default=default_number_of_process, help="number of thread to load data")
parser.add_argument("--cam_id", type=str, default='02', help="number of thread to load data")
parser.add_argument("--img_height", type=int, default=376, help="number of thread to load data")
parser.add_argument("--img_width", type=int, default=1241, help="number of thread to load data")
parser.add_argument("--static_frames_file", type=str, default=None, help="static data file path")
parser.add_argument("--test_scene_file", type=str, default=None, help="test data file path")
parser.add_argument('--dump', action='store_true', default=False)
parser.add_argument("--with_X", action='store_true', default=False,
                    help="If available (e.g. with KITTI), will store visable rectified lidar points ground truth along with images, for validation")
parser.add_argument("--with_pose", action='store_true', default=True,
                    help="If available (e.g. with KITTI), will store pose ground truth along with images, for validation")
parser.add_argument("--with_sift", action='store_true', default=False,
                    help="If available (e.g. with KITTI), will store SIFT points ground truth along with images, for validation")
parser.add_argument("--with_SP", action='store_true', default=False,
                    help="If available (e.g. with KITTI), will store SuperPoint points ground truth along with images, for validation")
parser.add_argument("--dump_root", type=str, default='dump', help="Where to dump the data")

# args = parser.parse_args('--dump --with_X --with_pose --with_sift \
#     --static_frames_file /home/ruizhu/Documents/Projects/SfmLearner-Pytorch/data/static_frames.txt \
#     --test_scene_file /home/ruizhu/Documents/Projects/SfmLearner-Pytorch/data/test_scenes.txt \
#     --dataset_dir /home/ruizhu/Documents/Datasets/kitti/raw \
#     --dump_root /home/ruizhu/Documents/Datasets/kitti/corr_dump'.split())
args = parser.parse_args()
print(args)

# %reload_ext autoreload
# %autoreload 2

from kitti_odo_loader import KittiOdoLoader
assert args.cam_id in ['00', '02'], 'Only supported left greyscale/color cameras (cam 00 or 02)!'
data_loader = KittiOdoLoader(args.dataset_dir,
                             img_height=args.img_height,
                             img_width=args.img_width,
                             cam_ids=[args.cam_id],
                             get_X=args.with_X,
                             get_pose=args.with_pose,
                             get_sift=args.with_sift, 
                             get_SP=args.with_SP)

# drive_path_test = data_loader.get_drive_path('2011_09_28', '0016')
# data_loader.scenes = [drive_path_test]
# data_loader.scenes = data_loader.scenes[:10] # List of Paths
n_scenes = {'train': len(data_loader.scenes['train']), 'test': len(data_loader.scenes['test'])}
print('Found %d potential train scenes, and %d test scenes.'%(n_scenes['train'], n_scenes['test']))

args_dump_root = Path(args.dump_root)
args_dump_root.mkdir_p()

sample_name_lists = []
for split in ['train', 'test']:
    print('> Retrieving frames for %s...'%split)
    seconds = time.time()
    def dump_scenes_from_drive(args, split, drive_path):
        # scene = data_loader.collect_scene_from_drive(drive_path)
        sample_name_list = data_loader.dump_drive(args, drive_path, split=split, scene_data=None)
        return sample_name_list

    if args.num_threads == 1:
        for drive_path in tqdm (data_loader.scenes[split]):
            print('Dumping ', drive_path)
            sample_name_list = dump_scenes_from_drive(args, split, drive_path)
            if split=='train':
                sample_name_lists.append(sample_name_list)
            # time.sleep(10)
    # else:
    #     with ProcessPool(max_workers=args.num_threads) as pool:
    #         tasks = pool.map(dump_scenes_from_drive, [args]*n_scenes[split], [split]*n_scenes[split], data_loader.scenes[split])
    #         try:
    #             for result in tqdm(tasks.result(), total=n_scenes[split]):
    #                 if split=='train':
    #                     sample_name_lists.append(result)
    #         except KeyboardInterrupt as e:
    #             tasks.cancel()
    #             raise e
    print("<<< Finished dump %s scenes. "%split, time.time() - seconds)

sample_name_flat_list = [item for sublist in sample_name_lists for item in sublist]

print('> Generating train val lists from %d samples...'%len(sample_name_flat_list))
np.random.seed(8964)
val_ratio = 0.2
# to avoid data snooping, we will make two cameras of the same scene to fall in the same set, train or val
subdirs = args_dump_root.dirs() # e.g. Path('./data/kitti_dump/2011_09_30_drive_0034_sync_02')
# canonic_prefixes = set([subdir.basename()[:-2] for subdir in subdirs]) # e.g. '2011_09_28_drive_0039_sync_'
with open(args_dump_root / 'train.txt', 'w') as tf:
    with open(args_dump_root / 'val.txt', 'w') as vf:
        for pr in tqdm(sample_name_flat_list):
            # corresponding_dirs = args_dump_root.dirs('{}*'.format(pr)) # e.g. [Path('./data/kitti_dump/2011_09_30_drive_0033_sync_03'), Path('./data/kitti_dump/2011_09_30_drive_0033_sync_02')]
            if np.random.random() < val_ratio:
            # if pr[:2] in ['06', '07', '08', '09', '10']:
                vf.write('{}\n'.format(pr))
            else:
                tf.write('{}\n'.format(pr))