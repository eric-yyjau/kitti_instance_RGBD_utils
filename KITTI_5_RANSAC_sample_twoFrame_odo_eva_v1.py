#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import os
import sys
module_path = os.path.abspath(os.path.join('../deepSfm'))
if module_path not in sys.path:
    sys.path.append(module_path)
print("module path: ", module_path)


# In[41]:


from KITTI_5_RANSAC_sample_twoFrame_odo_eva import *
from KITTI_5_RANSAC_sample_twoFrame_odo_eva import get_error_from_sequence


# In[54]:


# def main():
parser = argparse.ArgumentParser(description='Foo')
# parser.add_argument('config', type=str)
parser.add_argument("--dataset_dir", type=str, default="/data/kitti/odometry", help="path to dataset")   
parser.add_argument("--num_threads", type=int, default=1, help="number of thread to load data")
parser.add_argument("--img_height", type=int, default=376, help="number of thread to load data")
parser.add_argument("--img_width", type=int, default=1241, help="number of thread to load data")
# parser.add_argument("--img_height", type=int, default=370, help="number of thread to load data")
# parser.add_argument("--img_width", type=int, default=1226, help="number of thread to load data")
parser.add_argument('--dump', action='store_true', default=False)
parser.add_argument("--with_X", action='store_true', default=False,
                    help="If available (e.g. with KITTI), will store visable rectified lidar points ground truth along with images, for validation")
parser.add_argument("--with_pose", action='store_true', default=True,
                    help="If available (e.g. with KITTI), will store pose ground truth along with images, for validation")
parser.add_argument("--with_sift", action='store_true', default=False,
                    help="If available (e.g. with KITTI), will store SIFT points ground truth along with images, for validation")
parser.add_argument("--dump_root", type=str, default='dump', help="Where to dump the data")

args = parser.parse_args('--dump --with_pose --with_X     --dataset_dir /data/kitti/odometry     --dump_root /data/kitti/odometry/dump_tmp'.split())
print(args)



#### load config ####

from utils.utils import loadConfig
# filename = '../deepSfm/configs/superpoint_coco_test.yaml'
filename = '../deepSfm/configs/superpoint_kitti_test.yaml'
# filename = args.config
config = loadConfig(filename)
print("config path: ", filename)
print("config: ", config)

# load kitti data



# In[ ]:


data_loader, scene_data = loadKitti(args)


# In[55]:


# # Get ij

i = 23
delta_ij = 3
j = i + delta_ij
X_rect_i, X_rect_i_vis, delta_Rtij, delta_Rtij_inv, img1_rgb, img2_rgb = get_ij(i, j, data_loader, scene_data, visualize=False)

# write to files
f = open(config['save_file'], "a")
print("="*50, file=f)
print("="*20, " start ", "="*20, file=f)
print("="*50, file=f)
import datetime
print(datetime.datetime.now(), file=f)


print("config path: ", filename, file=f)

error_names = ['dsac', 'opencv_Rt', 'epi_dist_mean']
errors = {error_name:[] for error_name in error_names}
print(errors)

errors = get_error_from_sequence(data_loader, scene_data, args, config, errors, file=f)


# In[61]:



err = errors['epi_dist_mean']
thd_1, thd_2 = 0.1, 1
accs, num_corrs = [], []
if config['five_point']:
    tag = '5_point'
else:
    tag = '8_point'
    
print("="*20, " ", tag, " ", "="*20, file=f)
for e in err:
    errs = output_epi_dist_mean_est_5p(e, thd_1, thd_2, tag=tag, file=f)
    accs.append(np.array(errs))
    num_corrs.append(e.shape[0])
accs = np.array(accs)
accs_mean = accs.mean(axis=0)
num_corrs_mean = np.array(num_corrs).mean()
print("average number of correspondences: %.2f"%num_corrs_mean, file=f)
print("percentage of inliers over %d frames, thd=(%.2f, %.2f): (%.2f, %.2f)"%      (accs.shape[0], thd_1, thd_2, accs_mean[0], accs_mean[1]), file=f)


# In[62]:


tag = 'opencv_Rt'
errors_Rt = np.array(errors['opencv_Rt'])
print("="*20, " ", tag, " ", "="*20, file=f)
for e in errors_Rt:
    print('error_R = %.4f, error_t = %.4f'%(e[0], e[1]), file=f)

thds = np.array([0.1, 1, 10])
# print("errors_Rt: ", errors_Rt)
rt_inliers = output_Rt_est(errors_Rt, thds, tag=tag, file=f, verbose=True)

# print('length: ', len(dsac_errors))
# print('DSAC:'); get_mean_std(dsac_errors)
# print('OpenCV 5-point:'); get_mean_std(opencv_5point_errors)
# print('OpenCV 8-point:'); get_mean_std(opencv_8point_errors)


# In[ ]:





# In[60]:


print(errors_Rt)


# In[59]:


print("="*20, "   end   ", "="*20, file=f)
print("="*50, file=f)
f.close()
f=None


# In[ ]:




