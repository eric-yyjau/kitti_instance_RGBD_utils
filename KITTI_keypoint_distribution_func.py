#!/usr/bin/env python
# coding: utf-8

# # Kitti odo loader

# In[1]:


import os
import sys
import numpy as np 
import scipy.misc
import os, sys
import cv2
# from glob import glob
# sys.path.append('/home/ruizhu/Documents/Projects/kitti_instance_RGBD_utils/kitti_tools')
sys.path.append('/home/yoyee/Documents/kitti_instance_RGBD_utils/kitti_tools')
np.set_printoptions(precision=8, suppress=True)
from path import Path
from tqdm import tqdm
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from torch.utils.data import Dataset

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))
import argparse
from pebble import ProcessPool
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import torch

import dsac_tools.utils_F as utils_F
import dsac_tools.utils_geo as utils_geo
import dsac_tools.utils_misc as utils_misc
import dsac_tools.utils_vis as utils_vis
import dsac_tools.utils_opencv as utils_opencv

######

import os
import sys
module_path = os.path.abspath(os.path.join('../deepSfm'))
if module_path not in sys.path:
    sys.path.append(module_path)
print("module path: ", module_path)
# os.chdir('../')
# print(os.getcwd())

######

#### load kitti data ####
def loadKitti(args):
    from kitti_odo_loader_memoryHungry import KittiOdoLoader

    data_loader = KittiOdoLoader(args.dataset_dir,
                                 img_height=args.img_height,
                                 img_width=args.img_width,
                                 cam_ids=['02'],
                                 get_X=args.with_X,
                                 get_pose=args.with_pose,
                                 get_sift=args.with_sift)

    print(data_loader.scenes['train'])
    print(data_loader.scenes['test'])

    # drive_path_test = data_loader.get_drive_path(date_name, seq_name)
    # data_loader.scenes = [drive_path_test]

    # n_scenes = len(data_loader.scenes)
    # print('Found {} potential scenes'.format(n_scenes))

    # args_dump_root = Path(args.dump_root)
    # args_dump_root.mkdir_p()

    # print('== Retrieving frames')
    scene_list = data_loader.collect_scene_from_drive(data_loader.scenes['train'][1])
    scene_data = scene_list[0]

    print(scene_data['N_frames'])


    # # Verify my rectification

    # In[186]:


    idx = 0
    # im1 = plt.imread(scene_data['img_files'][idx])
    im1, _ = data_loader.load_image(scene_data, idx)
    print(im1.shape)

    P = scene_data['calibs']['P_rect_ori_dict']['02']
    X_rect = scene_data['X_cam2'][idx][scene_data['val_idxes'][idx], :]
    X_cam0 = scene_data['X_cam0'][idx][scene_data['val_idxes'][idx], :]
    print(scene_data['calibs']['im_shape'], scene_data['calibs']['zoom_xy'])
    X_show = X_rect if scene_data['cid']=='02' else X_cam0
    c, x1 = utils_vis.reproj_and_scatter(utils_misc.identity_Rt(), X_show, im1, visualize=True, param_list=[scene_data['calibs']['K'], scene_data['calibs']['im_shape']])

    velo_homo_trans = utils_misc.homo_np(X_cam0).T
    # # velo = load_velo_scan('/data/kitti/raw/2011_09_30/2011_09_30_drive_0016_sync/velodyne_points/data/0000000000.bin')[:, :3]
    # velo = load_velo_scan('/data/kitti/odometry/sequences/04/velodyne/000000.bin')[:, :3]
    # velo_homo = utils_misc.homo_np(velo)
    # velo_homo_trans = scene_data['calibs']['cam_2rect'] @ scene_data['calibs']['velo2cam'] @ velo_homo.T

    P = scene_data['calibs']['P_rect_ori_dict'][scene_data['cid']]
    print(P)
    x_homo = (P @ velo_homo_trans).T
    x = utils_misc.de_homo_np(x_homo)
    plt.figure(figsize=(30, 8))
    plt.imshow(im1, cmap=None if len(im1.shape)==3 else plt.get_cmap('gray'))
    val_inds = utils_vis.scatter_xy(x, x_homo[:, 2], scene_data['calibs']['im_shape'], '', new_figure=False, set_lim=True)
    print(x-x1)

    return data_loader, scene_data

def get_ij(i, j, data_loader, scene_data, visualize=True, verbose=False):
    from numpy.linalg import inv
    im1, _ = data_loader.load_image(scene_data, i)
    im2, _ = data_loader.load_image(scene_data, j)
    pose1 = scene_data['poses'][i]
    pose2 = scene_data['poses'][j]
    delta_Rtij = inv(utils_misc.Rt_pad(pose2)) @ utils_misc.Rt_pad(pose1) # pose is for **camera**(0) pose; delta_Rij is for **scene** motion

    X_rect_i = scene_data['X_cam2'][i][scene_data['val_idxes'][i], :]
    X_cam0_i = scene_data['X_cam0'][i][scene_data['val_idxes'][i], :]
    X_rect_j = scene_data['X_cam2'][j][scene_data['val_idxes'][j], :]
    X_cam0_j = scene_data['X_cam0'][j][scene_data['val_idxes'][j], :]

    P = scene_data['calibs']['P_rect_ori_dict'][scene_data['cid']]

    # Vis1: with recfitied proj on Cam_cid
    if scene_data['cid']=='02':
        print('Recfitying delta_Rtij to cam %s.'%scene_data['cid'])
        delta_Rtij_camid = scene_data['calibs']['Rtl_gt'] @ delta_Rtij @ inv(scene_data['calibs']['Rtl_gt'])
    else:
        delta_Rtij_camid = delta_Rtij
    
    delta_Rtij_1 = utils_misc.normalize_Rt_to_1(delta_Rtij)
    delta_Rtij_T_1 = utils_misc.normalize_Rt_to_1(scene_data['calibs']['Rtl_gt'] @ delta_Rtij @ inv(scene_data['calibs']['Rtl_gt']))
    if verbose:
        print(delta_Rtij_1)
        print(delta_Rtij_T_1)
        print(utils_geo.vector_angle(delta_Rtij_1[:, 3:4], delta_Rtij_T_1[:, 3:4]))
    
    delta_Rtij_camid_inv = inv(delta_Rtij_camid)
    
    param_list = [scene_data['calibs']['K'], scene_data['calibs']['im_shape']]
    
    X_show_i = X_rect_i if scene_data['cid']=='02' else X_cam0_i
    utils_vis.reproj_and_scatter(utils_misc.identity_Rt(), X_show_i, im1, visualize=visualize, title_appendix='[1]-%d_to_i%d'%(i, j), param_list=param_list)
    _, x1 = utils_vis.reproj_and_scatter(utils_misc.Rt_depad(delta_Rtij_camid), X_show_i, im2, visualize=visualize, title_appendix='[1]-%d_to_%d'%(i, j), param_list=param_list, set_lim=True)
    #     print(X_show_i, im2.shape)
    if verbose: print(x1)
    #     utils_vis.reproj_and_scatter(utils_misc.identity_Rt(), X_rect_j, im2, visualize=True, title_appendix='[1]-j_to_j', param_list=param_list)

    # Vis2: with vanilla proj on Cam 0
    # # i to i
    x_homo_i = (P @ utils_misc.homo_np(X_cam0_i).T).T
    x_i = utils_misc.de_homo_np(x_homo_i)
    if visualize:
        plt.figure(figsize=(30, 8))
        plt.imshow(im1, cmap=None if len(im1.shape)==3 else plt.get_cmap('gray'))
        val_inds = utils_vis.scatter_xy(x_i, x_homo_i[:, 2], scene_data['calibs']['im_shape'], title='[2]-i_to_i', new_figure=False, set_lim=True)

    # i to j
    x_homo_j = (P @ delta_Rtij @ utils_misc.homo_np(X_cam0_i).T).T
    x_j = utils_misc.de_homo_np(x_homo_j)
    if visualize:
        plt.figure(figsize=(30, 8))
        plt.imshow(im2, cmap=None if len(im2.shape)==3 else plt.get_cmap('gray'))
        val_inds = utils_vis.scatter_xy(x_j, x_homo_j[:, 2], scene_data['calibs']['im_shape'], title='[2]-i_to_j', new_figure=False, set_lim=True)
    #     print(x_j)
    
    # # j to j
    # x_homo_j = (P @ utils_misc.homo_np(X_cam0_j).T).T
    # x_j = utils_misc.de_homo_np(x_homo_j)
    # plt.figure(figsize=(30, 8))
    # plt.imshow(im2)
    # val_inds = utils_vis.scatter_xy(x_j, x_homo_j[:, 2], scene_data['calibs']['im_shape'], title='[2]-j_to_j', new_figure=False, set_lim=True)

    # print(x1)
    if verbose: 
        print(x_j-x1)
    
    angle_R = utils_geo.rot12_to_angle_error(np.eye(3), delta_Rtij_camid_inv[:3, :3])
    angle_t = utils_geo.vector_angle(np.array([[0.], [0.], [1.]]), delta_Rtij_camid_inv[:3, 3:4])
    print('>>>>>>>>>>>>>>>> Between frame %d and %d: \nThe rotation angle (degree) %.4f, and translation angle (degree) %.4f'%(i, j, angle_R, angle_t))
    
    if verbose: print(delta_Rtij_camid_inv[:3, 3:4])
    
    return X_rect_i, X_rect_i, utils_misc.Rt_depad(delta_Rtij_camid), utils_misc.Rt_depad(delta_Rtij_camid_inv), im1, im2

# ## val_rt function
import dsac_tools.utils_F as utils_F
import dsac_tools.utils_opencv as utils_opencv

def val_rt(idx, K_np, x1_single_np, x2_single_np, E_est_np, E_gt_np, F_est_np, F_gt_np, delta_Rtijs_4_4_cpu_np, five_point, if_opencv=True):
    delta_Rtij_inv = np.linalg.inv(delta_Rtijs_4_4_cpu_np)[:3]

    error_Rt_estW = None
    epi_dist_mean_estW = None
    error_Rt_opencv = None
    epi_dist_mean_opencv = None

    # Evaluating with our weights
    # _, error_Rt_estW = utils_F._E_to_M(E_est.detach(), K, x1_single_np, x2_single_np, w>0.5, \
    #     delta_Rtij_inv, depth_thres=500., show_debug=False, show_result=False, method_name='Est ws')
    M_estW, error_Rt_estW = utils_F.goodCorr_eval_nondecompose(x1_single_np, x2_single_np, E_est_np.astype(np.float64), delta_Rtij_inv, K_np, None)
    M_gt, error_Rt_gt = utils_F.goodCorr_eval_nondecompose(x1_single_np, x2_single_np, E_gt_np.astype(np.float64), delta_Rtij_inv, K_np, None)
    epi_dist_mean_estW, _, _ = utils_F.epi_distance_np(F_est_np, x1_single_np, x2_single_np, if_homo=False)
    epi_dist_mean_gt, _, _ = utils_F.epi_distance_np(F_gt_np, x1_single_np, x2_single_np, if_homo=False)

    # print('-0', F_est_np, epi_dist_mean_estW)

    # Evaluating with OpenCV 5-point
    if if_opencv:
        M, error_Rt_opencv, _, E_return = utils_opencv.recover_camera_opencv(K_np, x1_single_np, x2_single_np, \
            delta_Rtij_inv, five_point=five_point, threshold=0.01, show_result=False)
        if five_point:
            E_recover_opencv = E_return
            F_recover_opencv = utils_F.E_to_F_np(E_recover_opencv, K_np)
        else:
            E_recover_opencv, F_recover_opencv = E_return[0], E_return[1]
        # print('+++', K_np)
        epi_dist_mean_opencv, _, _ = utils_F.epi_distance_np(F_recover_opencv, x1_single_np, x2_single_np, if_homo=False)
        # print('-0-', utils_F.E_to_F_np(E_recover_5point, K_np))
        # print('-1', utils_F.E_to_F_np(E_recover_5point, K_np), epi_dist_mean_5point)

    return error_Rt_estW, epi_dist_mean_estW, error_Rt_opencv, epi_dist_mean_opencv, idx, M_estW, epi_dist_mean_gt, epi_dist_mean_gt


def save_to_file(save_file, content, next_line=True):
    with open(save_file, "a") as myfile:
        myfile.write(content)
        if next_line:
            myfile.write('\n')


def print_config(config, file=None):
    print('='*10, ' important config: ', '='*10, file=file)
    for item in list(config):
        print(item, ": ", config[item], file=file)
    
    print('='*32)


def filter_points_np(points, shape_lo=[0,0], shape_hi=[1,1], return_mask=False):
    ### check!
    # points = points.float()
    # shape = shape.float()
    mask = (points >= shape_lo) * (points <= shape_hi-1)
    mask = (np.prod(mask, axis=-1) == 1)
    if return_mask:
        return points[mask], mask
    return points [mask]

class Val_model_kitti(object):
    def __init__(self, data_loader, scene_data, args, config, seed=0):
        np.random.seed(seed)
        np.set_printoptions(precision=8, suppress=True)

        config_eva = config['evaluations']
        self.config = config

        self.data_loader = data_loader
        self.scene_data = scene_data

        self.params = config['evaluations']['params']
        self.feature_mode = config['feature_mode']
        self.feature_type = self.get_feature_type(feature_mode=self.feature_mode)

        self.use_est_E = config_eva['use_est_E']
        self.five_point = config_eva['five_point']
        self.iter_max = config_eva['iter_max']
        self.delta_i = config_eva['frame_diff']
        self.st_frame = config_eva['starting_frame']
        self.frame_arr = self.get_shuffle_arr(scene_data['N_frames']-self.delta_i, 
                st_frame=self.st_frame, shuffle=config_eva['random_frame'])
        self.K= scene_data['calibs']['K'].astype(np.float)

        self.sift_result = {}
        self.sift_result['enable'] = False
        pass

    def sift_enable(self, mod=True):
        self.sift_result['enable'] = mod
    

    def load_model(self, file=None):
        if self.feature_mode == 2:
            config = self.config
            from train3 import SPInferLoader_heatmap as SPInferLoader
            output_dir = './'
            self.sp_inferrer = SPInferLoader(config, output_dir, args=None)
            print("config: ", config, file=file)
            pretrained_model = config['model']['pretrained']
            print("use model: ", pretrained_model, file=file)
        pass

    @staticmethod
    def get_feature_type(feature_mode=1):
        if feature_mode == 1:
            feature_type = 'sift'
            print("use model: sift")
        elif feature_mode == 2:
            feature_type = 'superpoint'
        return feature_type

    # get frame list
    @staticmethod
    def get_shuffle_arr(num, st_frame=0, shuffle=False):
        arr = np.arange(num)
        if shuffle:
            np.random.shuffle(arr)
        else:
            if st_frame < num:
                np.concatenate((arr[st_frame:], 
                    arr[:st_frame]), axis=0)
        return arr

    def getFeatures(self, img1, img1_rgb, img2, img2_rgb, visualize=False, feature_type='sift'):
        # Keypoint detection and matching with SIFT
        if feature_type == 'sift':
            x1_all, kp1, des1 = utils_opencv.SIFT_det(img1, img1_rgb, visualize=visualize)
            x2_all, kp2, des2 = utils_opencv.SIFT_det(img2, img2_rgb, visualize=visualize)
            x1, x2, _, _ = utils_opencv.KNN_match(des1, des2, x1_all, x2_all, kp1, kp2, img1_rgb, img2_rgb, visualize=visualize)
        elif feature_type == 'superpoint':
            sp_inferrer = self.sp_inferrer
            img1_rgb_np, img2_rgb_np = np.array(img1_rgb), np.array(img2_rgb)
            # Keypoint detection and matching with SuperPoint inference model
            sp_pred = sp_inferrer.run_two_imgs(sp_inferrer.img_array_to_input(img1_rgb_np), sp_inferrer.img_array_to_input(img2_rgb_np))
            matches = sp_inferrer.get_matches(sp_pred)
            x1 = matches[0][:, :2]
            x2 = matches[0][:, 2:4]
        return x1, x2

    @staticmethod
    def filter_center_np(points, shape, center=[0,0], return_mask=False):
        """
        param:
            points: [n, 2] (x, y)
            center: [2,]   (y, x)
            shape:  [2,]   (y, x)
        """
        ### check!
        # points = points.float()
        # shape = shape.float()

        # mask = (points >= 0) * (points <= shape-1)
        res = lambda x: np.flip(np.array(x), axis=0).reshape(1, 2)
        center = res(center)
        shape = res(shape)/2
        points = abs(points - center)
        print("points: ", points[:5])
        print("shape: ", shape)
        print("center: ", center)
        mask = points < shape # inside area is marked as true
        mask = (np.prod(mask, axis=-1) == 0) # filter out the inside area
        if return_mask:
            return points[mask], mask
        return points [mask]


    def block_center(self):
        if config_eva['block_center']['enable']:
            shape = config_eva['block_center']['size']
            div = lambda x: [ x/2.0 for x in img1_rgb_np.shape[:2]]
            _, m1 = filter_center_np(x1, shape, center=div(img1_rgb_np.shape[:2]), return_mask=True)
            # _, m2 = filter_center_np(x2, shape, center=div(img1_rgb_np.shape[:2]), return_mask=True)
            # m = m1*m2
            m = m1
            print("x1 before: ", x1.shape)
            x1_m, x2_m = x1[m], x2[m]
            x1_f, x2_f = x1[m==False], x2[m==False]
            print("x1_m after: ", x1_m.shape)
            print("x1_f after: ", x1_f.shape)
            if visualize:
                print("points not used")
                utils_vis.draw_corr(img1_rgb_np, img2_rgb_np, x1_f, x2_f, 1)
        else:
            x1_m = x1
            x2_m = x2
        return x1_m, x2_m

    @staticmethod
    def get_center(pts, x_l, x_r, y_l, y_r):
        pts_f = filter_points_np(pts, shape_lo=np.array([x_l, y_l]), shape_hi=np.array([x_r, y_r]))
        print("pts_f: ", pts_f.shape)
        if pts_f.shape[0] == 0:
            return {'centers': np.array([0,0]), 'num_points':0}
        print("pts_f min: ", pts_f.min(axis=0))
        print("pts_f max: ", pts_f.max(axis=0))
        print("pts_f mean: ", pts_f.mean(axis=0))
        results = {'centers': pts_f.mean(axis=0), 'num_points':pts_f.shape[0]}
        return results
        pass

    def plot_points_distri(self, img1_rgb_np, x1, block_size=3, visualize=False):
        # round and scatter x1
        img_h, img_w = img1_rgb_np.shape[0], img1_rgb_np.shape[1]
        # x1_int = x1.round().astype(int)
        # print("x1: ", x1.shape)
        # img_pts = np.zeros((img_h, img_w))
        # img_pts[x1_int[:,1], x1_int[:,0]] = 1
        # for to get the center points and size
        def get_centers_from_points(x1):
            results = {'centers': [], 'num_points': []}
            blk_h, blk_w = img_h//block_size, img_w//block_size
            for i in range(block_size):
                y = blk_h*i
                for j in range(block_size):
                    x = blk_w*j
                    data = self.get_center(x1, x, x+blk_w, y, y+blk_h)
                    results['centers'].append(data['centers'])
                    results['num_points'].append(data['num_points'])
                    pass
            for i in list(results):
                results[i] = np.stack(results[i], axis=0) # .squeeze(1)
                print(i, ": ", results[i].shape)
            return results
        results = get_centers_from_points(x1)
        print("results: ", results)

        
        if visualize:
            from kitti_tools.kitti_draw import vis_geoDist
            if self.sift_result['enable']:
                sifts = get_centers_from_points(self.sift_result['x1'])
                self.sift_result.update(sifts)
                vis_geoDist(img1_rgb_np, geo_dists=results['num_points'], x1=results['centers'], 
                    geo_dists_2=self.sift_result['num_points'], x2=self.sift_result['centers'],
                    mask=None, show=False)
            else:
                vis_geoDist(img1_rgb_np, results['num_points'], results['centers'], mask=None, show=False)
            plt.title('superpoint(red), sift(blue)')
            def plt_blocks(shape, block_size):
                """
                shape: [y, x]
                """
                for i in range(block_size):
                    plt.axhline(y=shape[0]//block_size*i, color='b', linestyle='-')
                    plt.axvline(x=shape[1]//block_size*i, color='b', linestyle='-')              
            plt_blocks(img1_rgb_np.shape[:2], block_size)

            plt.show()

        return results 
        pass

    def run(self, visualize=False):  
        count = 1
        while count < self.frame_arr.shape[0]:
            print('=' * 50)
            print('=' * 20, " start a pair ", '=' * 20)
            print("iter: ", count)
            
            i = self.frame_arr[count]
            #   clear_output()
            j = i + self.delta_i

            # Get two frames
            X_rect_i, X_rect_i_vis, delta_Rtij, delta_Rtij_inv, img1_rgb, img2_rgb = \
                    get_ij(i, j, self.data_loader, self.scene_data, visualize=False)

            #     print('-- delta_Rtij_gt (scene)\n', delta_Rtij)
            #     print('-- delta_Rtij_inv_gt (camera)\n', delta_Rtij_inv)
            E_gt_th, F_gt_th = utils_F._E_F_from_Rt(delta_Rtij[:, :3], delta_Rtij[:, 3:4], self.K)    
            F_gt = F_gt_th.numpy()
            E_gt_th_inv, F_gt_th_inv = utils_F._E_F_from_Rt(delta_Rtij_inv[:, :3], delta_Rtij_inv[:, 3:4], self.K)    
            F_gt_inv = F_gt_th_inv.numpy()
            R2s, t2s, M2s = utils_F._get_M2s(E_gt_th)
            #     print('-- E_gt (scene)\n', E_gt_th.numpy())

            img1_rgb_np, img2_rgb_np = np.array(img1_rgb), np.array(img2_rgb)
            img1, img2 = utils_opencv.PIL_to_gray(img1_rgb), utils_opencv.PIL_to_gray(img2_rgb)
            
            x1, x2 = self.getFeatures(img1, img1_rgb, img2, img2_rgb, visualize=visualize, 
                        feature_type=self.feature_type)
            if self.sift_result['enable']:
                print ("get sift results")
                self.sift_result['x1'], self.sift_result['x2'] = self.getFeatures(img1, img1_rgb, 
                        img2, img2_rgb, visualize=visualize, feature_type='sift')
                
            # if config_eva['round']:
                # x1, x2 = x1.round(), x2.round()
            print("x1: some points: ", x1[:5])
            print("img1_rgb_np: ", img1_rgb_np.shape)

            x1_m = x1
            x2_m = x2
            # visualize matches
            if visualize:
                print("points used for RANSAC")
                utils_vis.draw_corr(img1_rgb_np, img2_rgb_np, x1_m, x2_m, 1)

            self.results = self.plot_points_distri(img1_rgb_np, x1_m, visualize=visualize)

            count += 1
            if count > self.iter_max:
                break
            pass


    def _stash(self):
        # OpenCV results
        ## 5 point
        if five_point:
            print("use five_point")
        else:
            print("use eight_point")
            
        M, error_Rt_5point, mask2, E_return  = utils_opencv.recover_camera_opencv(K, x1_m, x2_m, delta_Rtij_inv, five_point=five_point,
            threshold=params['ransac_thresh'])
        K_np = K
        # x1_single_np = x1_m
        # x2_single_np = x2
        
        def get_E_F(five_point, E_return, K):
            if five_point:
                E_recover_opencv = E_return
                F_recover_opencv = utils_F.E_to_F_np(E_recover_opencv, K)
            else:
                E_recover_opencv, F_recover_opencv = E_return[0], E_return[1]
            return E_recover_opencv, F_recover_opencv

        if use_est_E:
            E_recover_opencv, F_recover_opencv = get_E_F(five_point, E_return, K_np)
            print("use estimated essential matrix for epipolar distance evaluation")
        else:
            E_recover_opencv = E_gt_th.numpy()
            F_recover_opencv = utils_F.E_to_F_np(E_recover_opencv, K_np)
        print("E_return: ", E_return)
        print("E_gt_th: ", E_gt_th.numpy())
        
        # use recovered essential matrix
        epi_dist_mean_5point, _, _ = utils_F.epi_distance_np(F_recover_opencv, 
                                            x1_m, x2_m, if_homo=False)
        
        errors['epi_dist_mean'].append(epi_dist_mean_5point)
        
        #     M, error_Rt_5point = utils_opencv.recover_camera_0(E_gt_th.numpy(), K, x2, x1, delta_Rtij_inv, five_point=True, threshold=0.001)
        errors['opencv_Rt'].append([error_Rt_5point[0], error_Rt_5point[1]])
        
        #     print("count: ", count, ", error: ", error_Rt_5point, ", epi_dist_mean_5point: ",  epi_dist_mean_5point)
        

        ## Check geo dists
        ##### need modification
        def checkGeoDist(x1, x2, geo_dists, geo_thres=1, visualize=False):
            # x1_normalizedK = utils_misc.de_homo_np((np.linalg.inv(K) @ utils_misc.homo_np(x1).T).T)
            # x2_normalizedK = utils_misc.de_homo_np((np.linalg.inv(K) @ utils_misc.homo_np(x2).T).T)
            # K_th = torch.from_numpy(K)
            # F_gt_normalized = K_th.t()@F_gt_th@K_th

            # geo_dists = utils_F._sym_epi_dist(F_gt_normalized, 
            #     torch.from_numpy(x1_normalizedK), torch.from_numpy(x2_normalizedK)).numpy()
            # geo_thres = 1e-4
            mask_in = geo_dists<geo_thres
            mask_out = geo_dists>=geo_thres


            if visualize:
                print("visualize: correspondences in mask 2")
                print("mask2: ", mask2.shape)
                print("x1: ", x1.shape)
                utils_vis.draw_corr(img1_rgb_np, img2_rgb_np, x1[mask2, :], x2[mask2, :], linewidth=2.)

                if config_eva['block_center']['enable']:
                    epi_dist, _, _ = utils_F.epi_distance_np(F_recover_opencv, 
                                                        x1_f, x2_f, if_homo=False)
                    print("visualize: scattered geometry distance (filtered points)")
                    vis_geoDist(img1_rgb_np, epi_dist, x1_f, mask=None)

                print("visualize: scattered geometry distance (w/i threshold)")
                vis_geoDist(img1_rgb_np, geo_dists.copy(), x1, mask=mask_in)
                # print("geo_dists max after: ", geo_dists.max())

                print("visualize: scattered geometry distance (out of threshold)")
                vis_geoDist(img1_rgb_np, geo_dists.copy(), x1, mask=mask_out)

                print("visualize: correspondences within geometry threshold")
                utils_vis.draw_corr(img1_rgb_np, img2_rgb_np, x1[mask_in, :], x2[mask_in, :], linewidth=2.)
        #     print(np.sort(geo_dists[mask2 & mask_in] / geo_thres))
                print("visualize: correspondences out of geometry threshold")

                line_widths = geo_dists[mask_out] / geo_thres
                line_widths[line_widths>10] = 10.
                utils_vis.draw_corr_widths(img1_rgb_np, img2_rgb_np, x1[mask_out, :], x2[mask_out, :], linewidth=line_widths, rescale=False, scale=2.)
        #     print(np.sort(geo_dists[mask2 & mask_out] / geo_thres)[::-1])
        #     print(x1[mask2, :] - x2[mask2, :])
        
        if checkGeoDist:
            print("check Geometry Dist!")
            print("x1_m after: ", x1_m.shape)
            checkGeoDist(x1_m, x2_m, epi_dist_mean_5point, geo_thres=1, visualize=visualize)
        
        def check_epipolar_contraints(x1, x2):
            ## Check epipolar constraints

            random_idx, _, _, colors = utils_opencv.sample_and_check(x1[mask2, :], x2[mask2, :], img1_rgb, img2_rgb, img1_rgb_np, img2_rgb_np, F_gt,                                                                  visualize=True, if_sample=False)

            F_est = np.linalg.inv(K).T @ E_recover_opencv @ np.linalg.inv(K)
            print("epipolor lines from estimated F matrix")
            _, _, _, _ = utils_opencv.sample_and_check(x1[mask2, :], x2[mask2, :], img1_rgb, img2_rgb, img1_rgb_np, img2_rgb_np, F_est, None,                                                 visualize=True, if_sample=False, colors=colors, random_idx=random_idx)

            print("epipolor lines from estimated ground truth essential matrix")
            _, _, _, _ = utils_opencv.sample_and_check(x1[mask2, :], x2[mask2, :], img1_rgb, img2_rgb, img1_rgb_np, img2_rgb_np, 
                                                       np.linalg.inv(K).T @ E_gt_th.numpy() @ np.linalg.inv(K), None, \
                                                    visualize=True, if_sample=False, colors=colors, random_idx=random_idx)
        #     colors = utils_opencv.show_epipolar_opencv(x1[mask2, :], x2[mask2, :], img1_rgb, img2_rgb, F_gt)
        #     _ = utils_opencv.show_epipolar_opencv(x1[mask2, :], x2[mask2, :], img1_rgb, img2_rgb, F_est, colors=colors)

        #     print('GT camera matrix: (camera)\n', delta_Rtij_inv)   

        if check_epipolar_contraints:
            check_epipolar_contraints(x1_m, x2_m)

        print(F_gt)
        print('=' * 20, " end a pair ", '=' * 20)
        print('=' * 50)
        

def output_epi_dist_mean_est_5p(epi_dist_mean_est_5p, thd_1, thd_2=None, tag='', file=None):
    thd_1 = 0.1
    err_1 = np.sum(epi_dist_mean_est_5p<thd_1)/epi_dist_mean_est_5p.shape[0]
    if thd_2 is not None:
        thd_2 = 1
        err_2 = np.sum(epi_dist_mean_est_5p<thd_2)/epi_dist_mean_est_5p.shape[0]
    else:
        thd_2 = -1
        err_2 = -1
    print('======= %s, thd=(%.2f, %.2f): %.2f, %.2f'%(tag, thd_1, thd_2, err_1, err_2), file=file)
    return [err_1, err_2]

def output_Rt_est(errors_Rt, thds, tag='', file=None, verbose=False):
    """
    Input:
        errors_Rt: np [n, 2] (Rotation, translation error)
        thds: np [k] (k different thds)

    return:
        np [k, 2]  (different thresholds, rotation/ translation)

    """
    num_frame = errors_Rt.shape[0]
    errors_Rt = np.expand_dims(errors_Rt, axis=1) # [n, 1, 2]
    thds = thds.reshape(1,-1, 1)
    inliers = errors_Rt < thds  # [n, k, 2]
    inliers = inliers.sum(axis=0)
    print("inliers dim: ", inliers.shape)
    # .squeeze(0) # [1, k, 2] --> [k, 2]
    inliers = inliers/num_frame

    if verbose:
        np.set_printoptions(precision=2)
        print('======= %s, thd: (%s), (%s)'%(tag, ', '.join(map(str, thds.squeeze())), 
            ', '.join(map(str, inliers))), file=file)
    return inliers



# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Foo')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')

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

    # cmd = '--dump --with_pose --with_X     --dataset_dir /data/kitti/odometry     --dump_root /data/kitti/odometry/dump_tmp'.split()
    args = parser.parse_args()
    print("args: ", args)
    print()


    #### load config ####

    from utils.utils import loadConfig
    # filename = '../deepSfm/configs/superpoint_coco_test.yaml'
    config_base = '../deepSfm/configs/'
    if args.config is not '':
        filename = config_base + args.config
    else:
        filename = config_base + 'superpoint_kitti_test.yaml'
    # filename = args.config
    print("config path: ", filename)
    config = loadConfig(filename)
    print("config: ", config)

    # load kitti data
    # if data_loader is None:
    data_loader, scene_data = loadKitti(args)

    # # Get ij
    from numpy.linalg import inv

    i = 23
    delta_ij = 3
    j = i + delta_ij
    X_rect_i, X_rect_i_vis, delta_Rtij, delta_Rtij_inv, img1_rgb, img2_rgb = get_ij(i, j, data_loader, scene_data, visualize=False)

    # write to files
    if config['save']['enable']:
        f = open(config['save']['path'], "a")
    else: 
        f = None


    # visualize images and feature points
    val_agent = Val_model_kitti(data_loader, scene_data, args, config)
    val_agent.get_features()
    val_agent.run()

    # print("="*50, file=f)
    # print("="*20, " start ", "="*20, file=f)
    # print("="*50, file=f)
    # import datetime
    # print(datetime.datetime.now(), file=f)

    # print("config path: ", filename, file=f)
    # print("config: ", config, file=f)

    # # get errors
    # error_names = ['dsac', 'opencv_Rt', 'epi_dist_mean']
    # errors = {error_name:[] for error_name in error_names}
    # print(errors)

    # errors = get_error_from_sequence(data_loader, scene_data, args, config, errors, file=f)

    # print errors
    ## epipolar distance error
    # err = errors['epi_dist_mean']
    # thd_1, thd_2 = 0.1, 1
    # accs, num_corrs = [], []
    # if config['evaluations']['five_point']:
    #     tag = '5_point'
    # else:
    #     tag = '8_point'
        
    # print("="*20, " ", tag, " ", "="*20, file=f)
    # for e in err:
    #     errs = output_epi_dist_mean_est_5p(e, thd_1, thd_2, tag=tag, file=f)
    #     # print("error: ", e)
    #     accs.append(np.array(errs))
    #     num_corrs.append(e.shape[0])
    # accs = np.array(accs)
    # accs_mean = accs.mean(axis=0)
    # num_corrs_mean = np.array(num_corrs).mean()
    # print("average number of correspondences: %.2f"%num_corrs_mean, file=f)
    # print("percentage of inliers over %d frames, thd=(%.2f, %.2f): (%.2f, %.2f)"%      (accs.shape[0], thd_1, thd_2, accs_mean[0], accs_mean[1]), file=f)    

    ## opencv error
    # tag = 'opencv_Rt'
    # errors_Rt = np.array(errors['opencv_Rt'])
    # print("="*20, " ", tag, " ", "="*20, file=f)
    # for e in errors_Rt:
    #     print('error_R = %.4f, error_t = %.4f'%(e[0], e[1]), file=f)

    # thds = np.array([0.1, 0.5, 1, 2, 5, 10])
    # # print("errors_Rt: ", errors_Rt)
    # rt_inliers = output_Rt_est(errors_Rt, thds, tag=tag, file=f, verbose=True)


    # ###### end ######
    # print("="*20, "   end   ", "="*20, file=f)
    # print("="*50, file=f)
    # if f is not None: f.close()
    # # f=None

    # ##### write to cvs #####
    # from utils.utils import append_csv
    # mode = config['feature_mode']
    # feature_type = get_feature_type(mode)
    # desc = feature_type if mode == 1 else feature_type + ': ' +  pretrained_model
    # append_csv(file=config['csv_file'], arr=[desc])
    # # inliers percentage
    # append_csv(file=config['csv_file'], arr=[thd_1, thd_2])
    # append_csv(file=config['csv_file'], arr=accs_mean)
    # # R t error
    # zero_insert = lambda x: np.dstack((x,np.zeros_like(x))).flatten()

    # append_csv(file=config['csv_file'], arr=thds)
    # append_csv(file=config['csv_file'], arr=rt_inliers.transpose().flatten())
    # append_csv(file=config['csv_file'], arr=rt_inliers.transpose())


# In[ ]:




