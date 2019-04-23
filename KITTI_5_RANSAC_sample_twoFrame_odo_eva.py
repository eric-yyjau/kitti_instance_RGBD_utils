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

def val_rt(idx, K_np, x1_single_np, x2_single_np, E_est_np, F_est_np, delta_Rtijs_4_4_cpu_np, if_opencv=True):
    delta_Rtij_inv = np.linalg.inv(delta_Rtijs_4_4_cpu_np)[:3]

    error_Rt_estW = None
    epi_dist_mean_estW = None
    error_Rt_5point = None
    epi_dist_mean_5point = None

    # Evaluating with our weights
    # _, error_Rt_estW = utils_F._E_to_M(E_est.detach(), K, x1_single_np, x2_single_np, w>0.5, \
    #     delta_Rtij_inv, depth_thres=500., show_debug=False, show_result=False, method_name='Est ws')
    M_estW, error_Rt_estW = utils_F.goodCorr_eval_nondecompose(x1_single_np, x2_single_np, E_est_np.astype(np.float64), delta_Rtij_inv, K_np, None)
    epi_dist_mean_estW, _, _ = utils_F.epi_distance_np(F_est_np, x1_single_np, x2_single_np, if_homo=False)
    # print('-0', F_est_np, epi_dist_mean_estW)

    # Evaluating with OpenCV 5-point
    if if_opencv:
        M, error_Rt_5point, _, E_recover_5point = utils_opencv.recover_camera_opencv(K_np, x1_single_np, x2_single_np,             delta_Rtij_inv, five_point=False, threshold=0.01, show_result=False)
        # print('+++', K_np)
        epi_dist_mean_5point, _, _ = utils_F.epi_distance_np(utils_F.E_to_F_np(E_recover_5point, K_np), x1_single_np, x2_single_np, if_homo=False)
        # print('-0-', utils_F.E_to_F_np(E_recover_5point, K_np))
        # print('-1', utils_F.E_to_F_np(E_recover_5point, K_np), epi_dist_mean_5point)

    return error_Rt_estW, epi_dist_mean_estW, error_Rt_5point, epi_dist_mean_5point, idx, M_estW

def save_to_file(save_file, content, next_line=True):
    with open(save_file, "a") as myfile:
        myfile.write(content)
        if next_line:
            myfile.write('\n')

def get_error_from_sequence(data_loader, scene_data, args, config, errors, file=None, checkGeoDist=False, check_epipolar_contraints=False): 
    # global feature_type
    # feature mode
    from train3 import SPInferLoader

    config_eva = config['evaluations']

    params = config['evaluations']['params']

    feature_mode = config['feature_mode']

    use_est_E = config_eva['use_est_E']


    # load feature type
    if feature_mode == 1:
        feature_type = 'sift'
        print("use model: sift")
    elif feature_mode == 2:
        feature_type = 'superpoint'
        output_dir = './'
        sp_inferrer = SPInferLoader(config, output_dir, args)
        print("config: ", config, file=file)
        print("use model: ", config['pretrained'], file=file)

    print("feature_type: ", feature_type, file=file)


    from IPython.display import clear_output

    delta_i = config_eva['frame_diff']
    # i = np.random.randint(N_frames-delta_i)
    i = config_eva['starting_frame']
    # five_point = False
    
    print("starting frame: ", i, file=file)
    five_point = config_eva['five_point']
    iter_max = config_eva['iter_max']
    print("iter_max: ", iter_max)

    np.set_printoptions(precision=8, suppress=True)
    count = 1


    K= scene_data['calibs']['K'].astype(np.float)
    while i + delta_i < scene_data['N_frames']:
        print('=' * 50)
        print('=' * 20, " start a pair ", '=' * 20)
        print("iter: ", count)
        
        #   clear_output()
        j = i + delta_i

        # Get two frames
        X_rect_i, X_rect_i_vis, delta_Rtij, delta_Rtij_inv, img1_rgb, img2_rgb = get_ij(i, j, data_loader, scene_data, visualize=False)

        #     print('-- delta_Rtij_gt (scene)\n', delta_Rtij)
        #     print('-- delta_Rtij_inv_gt (camera)\n', delta_Rtij_inv)
        E_gt_th, F_gt_th = utils_F._E_F_from_Rt(delta_Rtij[:, :3], delta_Rtij[:, 3:4], K)    
        F_gt = F_gt_th.numpy()
        E_gt_th_inv, F_gt_th_inv = utils_F._E_F_from_Rt(delta_Rtij_inv[:, :3], delta_Rtij_inv[:, 3:4], K)    
        F_gt_inv = F_gt_th_inv.numpy()
        R2s, t2s, M2s = utils_F._get_M2s(E_gt_th)
        #     print('-- E_gt (scene)\n', E_gt_th.numpy())

        img1_rgb_np, img2_rgb_np = np.array(img1_rgb), np.array(img2_rgb)
        img1, img2 = utils_opencv.PIL_to_gray(img1_rgb), utils_opencv.PIL_to_gray(img2_rgb)

        def getFeatures(img1, img1_rgb, img2, img2_rgb, visualize=False, feature_type='sift'):
            # Keypoint detection and matching with SIFT
            if feature_type == 'sift':
                x1_all, kp1, des1 = utils_opencv.SIFT_det(img1, img1_rgb, visualize=visualize)
                x2_all, kp2, des2 = utils_opencv.SIFT_det(img2, img2_rgb, visualize=visualize)
                x1, x2, _, _ = utils_opencv.KNN_match(des1, des2, x1_all, x2_all, kp1, kp2, img1_rgb, img2_rgb, visualize=False)
            elif feature_type == 'superpoint':
            # Keypoint detection and matching with SuperPoint inference model
                sp_pred = sp_inferrer.run_two_imgs(sp_inferrer.img_array_to_input(img1_rgb_np), sp_inferrer.img_array_to_input(img2_rgb_np))
                matches = sp_inferrer.get_matches(sp_pred)
                x1 = matches[0][:, :2]
                x2 = matches[0][:, 2:4]
            return x1, x2
        

        x1, x2 = getFeatures(img1, img1_rgb, img2, img2_rgb, visualize=False, feature_type=feature_type)
        
        # visualize matches
        utils_vis.draw_corr(img1_rgb_np, img2_rgb_np, x1, x2, 1)

        #     print(des1.shape, des2.shape)
        
        # OpenCV results
        ## 5 point
        if five_point:
            print("use five_point")
        else:
            print("use eight_point")
            
        M, error_Rt_5point, mask2, E_return  = utils_opencv.recover_camera_opencv(K, x1, x2, delta_Rtij_inv, five_point=five_point,
            threshold=params['ransac_thresh'])
        K_np = K
        x1_single_np = x1
        x2_single_np = x2
        
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
                                            x1_single_np, x2_single_np, if_homo=False)
        
        errors['epi_dist_mean'].append(epi_dist_mean_5point)
        
        #     M, error_Rt_5point = utils_opencv.recover_camera_0(E_gt_th.numpy(), K, x2, x1, delta_Rtij_inv, five_point=True, threshold=0.001)
        errors['opencv_Rt'].append([error_Rt_5point[0], error_Rt_5point[1]])
        
        #     print("count: ", count, ", error: ", error_Rt_5point, ", epi_dist_mean_5point: ",  epi_dist_mean_5point)
        
        
        def vis_geoDist(img1_rgb, geo_dists, x1, mask=None):
            # geo_dists = np.sqrt(utils_F._sym_epi_dist(F_gt_th, torch.from_numpy(x1[unique_rows_all_idxes]), torch.from_numpy(x2[unique_rows_all_idxes])).numpy())
            plt.hist(geo_dists, 100)
            plt.show()
            geo_dists = np.clip(geo_dists, 0, 10.)
            factor = 1/(geo_dists.max() + 1e-8)
            dot_size = 500
            print("factor: ", factor)
            print("geo_dists: ", geo_dists.shape)
            # print("geo_dists norm: ", geo_dists*factor*dot_size)
            print("x1: ", x1.shape)
            # print("x1: ", x1)
            plt.figure(figsize=(30, 8))
            plt.imshow(img1_rgb)
            if mask == None:
                plt.scatter(x1[:, 0], x1[:, 1], s=geo_dists*factor*dot_size, c='r', edgecolors='w', linewidths=2.)  
                # plt.scatter(x1[:, 0], x1[:, 1], s=1, c='r', edgecolors='w', linewidths=2.)  
            else:
                plt.scatter(x1[mask, 0], x1[mask, 1], s=geo_dists*50, c='r', edgecolors='w', linewidths=2.)
            plt.show()

        ## Check geo dists
        def checkGeoDist(visualize=False):
            x1_normalizedK = utils_misc.de_homo_np((np.linalg.inv(K) @ utils_misc.homo_np(x1).T).T)
            x2_normalizedK = utils_misc.de_homo_np((np.linalg.inv(K) @ utils_misc.homo_np(x2).T).T)
            K_th = torch.from_numpy(K)
            F_gt_normalized = K_th.t()@F_gt_th@K_th

            geo_dists = utils_F._sym_epi_dist(F_gt_normalized, 
                torch.from_numpy(x1_normalizedK), torch.from_numpy(x2_normalizedK)).numpy()
            geo_thres = 1e-4
            mask_in = geo_dists<geo_thres
            mask_out = geo_dists>=geo_thres


            if visualize:
                print("visualize: correspondences in mask 2")
                utils_vis.draw_corr(img1_rgb_np, img2_rgb_np, x1[mask2, :], x2[mask2, :], linewidth=2.)

                print("visualize: scattered geometry distance")
                # print("geo_dists max before: ", geo_dists.max())
                vis_geoDist(img1_rgb_np, geo_dists.copy(), x1, mask=None)
                # print("geo_dists max after: ", geo_dists.max())

                print("visualize: correspondences within geometry threshold")
                utils_vis.draw_corr(img1_rgb_np, img2_rgb_np, x1[mask_in, :], x2[mask_in, :], linewidth=2.)
        #     print(np.sort(geo_dists[mask2 & mask_in] / geo_thres))

                line_widths = geo_dists[mask_out] / geo_thres
                line_widths[line_widths>10] = 10.
                utils_vis.draw_corr_widths(img1_rgb_np, img2_rgb_np, x1[mask_out, :], x2[mask_out, :], linewidth=line_widths, rescale=False, scale=2.)
        #     print(np.sort(geo_dists[mask2 & mask_out] / geo_thres)[::-1])
        #     print(x1[mask2, :] - x2[mask2, :])
        
        if checkGeoDist:
            print("check Geometry Dist!")
            checkGeoDist()
        
        def check_epipolar_contraints():
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
            check_epipolar_contraints()

        print(F_gt)
        print('=' * 20, " end a pair ", '=' * 20)
        print('=' * 50)
        

        # clear_output()
        # for idx in range(i):
        #     def format_02f(x_tuple):
        #         return '(%.2f, %.2f)'%(x_tuple[0], x_tuple[1])
        #     print('- Frame %d and %d'%(idx, idx+1), format_02f(dsac_errors['%d'%idx]), format_02f(opencv_5point_errors['%d'%idx]), format_02f(opencv_8point_errors['%d'%idx]))
        
        i = i+delta_i
        count += 1

        if count > iter_max:
            break

    return errors

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
    print("inliers: ", inliers.shape)
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

    print("="*50, file=f)
    print("="*20, " start ", "="*20, file=f)
    print("="*50, file=f)
    import datetime
    print(datetime.datetime.now(), file=f)

    print("config path: ", filename, file=f)
    print("config: ", config, file=f)

    # get errors
    error_names = ['dsac', 'opencv_Rt', 'epi_dist_mean']
    errors = {error_name:[] for error_name in error_names}
    print(errors)

    errors = get_error_from_sequence(data_loader, scene_data, args, config, errors, file=f)

    # print errors
    ## epipolar distance error
    err = errors['epi_dist_mean']
    thd_1, thd_2 = 0.1, 1
    accs, num_corrs = [], []
    if config['evaluations']['five_point']:
        tag = '5_point'
    else:
        tag = '8_point'
        
    print("="*20, " ", tag, " ", "="*20, file=f)
    for e in err:
        errs = output_epi_dist_mean_est_5p(e, thd_1, thd_2, tag=tag, file=f)
        # print("error: ", e)
        accs.append(np.array(errs))
        num_corrs.append(e.shape[0])
    accs = np.array(accs)
    accs_mean = accs.mean(axis=0)
    num_corrs_mean = np.array(num_corrs).mean()
    print("average number of correspondences: %.2f"%num_corrs_mean, file=f)
    print("percentage of inliers over %d frames, thd=(%.2f, %.2f): (%.2f, %.2f)"%      (accs.shape[0], thd_1, thd_2, accs_mean[0], accs_mean[1]), file=f)

    ## opencv error
    tag = 'opencv_Rt'
    errors_Rt = np.array(errors['opencv_Rt'])
    print("="*20, " ", tag, " ", "="*20, file=f)
    for e in errors_Rt:
        print('error_R = %.4f, error_t = %.4f'%(e[0], e[1]), file=f)

    thds = np.array([0.1, 0.5, 1, 2, 5, 10])
    # print("errors_Rt: ", errors_Rt)
    rt_inliers = output_Rt_est(errors_Rt, thds, tag=tag, file=f, verbose=True)


    ###### end ######
    print("="*20, "   end   ", "="*20, file=f)
    print("="*50, file=f)
    f.close()
    f=None







# In[ ]:




