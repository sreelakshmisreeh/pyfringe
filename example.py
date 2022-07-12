#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 11:15:30 2022

@author: Sreelakshmi
"""

import sys
sys.path.append('/Users/Sreelakshmi/Documents/Raspberry/raspi_files/')
import calib
import numpy as np
import os

#Initial parameters for calibration and testing results
# proj properties
width = 800; height = 1280
#type of unwrapping 
type_unwrap =  'multifreq'
# modulation mask limit
limit = 2
# circle detection parameters
bobdetect_areamin = 100; bobdetect_convexity = 0.75
no_pose = 60
dist_betw_circle = 25; #Distance between centers
board_gridrows = 5; board_gridcolumns = 15 # calibration board parameters
#multifrequency unwrapping parameters
pitch_list =[1375, 275, 55, 11] 
N_list = [3, 3, 3, 9]
# For calibration reconstruction
distance = 700 # distance between board and camera projector
delta_distance = 300 #
int_limit = 170 # intensity limit to extract white region of board
resid_outlier_limit = 10
val_label = 176.777
mf_path = '/Users/Sreelakshmi/Documents/Raspberry/reconstruction/July_5_cali_img'

#%% Instantiate calibration class
calib_inst = calib.calibration(width, height, limit, type_unwrap, N_list, pitch_list, board_gridrows, board_gridcolumns, dist_betw_circle, mf_path)
# Calibration of both camera na dprojector. calibration parameters are saved as path + {type_unwrap}__calibration_param.npz
unwrapv_lst, unwraph_lst, white_lst, mod_lst, proj_img_lst, cam_objpts, cam_imgpts, proj_imgpts, euler_angles, cam_mean_error, cam_delta, cam_df1, proj_mean_error, proj_delta, proj_df1   = calib_inst.calib(no_pose, bobdetect_areamin, bobdetect_convexity, kernel_v = 1, kernel_h=1)
#%% If required this function removes poses based on absolute reprojection error
unwrapv_lst, unwraph_lst, white_lst, mod_lst, proj_img_lst,cam_objpts, cam_imgpts, proj_imgpts, euler_angles, cam_mean_error, cam_delta, cam_df1, proj_mean_error, proj_delta, proj_df1   = calib_inst.update_list_calib(proj_df1, unwrapv_lst, unwraph_lst, white_lst, mod_lst,proj_img_lst, bobdetect_areamin, bobdetect_convexity)
#%% Plot for reprojection error analysis
calib_inst.intrinsic_errors_plts( cam_mean_error, cam_delta, cam_df1, 'Camera')
calib_inst.intrinsic_errors_plts( proj_mean_error, proj_delta, proj_df1, 'Projector') 
#%% To reconstruct calibration board and plot error in x,y and z directions
mf_delta_df, mf_abs_delta_df,center_cordi_lst = calib_inst.calib_center_reconstruction(cam_imgpts, unwrapv_lst) 
#%% To reconstruct the white region of the board and fit plane to calculate residue
white_img = white_lst.copy()
white_cord = calib_inst.white_centers(unwrapv_lst, distance, delta_distance, white_img, int_limit, resid_outlier_limit)
#%% To calculate error in calculated distance between reconstructed centers
distance_df = calib_inst.pp_distance_analysis(center_cordi_lst, val_label)
#%%
calibration = np.load(os.path.join(mf_path,'{}_calibration_param.npz'.format(type_unwrap)))
c_mtx = calibration["arr_0"]
c_dist = calibration["arr_1"]
p_mtx = calibration["arr_2"]
cp_rot_mtx = calibration["arr_3"]
cp_trans_mtx = calibration["arr_4"]
