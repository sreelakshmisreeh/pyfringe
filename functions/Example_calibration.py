#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 11:15:30 2022

@author: Sreelakshmi
"""
 # The example.py shows how to call each functions.
  
import sys
sys.path.append(r'/Users/Sreelakshmi/Documents/Github/pyfringe/functions')
import calib
import numpy as np
import os
import matplotlib.pyplot as plt

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

# Define the path from which data is to be read. The calibration parameters will be saved in the same path. 
#White region reconstruction will also be saved in the same path folder white

# Setting the dir on the desktop pointing to the google drive folder
root_dir = r'/Users/Sreelakshmi/Documents/Raspberry/reconstruction/July_8_Calib_img/calib_images/' 
#path = os.path.join(root_dir, '%s_calib_images' %type_unwrap)

#multifrequency unwrapping parameters
if type_unwrap == 'multifreq':
    pitch_list =[1375, 275, 55, 11] 
    N_list = [3, 3, 3, 9]
    kernel_v = 1; kernel_h=1
    
# multiwavelength unwrapping parameters
if type_unwrap == 'multiwave':
    pitch_list = [139,21,18]
    N_list =[5,5,9]
    kernel_v = 40; kernel_h=30   

# phase coding unwrapping parameters
if type_unwrap == 'phase':
    pitch_list =[18]
    N_list =[9]
    kernel_v = 30; kernel_h=35
    
# For calibration reconstruction
distance = 700 # distance between board and camera projector
delta_distance = 300 #
int_limit = 140 # intensity limit to extract white region of board
resid_outlier_limit = 10
val_label = 176.777

# Reprojection criteria
reproj_criteria = 0.45

#%% Instantiate calibration class
calib_inst = calib.calibration(width, height, limit, type_unwrap, N_list, pitch_list, board_gridrows, board_gridcolumns, dist_betw_circle, root_dir)

#%% Calibration of both camera and projector. calibration parameters are saved as path + {type_unwrap}__calibration_param.npz
unwrapv_lst, unwraph_lst, white_lst, mod_lst, proj_img_lst, cam_objpts, cam_imgpts, proj_imgpts, euler_angles, cam_mean_error, cam_delta, cam_df1, proj_mean_error, proj_delta, proj_df1   = calib_inst.calib(no_pose, bobdetect_areamin, bobdetect_convexity, kernel_v = 1, kernel_h=1)
#%% Plot projector images for each pose
calib_inst.image_analysis(proj_img_lst)
#%% plot horizontal unwrapped phase for each pose
calib_inst.image_analysis(unwraph_lst)
#%% plot vertical unwrapped phase for each pose
calib_inst.image_analysis(unwrapv_lst)
#%% plot modulation map for each pose
j = 0 # modulation id. The number of modulations depends on the unwrapping method used.
for i in range(0,len(mod_lst)):
    plt.figure()
    plt.imshow(mod_lst[i][j])
    plt.title('modulation map',fontsize=20)

#%% If required this function removes poses based on absolute reprojection error
unwrapv_lst, unwraph_lst, white_lst, mod_lst, proj_img_lst,cam_objpts, cam_imgpts, proj_imgpts, euler_angles, cam_mean_error, cam_delta, cam_df1, proj_mean_error, proj_delta, proj_df1   = calib_inst.update_list_calib(proj_df1, unwrapv_lst, unwraph_lst, white_lst, mod_lst,proj_img_lst, bobdetect_areamin, bobdetect_convexity, reproj_criteria)

#%% Plot for reprojection error analysis
calib_inst.intrinsic_errors_plts( cam_mean_error, cam_delta, cam_df1, 'Camera')
calib_inst.intrinsic_errors_plts( proj_mean_error, proj_delta, proj_df1, 'Projector') 

#%% To reconstruct calibration board and plot error in x,y and z directions
delta_df, abs_delta_df,center_cordi_lst = calib_inst.calib_center_reconstruction(cam_imgpts, unwrapv_lst) 
#%% To reconstruct the white region of the board and fit plane to calculate residue
white_img = list( white_lst)
# choose mask conditions either modulation or intensity.
#Intensity condition is used to extract white regions alone of the calibration board and 'modulation' is used to reconstruct full calibration board.
mask_cond = 'intensity'
white_cord, white_color = calib_inst.recon_xyz(unwrapv_lst, distance, delta_distance, white_img, mask_cond, modulation = None , int_limit = int_limit , resid_outlier_limit = resid_outlier_limit)
#%% To calculate error in calculated distance between reconstructed centers
distance_df = calib_inst.pp_distance_analysis(center_cordi_lst, val_label)
#%% For checking saved calibration parameters.
calibration = np.load(os.path.join(root_dir,'{}_calibration_param.npz'.format(type_unwrap)))
c_mtx = calibration["arr_0"]
c_dist = calibration["arr_1"]
p_mtx = calibration["arr_2"]
cp_rot_mtx = calibration["arr_3"]
cp_trans_mtx = calibration["arr_4"]
