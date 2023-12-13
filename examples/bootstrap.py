# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 09:27:09 2023

@author: kl001
"""

import sys
sys.path.append(r'C:\Users\kl001\pyfringe\functions')
sys.path.append(r'C:\Users\kl001\pyfringe')
import nstep_fringe as nstep
import calibration as calib
import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt

def main():
    #Initial parameters for calibration and testing results
    # proj properties
    proj_width = 912 ; proj_height = 1140 # 800 1280 912 1140
    cam_width = 1920; cam_height = 1200
    #type of unwrapping 
    type_unwrap =  'multifreq'
    
    # circle detection parameters
    bobdetect_areamin = 100; bobdetect_convexity = 0.85
    
    # calibration board properties
    dist_betw_circle = 25; #Distance between centers
    board_gridrows = 5; board_gridcolumns = 15 # calibration board parameters
    
    # Define the path from which data is to be read. The calibration parameters will be saved in the same path. 
    # reconstruction point clouds will also be saved in the same path
    
    path = r'C:\Users\kl001\Documents\pyfringe_test\multifreq_calib_images_bk'
    data_type = "npy"
    processing = 'gpu'
    dark_bias_path =  r"C:\Users\kl001\Documents\pyfringe_test\mean_pixel_std\exp_30_fp_42_retake\black_bias\avg_dark.npy"
    model_path = r"C:\Users\kl001\Documents\pyfringe_test\mean_pixel_std\exp_30_fp_42_retake\const_tiff\calib_fringes\variance_model.npy"
    model = cp.load(model_path)
    #multifrequency unwrapping parameters
    if type_unwrap == 'multifreq':
        pitch_list =[1375, 275, 55, 11] 
        N_list = [3, 3, 3, 9]
        kernel_v = 7; kernel_h= 7
        limit = 10
        
    # multiwavelength unwrapping parameters
    if type_unwrap == 'multiwave':
        pitch_list = [139,21,18]
        N_list =[5,5,9]
        kernel_v = 9; kernel_h= 9  
        limit = 10
    
    # phase coding unwrapping parameters
    if type_unwrap == 'phase':
        pitch_list =[20]
        N_list =[9]
        kernel_v = 25; kernel_h=25
        limit = 2
    
    calib_inst = calib.Calibration(proj_width=proj_width, 
                                   proj_height=proj_height,
                                   cam_width=cam_width,
                                   cam_height=cam_height,
                                   mask_limit=limit, 
                                   type_unwrap=type_unwrap, 
                                   N_list=N_list, 
                                   pitch_list=pitch_list, 
                                   board_gridrows=board_gridrows, 
                                   board_gridcolumns=board_gridcolumns, 
                                   dist_betw_circle=dist_betw_circle,
                                   bobdetect_areamin=bobdetect_areamin,
                                   bobdetect_convexity=bobdetect_convexity,
                                   kernel_v=kernel_v,
                                   kernel_h=kernel_h,
                                   path=path,
                                   data_type=data_type,
                                   processing=processing,
                                   dark_bias_path=dark_bias_path)
    
    delta_pose=25   # number of samples in each direction
    pool_size_list =np.arange(20,21,1) # number of poses 
    no_sample_sets = 200 # no of iterations
    cam_mtx_sample, cam_dist_sample, proj_mtx_sample, proj_dist_sample, st_rmat_sample, st_tvec_sample, cam_h_mtx_sample, proj_h_mtx_sample = calib_inst.bootstrap_intrinsics_extrinsics(delta_pose,
                                                                                                                                                                                         pool_size_list, 
                                                                                                                                                                                         no_sample_sets,
                                                                                                                                                                                         model)
    return
def analysis():  
    pool_size_list =np.arange(40,41,1)
    type_unwrap =  'multifreq'
    path = r'C:\Users\kl001\Documents\pyfringe_test\multifreq_calib_images'
    calibration_std = np.load(os.path.join(path, '{}_std_calibration_param.npz'.format(type_unwrap)))
    
    cam_mtx_std = calibration_std["cam_mtx_std"]
    proj_mtx_std = calibration_std["proj_mtx_std"]
    # trend_parm = np.polyfit(pool_size_list, cam_mtx_std[:,0,0], 3)
    # trend = np.poly1d(trend_parm)
    fig, ax = plt.subplots(1)
    fig.suptitle("Camera focal length", fontsize=20)
    ax.plot(pool_size_list,cam_mtx_std[:,0,0], '--o', color='purple', label='$f_x$')
    ax.plot(pool_size_list,cam_mtx_std[:,1,1],'--o', color='blue', label='$f_y$')
    # ax.plot(pool_size_list, trend(pool_size_list))
    ax.axvline(x=40, color='r', linestyle='--')
    ax.set_ylim(0,10)
    ax.set_xlim(13,55)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlabel("No. of poses", fontsize=15)
    ax.set_ylabel("Standard deviation (pixels)", fontsize=15)
    ax.legend(fontsize=15)
    
    fig, ax = plt.subplots(1)
    fig.suptitle("Projector focal length", fontsize=20)
    ax.plot(pool_size_list,proj_mtx_std[:,0,0],'--o', color='purple', label='$f_x$')
    ax.plot(pool_size_list,proj_mtx_std[:,1,1],'--o', color='blue', label='$f_y$')
    ax.axvline(x=40, color='r', linestyle='--')
    ax.set_ylim(0,14)
    ax.set_xlim(13,55)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlabel("No. of poses", fontsize=15)
    ax.set_ylabel("Standard deviation (pixels)", fontsize=15)
    ax.legend(fontsize=20)
    
    fig, ax = plt.subplots(1)
    fig.suptitle("Camera center", fontsize=20)
    ax.plot(pool_size_list,cam_mtx_std[:,0,2], '--go', label='$c_x$')
    ax.plot(pool_size_list,cam_mtx_std[:,1,2],'--o', color ='crimson', label='$c_y$')
    ax.axvline(x=40, color='r', linestyle='--')
    # ax.set_ylim(0,10)
    # ax.set_xlim(15,60)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlabel("No. of poses", fontsize=15)
    ax.set_ylabel("Standard deviation(pixels)", fontsize=15)
    ax.legend(fontsize=15)
    
    fig, ax = plt.subplots(1)
    fig.suptitle("Projector center", fontsize=20)
    ax.plot(pool_size_list,proj_mtx_std[:,0,2],'--go', label='$c_x$')
    ax.plot(pool_size_list,proj_mtx_std[:,1,2],'--o', color ='crimson',label='$c_y$')
    ax.axvline(x=40, color='r', linestyle='--')
    # ax.set_ylim(0,15)
    # ax.set_xlim(15,60)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlabel("No. of poses", fontsize=15)
    ax.set_ylabel("Standard deviation(pixels)", fontsize=15)
    ax.legend(fontsize=20)
    
    st_tvec_std = calibration_std["st_tvec_std"]
    
    fig, ax = plt.subplots(1,2)
    fig.suptitle("Stereo calibration parameter variance", fontsize=20)
    ax[0].plot(pool_size_list,st_tvec_std[:,0], '--bo', label="$T_x$")
    ax[0].plot(pool_size_list,st_tvec_std[:,1],'--go', label="$T_y$")
    ax[1].plot(pool_size_list,st_tvec_std[:,2],'--mo', label="$T_z$")
    ax[0].tick_params(axis='both', which='major', labelsize=15)
    ax[0].set_xlabel("No. of poses", fontsize=15)
    ax[0].set_ylabel("Standard deviation (mm)", fontsize=15)
    ax[0].legend(fontsize=20)
    ax[1].tick_params(axis='both', which='major', labelsize=15)
    ax[1].set_xlabel("No. of poses", fontsize=15)
    ax[1].set_ylabel("Standard deviation (mm)", fontsize=15)
    ax[1].legend(fontsize=20)
    return
if __name__ == '__main__':
    main()