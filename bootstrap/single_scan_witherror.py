#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 11:27:16 2022

@author: Sreelakshmi
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from time import sleep, perf_counter_ns
from flirpy.camera.lepton import Lepton
import sys
sys.path.append(r'C:\Users\kl001\pyfringe\functions')
sys.path.append(r'C:\Users\kl001\Documents\pyfringe_test')
import FringeAcquisition as fa
import GammaCalibration as gc
import BW_IR_joint_calibration as jcalib
import meanpixel_std
import nstep_fringe as nstep
sys.path.append(r'C:\Users\kl001\Documents\pyfringe_test\white_camera_error\varying_B\bootsrap')
import reconstruction_copy as rc




def capture_reconstruction_images(savedir, type_unwrap, temp):
    """    
    This program is used to capture reconstruction images. The images are saved into savedir.    

    Parameters
    ----------
    savedir : TYPE
        DESCRIPTION.
    type_unwrap : str
        phase unwrapping method
    Returns
    -------
    None.

    """
    # set camera-projector sleep time
    SLEEP_TIME = 0.07
    SLEEP_TIME_INIT = 0.5    
    # Initialize the camera
    result, system, cam_list, num_cameras = fa.sysScan()
    cam = cam_list[0]
    cam.Init()
    fa.cam_configuration(cam)
    cam_IR = Lepton()
    
    # read [unwrap_type]_fringes_distorted.npy. This file contains only vertical fringe patterns based on type of unwrapping method
    fringe_path = os.path.join(savedir, '%s_fringes_distorted.npy' % type_unwrap)
    fringes=np.load(fringe_path)
    white_img = np.full(fringes[0].shape, 255, dtype=np.uint8)    
    
    # live view        
    cam.BeginAcquisition()        
    while True:                
        ret, frame = fa.capture_image(cam)       
        img_show = cv2.resize(frame, None, fx=0.5, fy=0.5)
        cv2.imshow("Grasshopper3 (press q to quit)", img_show)    
        key = cv2.waitKey(1)
        
        if temp:
            raw_therm = cam_IR.grab()            
            img_therm = raw_therm.astype(np.float32)
            # Rescale to 8 bit
            img_therm = 255*(img_therm - img_therm.min())/(img_therm.max()-img_therm.min())        
            # Apply colourmap - try COLORMAP_JET if INFERNO doesn't work.
            # You can also try PLASMA or MAGMA
            frame_therm = cv2.applyColorMap(img_therm.astype(np.uint8), cv2.COLORMAP_INFERNO)
            img_show_therm = cv2.resize(frame_therm, None, fx=5, fy=5)
            cv2.imshow('Lepton (press q to stop)', img_show_therm)
            key = cv2.waitKey(1)
        
        if key == ord("q"):
            break
    cam.EndAcquisition()
    cv2.destroyAllWindows()
    # activate the camera trigger
    fa.activate_trigger(cam)
    # Initialting projector window    
    cv2.startWindowThread()
    cv2.namedWindow('proj',cv2.WINDOW_NORMAL)
    cv2.moveWindow('proj',1920,0)
    cv2.setWindowProperty('proj',cv2.WND_PROP_FULLSCREEN,1)
    # Start pattern projection and capture images
    cam.BeginAcquisition()        
    start = perf_counter_ns()
    for i,img in enumerate(fringes):
        cv2.imshow('proj',img)
        cv2.waitKey(1)
        if i == 0:
            sleep(SLEEP_TIME_INIT)
        else:
            sleep(SLEEP_TIME)
        cam.TriggerSoftware.Execute()            
        ret, image_array = fa.capture_image(cam)        
        save_path = os.path.join(savedir, "capt_%d.jpg"%i)            
        if ret:                    
            cv2.imwrite(save_path, image_array)
            print('Image saved at %s' % save_path)
        else:
            print('Capture fail')
    cv2.imshow('proj', white_img)
    cv2.waitKey(1)
    sleep(SLEEP_TIME)
    cam.TriggerSoftware.Execute()
    ret, image_array = fa.capture_image(cam)
    save_path = os.path.join(savedir, "white.jpg")
    if ret:                    
        cv2.imwrite(save_path, image_array)
        print('Image saved at %s' % save_path)
    else:
        print('Capture fail')  
    if temp:
        raw_IR = cam_IR.grab()
        temperature = raw_IR * 0.01 - 273.15            
    
        BW_IR_calib_dir = r'C:\Users\kl001\Documents\pyfringe_test\thermal_images'
        with open(os.path.join(BW_IR_calib_dir,'grasshopper_lepton.pkl'), 'rb') as file:    
            grasshopper_lepton = pickle.load(file)
        
        u_array, v_array = np.meshgrid(np.arange(1920), np.arange(1200))
        uv_coord_BW = np.array([u_array.flatten(), v_array.flatten()]).T.astype(np.float32)  # shape n x 2
        uv_coord_IR = jcalib.uv_projection_BW2IR(uv_coord_BW, grasshopper_lepton)
        temperature_array_resized = cv2.resize(temperature, None, fx=grasshopper_lepton.scale, fy=grasshopper_lepton.scale)
        temperature_image_trans = jcalib.gen_transformed_IR_img(temperature_array_resized, uv_coord_IR)        
        # Rescale to 8 bit
        img_IR = 255*(temperature_image_trans - temperature_image_trans.min())/(temperature_image_trans.max()-temperature_image_trans.min())        
        # Apply colourmap
        # You can also try PLASMA or MAGMA
        frame_IR = cv2.applyColorMap(img_IR.astype(np.uint8), cv2.COLORMAP_INFERNO)    
        save_path = os.path.join(savedir, 'IR.jpg')
        cv2.imwrite(save_path, frame_IR)
        print('Image saved at %s' % save_path)
        save_path = os.path.join(savedir, 'temperature.npy')
        np.save(save_path, temperature_image_trans)
        print('Temperature data saved at %s' % save_path)
    end = perf_counter_ns()
    t = (end - start)/1e9
    print('time spent: %2.3f s' % t)
    cam.EndAcquisition()
    cv2.destroyAllWindows()
    fa.deactivate_trigger(cam)
    cam.DeInit()
    del cam
    cam_list.Clear()
    system.ReleaseInstance()
    return

#%%
proj_width = 800; proj_height = 1280
cam_width = 1920; cam_height = 1200
inte_rang = [10,245]
#type of unwrapping 
type_unwrap =  'multifreq'
# modulation mask limit
sub_sample_size = 5
no_sample_sets = 100
root_dir = r'C:\Users\kl001\Documents\pyfringe_test\white_camera_error\varying_B\bootsrap' 
source_folder = os.path.join(root_dir, '%s_calib_images' %type_unwrap)
dest_folder = os.path.join(source_folder, 'sub_calib')
dir_to_gamma_param = r'C:\Users\kl001\Documents\pyfringe_test\gamma_calibration'
sigma_path =  r'C:\Users\kl001\Documents\pyfringe_test\white_camera_error\varying_B\mean_err\mean_pixel_std.npy'
pitch_list =[1375, 275, 55, 11] 
N_list = [3, 3, 3, 9]
phase_st = 0
direc = 'v'  
delta_pose = 25 # no of poses in each direction
bobdetect_areamin = 100; bobdetect_convexity = 0.75
dist_betw_circle = 25; #Distance between centers
board_gridrows = 5; board_gridcolumns = 15 # calibration board parameters 
kernel_v = 1; kernel_h=1

temp = False
           
savedir = r'C:\Users\kl001\Documents\pyfringe_test\white_camera_error\varying_B\bootsrap\obj_reconstruction\plane'
if not os.path.exists(savedir):
    os.makedirs(savedir)  
    
quantile_limit = 5.5
limit = rc.B_cutoff_limit(sigma_path, quantile_limit, N_list, pitch_list)

#%%
if not os.path.exists(os.path.join(savedir, '%s_fringes_distorted.npy'% type_unwrap)):        
    ## To generate patterns in vertical directions based on type of unwrapping for reconstruction.
    
    fringe_array, delta_deck_list = nstep.recon_generate(proj_width, proj_height, type_unwrap, N_list, pitch_list, phase_st, inte_rang, direc, savedir)    
    gamma_param = np.load(os.path.join(dir_to_gamma_param,'gamma_param.npz'))
    k = gamma_param['k']
    coeff_backward = gamma_param['coeff_backward']
    I_o_min = gamma_param['I_o_min']
    gamma_curve_normalized = gamma_param['gamma_curve_normalized']        
    fringe_array_distorted = gc.M_distorted(gamma_curve_normalized, coeff_backward, I_o_min, k, fringe_array)
    np.save(os.path.join(savedir, '%s_fringes_distorted.npy'% type_unwrap), fringe_array_distorted)
 #%%
capture_reconstruction_images(savedir, type_unwrap, temp)
#%%# Run sigma
meanpixel_std.main()
#%%
obj_cordi, obj_color, obj_t, cordi_sigma = rc.obj_reconst_wrapper(proj_width, proj_height, 
                                                     pitch_list, N_list, 
                                                     limit,  
                                                     phase_st, 
                                                     direc, 
                                                     type_unwrap, 
                                                     source_folder, 
                                                     savedir,sigma_path, temp, kernel_v)  

#%%
np.save(os.path.join(savedir,'single_cordi_std.npy'), cordi_sigma)
np.save(os.path.join(savedir,'single_cordi.npy'), obj_cordi)
#%%
plt.figure()
sns.distplot(cordi_sigma[:,0], label = '$\sigma x$', kde = False)
sns.distplot(cordi_sigma[:,1], label = '$\sigma y$', kde = False)
sns.distplot(cordi_sigma[:,2], label = '$\sigma z$', kde = False)
plt.xlabel('$\sigma$', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.legend(fontsize = 20)
plt.title('Test plane error', fontsize = 20)
#%%
gt_std = np.load(r'C:\Users\kl001\Documents\pyfringe_test\white_camera_error\varying_B\bootsrap\monte_carlo_scan\cordi_std.npy')
fig,ax = plt.subplots(1,3)
sns.distplot(gt_std[:,0], label = 'Ground truth plane', kde = False, ax = ax[0])
sns.distplot(cordi_sigma[:,0], label = 'Test plane', kde = False, ax = ax[0])
ax[0].set_xlabel('$\sigma$', fontsize = 15)
ax[0].set_ylabel('Count', fontsize = 15)
ax[0].tick_params(axis = 'both', labelsize = 15)
#ax[0].set_xlim(0,np.nanmax(sigma_z))
ax[0].set_title(' $\sigma_{x}$', fontsize = 20)
ax[0].legend(fontsize = 20)

sns.distplot(gt_std[:,1], label = 'Ground truth plane', kde = False, ax = ax[1])
sns.distplot(cordi_sigma[:,1], label = 'Test plane', kde = False, ax = ax[1])
ax[1].set_xlabel('$\sigma$', fontsize = 15)
ax[1].set_ylabel('Count', fontsize = 15)
ax[1].tick_params(axis = 'both', labelsize = 15)
#ax[1].set_xlim(0,np.nanmax(sigma_z))
ax[1].set_title(' $\sigma_{y}$', fontsize = 20)
ax[1].legend(fontsize = 20)

sns.distplot(gt_std[:,2], label = 'Ground truth plane', kde = False, ax = ax[2])
sns.distplot(cordi_sigma[:,2], label = 'Test plane', bins = 50, kde = False, ax = ax[2])
ax[2].set_xlabel('$\sigma$', fontsize = 15)
ax[2].set_ylabel('Count', fontsize = 15)
ax[2].tick_params(axis = 'both', labelsize = 15)
#ax[2].set_xlim(0,np.nanmax(sigma_z))
ax[2].set_title(' $\sigma_{z}$', fontsize = 20)
ax[2].legend(fontsize = 20)
#%%
delta_sigma_x = np.abs( gt_std[:,0] - cordi_sigma[:,0])
delta_sigma_y = np.abs(gt_std[:,1] - cordi_sigma[:,1])
delta_sigma_z = np.abs(gt_std[:,2] - cordi_sigma[:,2])
fig,ax = plt.subplots(1,3)
sns.distplot(delta_sigma_x, kde = False, ax = ax[0])
ax[0].set_xlabel('$\Delta\sigma$', fontsize = 15)
ax[0].set_ylabel('Density', fontsize = 15)
ax[0].tick_params(axis = 'both', labelsize = 15)
ax[0].set_title(' $\Delta\sigma_{x}$', fontsize = 20)


sns.distplot(delta_sigma_y, kde = False, ax = ax[1])
ax[1].set_xlabel('$\Delta\sigma$', fontsize = 15)
ax[1].set_ylabel('Density', fontsize = 15)
ax[1].tick_params(axis = 'both', labelsize = 15)
ax[1].set_title(' $\Delta\sigma_{y}$', fontsize = 20)

sns.distplot(delta_sigma_z, kde = False, ax = ax[2])
ax[2].set_xlabel('$\Delta\sigma$', fontsize = 15)
ax[2].set_ylabel('Density', fontsize = 15)
ax[2].tick_params(axis = 'both', labelsize = 15)
ax[2].set_title(' $\Delta\sigma_{z}$', fontsize = 20)
