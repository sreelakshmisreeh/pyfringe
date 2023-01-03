# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 12:26:54 2023

@author: kl001
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(r'C:\Users\kl001\Documents\pyfringe_test\proj4500')
sys.path.append(r'C:\Users\kl001\Documents\pyfringe_test')
import FringeAcquisition as fa
import proj4500
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

def gamma_capture(savedir):

    image_index_list = np.repeat(np.arange(5,22),3).tolist()
    proj_exposure_period = 27084
    proj_frame_period = 33334
    pat_number = [0,1,2]
    cam_triggerType = "hardware"
    result, system, cam_list, num_cameras = fa.sysScan()
    if result:
        # Run example on each camera
        fa.clearDir(savedir)
        for i, cam in enumerate(cam_list):    
            print('Running example for camera %d...'%i)
            acquisition_index=0
            result &= proj4500.run_proj_single_camera(cam, savedir, acquisition_index, cam_triggerType, image_index_list, pat_number, proj_exposure_period, proj_frame_period )
            print('Camera %d example complete...'%i)
    
        # Release reference to camera
        # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
        # cleaned up when going out of scope.
        # The usage of del is preferred to assigning the variable to None.
        if cam_list:    
            del cam
        else:
            print('Camera list is empty! No camera is detected, please check camera connection.')    
    else:
        pass
    # Clear camera list before releasing system
    cam_list.Clear()
    # Release system instance
    system.ReleaseInstance() 
    return result

def gamma_calculation(savedir):
    
    cam_width, cam_height = 1920, 1200
    
    camx, camy = int(cam_width/2), int(cam_height/2)
    half_cross_length = 100
    
    img_cam = np.array([cv2.imread(r'C:\Users\kl001\Documents\grasshopper3_python\images\Acquisition-00-%02d.jpg'%i,0) for i in range(0,51)])
    camera_captured = img_cam[:,camy - half_cross_length : camy + half_cross_length, camx - half_cross_length : camx + half_cross_length]
    camera_captured_max = np.max(camera_captured, axis=0)
    camera_captured_min = np.min(camera_captured, axis=0)
    camera_captured_normalized = 255 * (camera_captured - camera_captured_min * np.ones((camera_captured.shape[0],1,1))) / ((camera_captured_max - camera_captured_min)*np.ones((camera_captured.shape[0],1,1)))
    max_raw_per_frame = np.max(camera_captured.reshape((camera_captured.shape[0],-1)),axis=1)
    mean_normalized_per_frame = np.mean(camera_captured_normalized.reshape((camera_captured_normalized.shape[0],-1)), axis=1)
    x_axis = np.arange(5,256,5)
    plt.figure()
    plt.scatter(x_axis, max_raw_per_frame, label = 'captured max per frame')
    plt.scatter(x_axis, mean_normalized_per_frame, label='normalized')
    plt.xlabel("Input Intensity",fontsize = 20)
    plt.ylabel("Output Intensity",fontsize = 20)
    plt.legend()
    return img_cam
#%%
savedir = r'C:\Users\kl001\Documents\grasshopper3_python\images'
gamma_capture(savedir)
img_cam = gamma_calculation(savedir)

#%%
    
