# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 13:00:37 2023

@author: kl001
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(r'C:\Users\kl001\pyfringe\proj4500')
sys.path.append(r'C:\Users\kl001\Documents\pyfringe_test')
import FringeAcquisition as gspy
import proj4500
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

def calibration_capture(savedir, image_index_list, no_calib_images, proj_exposure_period = 27084, proj_frame_period = 33334):
    
    cam_triggerType = "hardware"
    result, system, cam_list, num_cameras = gspy.sysScan()
    if result:
        # Run example on each camera
        gspy.clearDir(savedir)
        cam = cam_list[0]
        for j in range(no_calib_images):
            result &= proj4500.run_proj_single_camera(cam, savedir, j, cam_triggerType, image_index_list, proj_exposure_period, proj_frame_period )
        print('Camera capture complete...')
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
#%%
savedir = r'C:\Users\kl001\Documents\pyfringe_test\multifreq_calib_images'
image_index_list =  np.repeat(np.arange(23,35),3).tolist()
no_calib_images = 10
calibration_capture(savedir, image_index_list, no_calib_images)