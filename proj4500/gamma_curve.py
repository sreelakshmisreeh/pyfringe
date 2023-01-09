# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 12:26:54 2023

@author: kl001
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import gspy
import proj4500
os.environ["KMP_DUPLICATE_LIB_OK"] = "True" # needed when openCV and matplotlib used at the same time

def gamma_capture(savedir,
                  gamma_image_index_list, 
                  gamma_pattern_num_list, 
                  proj_exposure_period, 
                  proj_frame_period, 
                  do_insert_black=True,
                  preview_image_index = 34,
                  pprint_proj_status = True):

    cam_triggerType = "hardware"
    result, system, cam_list, num_cameras = gspy.sysScan()
    if result:
        # Run example on each camera
        gspy.clearDir(savedir)
        for i, cam in enumerate(cam_list):    
            print('Running example for camera %d...'%i)
            acquisition_index=0
            result &= proj4500.run_proj_single_camera(cam, 
                                                      savedir, 
                                                      acquisition_index, 
                                                      cam_triggerType, 
                                                      gamma_image_index_list, 
                                                      gamma_pattern_num_list,
                                                      proj_exposure_period, 
                                                      proj_frame_period,
                                                      do_insert_black,
                                                      preview_image_index,
                                                      pprint_proj_status)
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

def plot_gamma_curve(savedir):
    
    cam_width, cam_height = 1920, 1200
    
    camx, camy = int(cam_width/2), int(cam_height/2)
    half_cross_length = 100
    
    img_cam = np.array([cv2.imread(os.path.join(savedir,'capt0_%d.jpg'%i),0) for i in range(0,51)])
    camera_captured = img_cam[:,camy - half_cross_length : camy + half_cross_length, camx - half_cross_length : camx + half_cross_length]
    mean_intensity = np.mean(camera_captured.reshape((camera_captured.shape[0],-1)),axis=1)
    x_axis = np.arange(5,256,5)
    a,b = np.polyfit(x_axis,mean_intensity,1)
    plt.figure(figsize=(16,9))
    plt.scatter(x_axis, mean_intensity, label = 'captured mean per frame')
    plt.plot(x_axis,a*x_axis+b, label = 'linear fit', color = 'r')
    plt.xlabel("Input Intensity",fontsize = 20)
    plt.ylabel("Output Intensity",fontsize = 20)
    plt.title("Projector gamma curve", fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.legend(fontsize = 15)
    plt.savefig(os.path.join(savedir, 'gamma_curve.png'))
    plt.show()
    plt.tight_layout()
    np.save(os.path.join(savedir, 'gamma_curve.npy'),mean_intensity)
    return 
#%%
def main():
    savedir = r'C:\Users\kl001\Documents\pyfringe_test\gamma_images'
    gamma_image_index_list = np.repeat(np.arange(5,22),3).tolist()
    gamma_pattern_num_list = [0,1,2] * len(set(gamma_image_index_list))
    proj_exposure_period = 27084; proj_frame_period = 33334
    gamma_capture(savedir, gamma_image_index_list, gamma_pattern_num_list,proj_exposure_period, proj_frame_period)
    plot_gamma_curve(savedir)

if __name__ == '__main__':
    main()

