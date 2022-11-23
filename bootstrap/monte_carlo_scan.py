# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 10:29:13 2022

@author: kl001
"""
import cv2
import numpy as np
import os
import glob
from time import sleep, perf_counter_ns
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(r'C:\Users\kl001\pyfringe\functions')
sys.path.append(r'C:\Users\kl001\Documents\pyfringe_test')
import nstep_fringe as nstep
import FringeAcquisition as fa
import GammaCalibration as gc
sys.path.append(r'C:\Users\kl001\Documents\pyfringe_test\white_camera_error\varying_B\bootsrap')



def capture_reconstruction_images(savedir, type_unwrap, no_scans):
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
    for x in range (0, no_scans):
        for i,img in enumerate(fringes):
            cv2.imshow('proj',img)
            cv2.waitKey(1)
            if i == 0:
                sleep(SLEEP_TIME_INIT)
            else:
                sleep(SLEEP_TIME)
            cam.TriggerSoftware.Execute()            
            ret, image_array = fa.capture_image(cam)        
            save_path = os.path.join(savedir, "capt%d_%d.jpg"%(x,i))            
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

def image_read(data_path, total_images, total_scans):
    full_images = []
    for j in range (0,total_images):
        path = [glob.glob(os.path.join(data_path,'capt%d_%d.jpg'%(x,j)))  for x in range(0,total_scans)]
        flat_path = [item for sublist in path for item in sublist]
        images = [cv2.imread(file, 0) for file in flat_path]
        full_images.append(images)
    full_images = np.array(full_images)
    images_std = np.std(full_images, axis = 1)
    return full_images, images_std

def opt_scan(savedir, type_unwrap, scan_no):
    sig_lst = []
    for no_scan in scan_no:
        full_images, images_std = image_read(savedir, total_images, no_scan)
        sig_lst.append(images_std)
    return np.array(sig_lst)
#%%
inte_rang = [10,245]
#type of unwrapping 
type_unwrap =  'multifreq'
pitch_list =[1375, 275, 55, 11] 
N_list = [3, 3, 3, 9]
phase_st = 0
proj_width = 800; proj_height = 1280
cam_width = 1920; cam_height = 1200
direc = 'v'
savedir = r'C:\Users\kl001\Documents\pyfringe_test\white_camera_error\varying_B\bootsrap\monte_carlo_scan'
dir_to_gamma_param = r'C:\Users\kl001\Documents\pyfringe_test\gamma_calibration'
scan_no = [20,25,30,35,40,45,50]
no_scans = 20
total_images = sum(N_list)
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
#%%test for optimal scans
capture_reconstruction_images(savedir, type_unwrap, scan_no[-1])
#%%
sig_lst = opt_scan(savedir, type_unwrap, scan_no)
#%%
delta_sig = np.diff(sig_lst, axis =0)

plt.figure()
for i in range (0,delta_sig.shape[0] ):
    sns.distplot(delta_sig[i,-1,:,:], hist = False)
    

plt.xlabel('$\Delta\sigma$', fontsize = 15)
plt.ylabel('Density', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.legend(fontsize = 20)
plt.title('Image 18', fontsize = 20)
#%%
plt.figure()
for i in range (0,sig_lst.shape[0] ):
    sns.distplot(sig_lst[i,-1,:,:], label = '$\sigma x$', hist = False)
    

plt.xlabel('$\sigma$', fontsize = 15)
plt.ylabel('Density', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.legend(fontsize = 20)
plt.title('Image 18', fontsize = 20)
#%%

sigma_avg = np.mean(sig_lst[:,:,:,:], axis =(2,3))
plt.figure()
for i in range (0,sigma_avg.shape[1] ):
    
    sns.scatterplot(scan_no, sigma_avg[:,i], label = 'Image %d'%i)
plt.ylabel('avg $\sigma$', fontsize = 15)
plt.xlabel('No. of scans', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.legend(fontsize = 15)
plt.title('Optimal scan', fontsize = 20)


#%% 




