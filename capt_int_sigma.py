# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:41:06 2023

@author: kl001
"""

import numpy as np
import image_acquisation as acq
import matplotlib.pyplot as plt
import glob
import os 
import pickle
import cv2
from tqdm import tqdm

"""
Intensity noise lookup table is created by projecting intensity ranging from (5,55,5). 50 images of each intensity is captured.
The experiment is conducted 10 times.
"""

def capture_pixel(base_dir, no_images, proj_exposure_period, proj_frame_period):
    intensity = np.arange(20,245,5)
    img_list = np.repeat(np.arange(1,16),3)
    patt_list = [0,1,2]*len(img_list)
    acq_list = np.arange(0,len(intensity))
    for i,p,a in zip(img_list, patt_list, intensity):
        acq.meanpixel_var(base_dir,
                          image_index=i,
                          pattern_no=p,
                          no_images=no_images,
                          proj_exposure_period=proj_exposure_period,
                          proj_frame_period=proj_frame_period,
                          cam_width=1920,
                          cam_height=1200,
                          half_cross_length=100,
                          acquisition_index=a)
    return

def var_calc(savedir, camx, camy, delta, acq_index ):
    img_arr = []
    var_region = []
    mean_region = []
    for a in tqdm(acq_index):
        path = sorted(glob.glob(os.path.join(savedir,'capt_%03d_*.jpeg'%a)),key=lambda x:int(os.path.basename(x)[-11:-5]))
        img = np.array([cv2.imread(file,0) for file in path])
        capt_cropped = img[:, camy : camy + delta, camx : camx + delta]
        img_arr.append(capt_cropped)
        var_region.append(np.var(capt_cropped, axis=0))
        mean_region.append(np.mean(capt_cropped, axis=0))
    return np.array(img_arr), np.array(var_region), np.array(mean_region)

def var_std_plot(savedir, camx, camy, delta):
    intensity = np.arange(20,245,5)
    int_img_arr, var_img, mean_img = var_calc(savedir, camx, camy, delta, intensity)
    mean_mean_img = np.mean(mean_img, axis=(1,2))
    mean_var = np.mean(var_img, axis=(1,2))
    std_img = np.sqrt(var_img)
    mean_std = np.mean(std_img, axis=(1,2))

    slope, intercept = np.polyfit(mean_mean_img,mean_var,1)
    new_x = np.arange(np.min(mean_mean_img), np.max(mean_mean_img))
    new_y = slope * new_x + intercept


    fig,ax = plt.subplots(1,2)
    fig.suptitle("Region size = %d x %d"%(delta,delta), fontsize=20)
    #ax[0].errorbar(mean_mean_img, mean_var, xerr= std_mean_img, yerr=std_var, fmt='o', markersize=2, ecolor='r', capsize=5)
    ax[0].plot(mean_mean_img,mean_var,"o",label="Data" )
    ax[0].plot(new_x,new_y, label="Fit")
    ax[0].set_xlabel("Intensity", fontsize=20)
    ax[0].set_ylabel(r"Variance $\sigma^2$", fontsize=20)
    ax[0].set_xlim(0,255)
    ax[0].tick_params(axis='both', which='major', labelsize=15)
    ax[0].legend(fontsize=15)
    #ax[1].errorbar(mean_mean_img, mean_std, xerr= std_mean_img, yerr=std_std, fmt='o', markersize=2, ecolor='r', capsize=5)
    ax[1].plot(mean_mean_img, mean_std,"o",label="Data" )
    ax[1].plot(new_x,np.sqrt(new_y), label="Fit")
    ax[1].set_xlabel("Intensity", fontsize=20)
    ax[1].set_ylabel(r"Std $\sigma$", fontsize=20)
    ax[1].tick_params(axis='both', which='major', labelsize=15)
    ax[1].set_xlim(0,255)
    ax[1].legend(fontsize=15)
    plt.show()
    return
  
def main():
    savedir = r'C:\Users\kl001\Documents\pyfringe_test\mean_pixel_std\mean_pixel_std4'
    option = input("Please choose: 1:Capture images\n2:Calculate pixel std")
    if option == '1':
        no_images = int(input("\nEnter number of images:"))
        proj_exposure_period = int(input("\nEnter proj exposure period(mu s):"))
        proj_frame_period = int(input("\nEnter proj frame period(mu s):"))
        capture_pixel(savedir, no_images, proj_exposure_period, proj_frame_period)
    elif option == '2':
        camx = int(input("\nEnter starting x coordinate for cropping:"))
        camy = int(input("\nEnter starting y coordinate for cropping:"))
        delta = int(input("\nEnter delta (no. of pixels in the region) for cropping:"))
        var_std_plot(savedir, camx, camy, delta)   
    return

if __name__ == '__main__':
    main()

