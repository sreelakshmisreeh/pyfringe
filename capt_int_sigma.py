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

"""
Intensity noise lookup table is created by projecting intensity ranging from (5,55,5). 50 images of each intensity is captured.
The experiment is conducted 10 times.
"""

def capture_pixel(base_dir, no_datasets):
    intensity = np.arange(5,256,5)
    img_list = np.repeat(np.arange(5,22),3)
    patt_list = [0,1,2]*len(img_list)
    acq_list = np.arange(0,len(intensity))
    for k in range(1,no_datasets+1):
        savedir = base_dir + '%d'%k
        for i,p,a in zip(img_list, patt_list, acq_list):
            acq.meanpixel_std(savedir,
                              image_index=i,
                              pattern_no=p,
                              no_images=50,
                              cam_width=1920,
                              cam_height=1200,
                              half_cross_length=100,
                              acquisition_index=a)
    return

def std_int_calc(base_dir, camx, camy, cross_length, no_datasets):
    mean_temp_std_lst = []
    mean_capt_int_lst = []
    for i in range(1,no_datasets+1):
        savedir = base_dir + '%d'%i
        path =  glob.glob(os.path.join(savedir,'capt_*'))
        data_arr = np.array([np.load(p) for p in path])
        data_arr_crop = data_arr[:,:,camy: camy + cross_length, camx: camx + cross_length]
        temp_pixel_std = np.std(data_arr_crop,axis=1)
        mean_temp_std_lst.append(np.mean(temp_pixel_std , axis=(-1,-2)))
        pixel_capt_intensity = np.mean(data_arr_crop, axis=1)
        mean_capt_int_lst.append(np.mean(pixel_capt_intensity, axis=(-1,-2)))
        np.save(os.path.join(savedir, 'temp_pixel_std%d.npy'%i), temp_pixel_std)
        np.save(os.path.join(savedir, 'pixel_capt_intensity%d.npy'%i), pixel_capt_intensity)
    return np.array(mean_temp_std_lst), np.array(mean_capt_int_lst)
  
def main():
    cam_width=1920
    cam_height=1200
    camx = int(cam_width/2)
    camy = int(cam_height/2)
    cross_length=100
    no_datasets = 10
    savedir = r'C:\Users\kl001\Documents\pyfringe_test\mean_pixel_std\mean_pixel_std'
    option = input("Please choose: 1:Capture images\n2:Calculate pixel std")
    if option == '1':
        capture_pixel(savedir, no_datasets)
    elif option == '2':
        mean_temp_std, pixel_mean_capt_int =  std_int_calc(savedir, camx, camy, cross_length, no_datasets)
        np.save(os.path.join(os.path.split(savedir)[0],'full_temp_std.npy'), mean_temp_std)
        np.save(os.path.join(os.path.split(savedir)[0],'full_pixel_int.npy'), pixel_mean_capt_int)
        average_std = np.mean(mean_temp_std, axis=0)
        std_std = np.std(mean_temp_std, axis=0)
        average_int = np.mean(pixel_mean_capt_int, axis=0)
        std_int = np.std(pixel_mean_capt_int, axis=0)

        plt.figure()
        plt.title("$\sigma$ Vs Captured intensity", fontsize=20)
        plt.errorbar(average_int, average_std, xerr=std_int, yerr=std_std, fmt='o', markersize=2, ecolor='r', capsize=2, label=" error bar" )
        plt.xlabel("Captured intensity", fontsize=20)
        plt.ylabel("$\sigma$", fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(fontsize=20)
        plt.ylim(0,3)
        plt.xlim(0,200)
        lut = {}
        average_int = np.round(average_int, 3)
        average_std = np.round(average_std, 3)
        for ai,ast in zip(average_int,average_std):
            lut[ai]=ast
        with open(os.path.join(os.path.split(savedir)[0],"lut.pkl"), "wb") as tt:
            pickle.dump(lut,tt)
        print("LUT saved at {}\lut.pkl".format(os.path.split(savedir)[0]))
    return

if __name__ == '__main__':
    main()

