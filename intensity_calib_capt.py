# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:22:29 2023

@author: kl001
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'Times New Roman',"mathtext.fontset":"cm"})
import sys
sys.path.append(r"C:\Users\kl001\pyfringe")
import image_acquisation as acq
import time
import glob
import os
import cv2
from tqdm import tqdm
#firmware: new_green_intensity_calib
#fringe capture 2-level: #52-1200 #53-120 #54-60 # 55-35 #56-25 # 57-20 # 58-18 #59-16 #60-12 
def inten_fr_calib():
    """
    To capture fringe patterns for intensity calibration
    pitch : no of finges per cycle
    img : image no on the firmware
    no_img : Number of sets of images
    """
    print("Delay")
    time.sleep(20)
    pitch = 18
    img = 58
    no_img = 1000
    fr_dir = r"E:\review_data\color_board_data\set4"
    image_indices = np.tile(np.repeat([52,img],3),no_img).reshape(no_img,6).tolist()
    patern_indices = [np.tile([0,1,2],2).tolist()]*no_img
    
    for j,(i, p) in enumerate(zip(image_indices, patern_indices)):
        result = acq.run_proj_single_camera(savedir=fr_dir,
                                            preview_option='Never',
                                            number_scan=1,
                                            acquisition_index=j,
                                            image_index_list=i,
                                            pattern_num_list=p,
                                            cam_gain=0,
                                            cam_bufferCount=15,
                                            cam_capt_timeout=10,
                                            cam_black_level=0,
                                            cam_ExposureCompensation=0,
                                            proj_exposure_period=30000,#27084,Check option 2 for recomended value
                                            proj_frame_period=40000,#34000,#33334,
                                            do_insert_black=True,
                                            led_select=2,
                                            preview_image_index=51,
                                            focus_image_index=None,
                                            image_section_size=None,
                                            pprint_status=True,
                                            save_npy=False,
                                            save_tiff=True,
                                            clear_dir = False)
    return result

def load_data(data_dir, camx, camy, deltax, deltay, dark_bias):
    
    path = sorted(glob.glob(os.path.join(data_dir,'*.tiff')), key=lambda x:(int(os.path.basename(x)[-15:-10]), int(os.path.basename(x)[-11:-5])))
    images = np.array([cv2.imread(file,0) for file in tqdm(path,desc="image loading")])
    imag_region = images[1200:,camy : camy + deltay, camx : camx + deltax] - dark_bias[camy : camy + deltay, camx : camx + deltax]
    return imag_region

def group_keys_gen(images):
    mean_images = np.round(np.nanmean(images, axis=0)).astype(np.uint8)
    mean_img = mean_images.reshape(mean_images.shape[0], mean_images.shape[-2]*mean_images.shape[-1])
    int_index = np.argsort(mean_img, axis=-1)
    unique_index_lst = []; key_lst = []
    for i in range(mean_img.shape[0]):
        sorted_mean = mean_img[i][int_index[i]]
        unique_map = np.concatenate(([True],sorted_mean[1:] != sorted_mean[:-1]))
        unique_index = np.nonzero(unique_map)[0]
        split_mean = np.split(sorted_mean, unique_index)
        keys_vect = [item[0] for item in split_mean[1:]]
        unique_index_lst.append(unique_index)
        key_lst.append(keys_vect)
    return int_index, unique_index_lst, key_lst

def var_calc(imag_vect, intensity_index, unique_index):
    """
    Two ways of generating variance. 
    1: group all the pixels with same mean intensity and calculate variance together.
    2: group all pixels with the same intensity, calculate variance of each pixel and then average
    """
    var_lst = []; var_lst2 = []
    for i in range(imag_vect.shape[1]):
        img = imag_vect[:,i]
        sort_img_vect = img[:,intensity_index[i]]  
        split_vect = np.split(sort_img_vect, unique_index[i], axis=-1)
        var_map = list(map(np.nanvar, split_vect[1:]))
        new_var = [np.nanvar(a, axis=0) for a in split_vect[1:]]
        quant_997 = list(map(np.nanquantile, new_var, [0.997]*len(new_var)))
        quant_003 = list(map(np.nanquantile, new_var, [0.003]*len(new_var)))
        for n , q3, q997 in zip(new_var, quant_003, quant_997):
            n[(n < q3) | (n > q997)] = np.nan
        var_map2 = list(map(np.nanmean, new_var))
        var_lst.append(var_map)
        var_lst2.append(var_map2)
    return  var_lst, var_lst2

def fringe_full_var(obj_dir, 
                    iterations, 
                    camx, camy, 
                    deltax, deltay, 
                    N_list, 
                    dark_bias, 
                    single_data):
    
    imag_region = load_data(obj_dir, camx, camy, deltax, deltay, dark_bias )
    images_resh = imag_region.reshape(iterations, sum(N_list), deltay, deltax)
    intensity_index_ref, unique_index_ref, key_lst_ref = group_keys_gen(images_resh[:,:N_list[0]])
    vect_imag_ref = images_resh[:,:N_list[0]].reshape(iterations,N_list[0], deltax*deltay)
    var_lst_ref, var_lst_ref_varmean = var_calc(vect_imag_ref, intensity_index_ref, unique_index_ref)
    intensity_index_h, unique_index_h, key_lst_h = group_keys_gen(images_resh[:,N_list[0]:])
    vect_imag_h = images_resh[:,N_list[0]:].reshape(iterations,N_list[1], deltax*deltay)
    var_lst_h, var_lst_h_varmean = var_calc(vect_imag_h, intensity_index_h, unique_index_h)
    return key_lst_ref, var_lst_ref, key_lst_h, var_lst_h, var_lst_h_varmean

def plot_model(full_key_h_list, full_var_h_lst_varmean):
    fig1, ax1 = plt.subplots()
    x_values = np.linspace(5,250, num=10000)
    slopes=[]; intercepts = []
    for i in range(len(full_var_h_lst_varmean)):
        index1 = np.where(np.array(full_key_h_list[i])<246)
        new_key1 = np.array(full_key_h_list[i])[index1]
        new_var1 = np.array(full_var_h_lst_varmean[i])[index1]
        slope1, intercept1 = np.polyfit(new_key1, new_var1,1)
        new_yvalues1 = slope1 * x_values + intercept1
        ax1.scatter(new_key1, new_var1, label="Pattern%d: $\sigma^2_{I_n} = %.3f I_n + %.3f$"%((i+1),slope1,intercept1))
        ax1.plot(x_values, new_yvalues1)
        ax1.tick_params(axis='both', which='major', labelsize=30)
        ax1.legend(loc="upper left", fontsize=28, labelspacing=0.3)
        ax1.set_xlabel("Intensity ($I_n$)", fontsize=30)
        ax1.set_ylabel("Variance ($\sigma^2_{I_n}$)", fontsize=30)
        slopes.append(slope1)
        intercepts.append(intercept1)
    
    return np.array(slopes), np.array(intercepts)
#%%
def main():
    result = inten_fr_calib()
    dark_bias = np.load(r"C:\Users\kl001\Documents\pyfringe_test\mean_pixel_std\exp_30_fp_42_retake\black_bias\avg_dark.npy")
    fringe_dir =  r"E:\review_data\color_board_data\set3"
    iterations = 800
    pitch_list =[1200, 18]
    N_list = [3] *len(pitch_list)
    camx = 450
    camy = 450
    deltax = 200
    deltay = 200
    full_key_r_list, full_var_r_lst, full_key_h_list, full_var_h_lst, full_var_h_lst_varmean = fringe_full_var(fringe_dir, 
                                                                                                               iterations, 
                                                                                                               camx, camy, 
                                                                                                               deltax, deltay, 
                                                                                                               N_list,  
                                                                                                               dark_bias, 
                                                                                                               False)
    slopes, intercepts = plot_model(full_key_h_list, full_var_h_lst_varmean)
    model = np.array([slopes[0],intercepts[0]])
    np.save(os.path.join(fringe_dir,"variance_model.npy"),model)
    np.save(os.path.join(fringe_dir,"slopes.npy"),slopes)
    np.save(os.path.join(fringe_dir,"intercepts.npy"),intercepts)
    return result
if __name__ == '__main__':
    if main():
        sys.exit(0)
    else:
        sys.exit(1)

#%%

dark_bias = np.load(r"C:\Users\kl001\Documents\pyfringe_test\mean_pixel_std\exp_30_fp_42_retake\black_bias\avg_dark.npy")
fringe_dir =  r"E:\review_data\color_board_data\set4"
iterations = 800
pitch_list =[1200, 18]
N_list = [3] *len(pitch_list)
camx_all = 400,800,1220
camy = 500
deltax = 50
deltay = 50
all_key_h_list=[];all_var_h_lst=[];
for camx in camx_all:
    full_key_r_list, full_var_r_lst, full_key_h_list, full_var_h_lst, full_var_h_lst_varmean = fringe_full_var(fringe_dir, 
                                                                                                               iterations, 
                                                                                                               camx, camy, 
                                                                                                               deltax, deltay, 
                                                                                                               N_list,  
                                                                                                               dark_bias, 
                                                                                                               False)
    all_key_h_list.append(full_key_h_list)
    all_var_h_lst.append(full_var_h_lst_varmean)

model = np.load(r"C:\Users\kl001\Documents\pyfringe_test\mean_pixel_std\exp_30_fp_42_retake\const_tiff\calib_fringes\variance_model.npy")
x_values = np.linspace(5,250, num=10000)
new_yvalues1 = model[0] * x_values + model[1]
#%%
fig1, ax1 = plt.subplots()

ax1.scatter(all_key_h_list[1][-1][:-4], all_var_h_lst[1][-1][:-4],
            color="orange", label="Yellow region", alpha=0.4, marker="v", edgecolor="k")
ax1.scatter(all_key_h_list[-1][-1][:-8], all_var_h_lst[-1][-1][:-8],
            color="g", label="Green region", alpha=0.5, marker="s")
ax1.scatter(all_key_h_list[0][0][:-1], all_var_h_lst[0][0][:-1],color="r", label="Red region", alpha=0.5)
ax1.plot(x_values, new_yvalues1, color="k",label="Model")
ax1.tick_params(axis='both', which='major', labelsize=50)
ax1.legend(loc="upper left", fontsize=28, labelspacing=0.3)
ax1.set_xlabel("Intensity ($\mu_{I_n}$)", fontsize=50)
ax1.set_ylabel("Variance ($\sigma^2_{I_n}$)", fontsize=50)
ax1.text(0.3,0.9,"$\sigma^2_{I_n}$=0.007$\mu_{I_n}$+0.0172", transform=ax1.transAxes, fontsize=50)
