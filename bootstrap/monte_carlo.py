# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 16:03:45 2022

@author: kl001
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pandas as pd
from copy import deepcopy
import sys
sys.path.append(r'C:\Users\kl001\pyfringe\functions')
sys.path.append(r'C:\Users\kl001\Documents\pyfringe_test\white_camera_error\varying_B\bootstrap')
import reconstruction_copy as rc
import nstep_fringe as nstep
from plyfile import PlyData, PlyElement

EPSILON = -0.5
TAU = 5.5

def B_cutoff_limit(sigma_path, quantile_limit, N_list, pitch_list):
    sigma = np.load(sigma_path)
    sigma_sq_delta_phi = (np.pi / quantile_limit)**2
    modulation_limit_sq = ((pitch_list[-1] / pitch_list[-2]) + 1) * (2 * sigma**2) / (N_list[-1]* sigma_sq_delta_phi)
    return np.sqrt(modulation_limit_sq)

def image_read(data_path, total_images, total_scans):
    full_images = []
    for j in range (0,total_images):
        path = [glob.glob(os.path.join(data_path,'capt%d_%d.jpg'%(x,j)))  for x in range(0,total_scans)]
        flat_path = [item for sublist in path for item in sublist]
        images = [cv2.imread(file, 0) for file in flat_path]
        full_images.append(images)
    full_images = np.array(full_images)
    images_mean = np.mean(full_images, axis = 1) # full images = sum(n_list), no.of scans, image shape
    images_std = np.std(full_images, axis = 1)
    np.savez(os.path.join(data_path,'images_stat.npz'),images_mean, images_std)
    return full_images, images_mean, images_std

def random_images(images_mean, images_std):
    mean_colm = images_mean.ravel()
    std_colm = images_std.ravel()
    df = pd.DataFrame( np.column_stack((mean_colm, std_colm)), columns = ['pixel_mean','pixel_std'])
    df['RV'] = np.random.normal(loc=df['pixel_mean'], scale=df['pixel_std'])
    rv_array = df['RV'].to_numpy()
    rv_split = np.array(np.split(rv_array, (images_mean.shape[1]*images_mean.shape[2] )))
    random_img = rv_split.reshape(18,images_mean.shape[1],images_mean.shape[2])
    #mask = np.full(random_img[0].shape, False)
    #mask[bound_cord[2]:bound_cord[3], bound_cord[0]:bound_cord[1]] = True
    #mask_img =  np.repeat(mask[np.newaxis,:,:], random_img.shape[0] , axis=0)
    #random_img[~mask_img] = np.nan
    
    return random_img

def random_reconst(width, height, pitch_list, N_list, limit, dist, delta_dist, phase_st, direc, type_unwrap, calib_path, obj_path, random_img, temp, kernel = 1):
    calibration = np.load(os.path.join(calib_path,'mean_calibration_param.npz'))
    c_mtx = np.random.normal(loc = calibration["arr_0"], scale =calibration["arr_1"])
    c_dist = np.random.normal(loc = calibration["arr_2"], scale =calibration["arr_3"])
    p_mtx = np.random.normal(loc = calibration["arr_4"], scale =calibration["arr_5"])
    #p_dist = np.random.normal(loc = calibration["arr_2"], scale =calibration["arr_3"])
    cp_rot_mtx = np.random.normal(loc = calibration["arr_8"], scale =calibration["arr_9"])
    cp_trans_mtx = np.random.normal(loc = calibration["arr_10"], scale =calibration["arr_11"])
    
    object_freq1, mod_freq1, avg_freq1, gamma_freq1, delta_deck_freq1  = nstep.mask_img(random_img[0:N_list[0]], limit)
    object_freq2, mod_freq2, avg_freq2, gamma_freq2, delta_deck_freq2 = nstep.mask_img(random_img[N_list[0]: N_list[0] + N_list[1]], limit)
    object_freq3, mod_freq3, avg_freq3, gamma_freq3, delta_deck_freq3 = nstep.mask_img(random_img[N_list[0] + N_list[1]: N_list[0]+ N_list[1]+ N_list[2]], limit)
    object_freq4, mod_freq4, avg_freq4, gamma_freq4, delta_deck_freq4 = nstep.mask_img(random_img[N_list[0]+ N_list[1]+ N_list[2]: N_list[0]+ N_list[1]+ N_list[2] + N_list[3]], limit)

    #wrapped phase
    phase_freq1 = nstep.phase_cal(object_freq1, N_list[0], delta_deck_freq1 )
    phase_freq2 = nstep.phase_cal(object_freq2, N_list[1], delta_deck_freq2 )
    phase_freq3 = nstep.phase_cal(object_freq3, N_list[2], delta_deck_freq3 )
    phase_freq4 = nstep.phase_cal(object_freq4, N_list[3], delta_deck_freq4 )
    phase_freq1[phase_freq1 < EPSILON] = phase_freq1[phase_freq1 < EPSILON] + 2 * np.pi

    #unwrapped phase
    phase_arr = np.stack([phase_freq1, phase_freq2, phase_freq3, phase_freq4])
    unwrap, k = nstep.multifreq_unwrap(pitch_list, phase_arr, kernel, direc)
    
    
    obj_x, obj_y, obj_z = rc.reconstruction_obj(unwrap, c_mtx, c_dist, p_mtx, cp_rot_mtx, cp_trans_mtx, phase_st, pitch_list[-1])
    roi_mask = np.full(unwrap.shape, False)
    roi_mask[mod_freq4 > limit] = True
    u_copy = deepcopy(unwrap)
    u_copy[~roi_mask] = np.nan
    obj_x[~roi_mask] = np.nan
    obj_y[~roi_mask] = np.nan
    obj_z[~roi_mask] = np.nan
    return obj_x, obj_y, obj_z

def virtual_scan(images_stat_path, virtual_scan_no, proj_width, proj_height, pitch_list, N_list, limit, dist, delta_dist, phase_st, direc, type_unwrap, calib_path, data_path, temp, kernel = 1):
    images_stat = np.load(os.path.join(images_stat_path,'images_stat.npz'))
    obj_x_lst = []
    obj_y_lst = []
    obj_z_lst = []
    
    for v in range(0, virtual_scan_no):
        random_img = random_images(images_stat["arr_0"], images_stat["arr_1"])
        obj_x, obj_y, obj_z = random_reconst(proj_width, proj_height, pitch_list, N_list, limit, dist, delta_dist, phase_st, direc, type_unwrap, calib_path, data_path, random_img, temp, kernel)
        obj_x_lst.append(obj_x)
        obj_y_lst.append(obj_y)
        obj_z_lst.append(obj_z)
    
    x_mean = np.mean(obj_x_lst, axis = 0)
    y_mean = np.mean(obj_y_lst, axis = 0)
    z_mean = np.mean(obj_z_lst, axis = 0)
    
    x_std = np.std(obj_x_lst, axis = 0)
    y_std = np.std(obj_y_lst, axis = 0)
    z_std = np.std(obj_z_lst, axis = 0)
    
    inte_img = cv2.imread(os.path.join(images_stat_path,'white.jpg'))       
    inte_rgb = inte_img[...,::-1].copy()
    inte_rgb = inte_rgb / np.nanmax(inte_rgb)
    rgb_intensity_vect = np.vstack((inte_rgb[:,:,0].ravel(), inte_rgb[:,:,1].ravel(),inte_rgb[:,:,2].ravel())).T
    cordi = np.vstack((x_mean.ravel(), y_mean.ravel(), z_mean.ravel())).T
    cordi_std = np.vstack((x_std.ravel(), y_std.ravel(), z_std.ravel())).T
    xyz_cordi = list(map(tuple, cordi)) # shape is Nx3
    xyz_std = list(map(tuple, cordi_std)) 
    color = list(map(tuple, rgb_intensity_vect))
    
    PlyData(
        [
            PlyElement.describe(np.array(xyz_cordi, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'points'),
            PlyElement.describe(np.array(color, dtype=[('r', 'f4'), ('g', 'f4'), ('b', 'f4')]), 'color'),
            PlyElement.describe(np.array(xyz_std, dtype=[('dx', 'f4'), ('dy', 'f4'), ('dz', 'f4')]), 'std'),
        ]).write(os.path.join(data_path,'obj_wstd.ply'))
    
    return cordi, cordi_std
#%%

type_unwrap =  'multifreq'
pitch_list =[1375, 275, 55, 11] 
N_list = [3, 3, 3, 9]
phase_st = 0
proj_width = 800; proj_height = 1280
direc = 'v'
total_images = sum(N_list)
total_scans = 30
root_dir = r'C:\Users\kl001\Documents\pyfringe_test\white_camera_error\varying_B\bootstrap' 
calib_path = os.path.join(root_dir, '%s_calib_images' %type_unwrap)
data_path = os.path.join(root_dir, 'monte_carlo_scan')
kernel_v = 1
dist = 550
delta_dist =200
direc = 'v' 
temp = False
virtual_scan_no = 500
bound_cord = [600,1300, 50,1100] # bounding box[row1,row2,col1,col2]
sigma_path =  r'C:\Users\kl001\Documents\pyfringe_test\white_camera_error\varying_B\mean_err\mean_pixel_std.npy'
quantile_limit = 5.5
#calculate limit based on R
limit = B_cutoff_limit(sigma_path, quantile_limit, N_list, pitch_list)


# Calculate pixel statistics based on total_scans
full_images, images_mean, images_std = image_read(data_path, total_images, total_scans)
#%%
# Run sigma
#meanpixel_std.main()

# Do virtual scan
cordi, cordi_std  = virtual_scan(data_path, virtual_scan_no, proj_width, proj_height, pitch_list, N_list, limit, dist, delta_dist, phase_st, direc, type_unwrap, calib_path, data_path, temp, kernel_v )

np.save(os.path.join(data_path,'cordi_std.npy'), cordi_std)
np.save(os.path.join(data_path,'cordi.npy'), cordi)
#%%
test_std = np.load(r'C:\Users\kl001\Documents\pyfringe_test\white_camera_error\varying_B\bootstrap\obj_reconstruction\plane\single_cordi_std.npy')
fig,ax = plt.subplots(2)
sns.distplot(cordi_std[:,0], label = '$\sigma x$', kde = False, ax = ax[0])
sns.distplot(cordi_std[:,1], label = '$\sigma y$', kde = False, ax = ax[0])
sns.distplot(cordi_std[:,2], label = '$\sigma z$', kde = False, ax = ax[0])
ax[0].set_xlabel('$\sigma$', fontsize = 15)
ax[0].set_ylabel('Count', fontsize = 15)
ax[0].tick_params(axis = 'both', labelsize = 15)
ax[0].set_xlim(0,np.nanmax(test_std[:,2]))
ax[0].set_title(' Ground truth plane', fontsize = 20)

sns.distplot(test_std[:,0], label = '$\sigma x$', kde = False, ax = ax[1])
sns.distplot(test_std[:,1], label = '$\sigma y$', kde = False, ax = ax[1])
sns.distplot(test_std[:,2], label = '$\sigma z$', kde = False, ax = ax[1])
ax[1].set_xlabel('$\sigma$', fontsize = 15)
ax[1].set_ylabel('Count', fontsize = 15)
ax[1].tick_params(axis = 'both', labelsize = 15)
ax[1].set_xlim(0,np.nanmax(test_std[:,2]))
ax[1].set_title(' Test plane', fontsize = 20)
plt.legend(fontsize = 20)

#%%


delta_sigma_x = cordi_std[:,0] - test_std[:,0]
delta_sigma_y = cordi_std[:,1] - test_std[:,1]
delta_sigma_z = cordi_std[:,2] - test_std[:,2]
#%%
plt.figure()
sns.distplot(delta_sigma_x, label = '$\Delta\sigma_{x}$')
sns.distplot(delta_sigma_y, label = '$\Delta\sigma_{y}$')
sns.distplot(delta_sigma_z, label = '$\Delta\sigma_{z}$')
plt.xlabel('$\Delta\sigma$', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.legend(fontsize = 20)
#plt.title(' Ground truth plane', fontsize = 20)