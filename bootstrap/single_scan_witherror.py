#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 11:27:16 2022

@author: Sreelakshmi
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from time import perf_counter_ns
import sys
sys.path.append(r'C:\Users\kl001\pyfringe')
import reconstruction as rc
import nstep_fringe as nstep


proj_width = 800; proj_height = 1280
cam_width = 1920; cam_height = 1200
inte_rang = [10,245]
#type of unwrapping 
type_unwrap =  'multifreq'
data_type = 'jpeg'
processing= 'gpu'
# modulation mask limit
sub_sample_size = 5
no_sample_sets = 100
root_dir = r'C:\Users\kl001\Documents\pyfringe_test\white_camera_error\varying_B\bootsrap' 
calib_path = r'C:\Users\kl001\Documents\pyfringe_test\white_camera_error\varying_B\bootstrap\New folder'
dir_to_gamma_param = r'C:\Users\kl001\Documents\pyfringe_test\gamma_calibration'
sigma_path =  r'C:\Users\kl001\Documents\pyfringe_test\white_camera_error\varying_B\mean_err\mean_pixel_std.npy'
pitch_list =[900, 100,16] 
N_list = [3, 3, 9]
phase_st = 0
direc = 'v'  
delta_pose = 25 # no of poses in each direction
bobdetect_areamin = 100; bobdetect_convexity = 0.75
dist_betw_circle = 25; #Distance between centers
board_gridrows = 5; board_gridcolumns = 15 # calibration board parameters 
kernel_v = 1; kernel_h=1

temp = False
           
savedir = r'C:\Users\kl001\Documents\pyfringe_test\white_camera_error\varying_B\bootstrap\obj_reconstruction\plane'
if not os.path.exists(savedir):
    os.makedirs(savedir)  
    
quantile_limit = 5.5
limit = nstep.B_cutoff_limit(sigma_path, quantile_limit, N_list, pitch_list)


#%%
start = perf_counter_ns()
obj_cordi, obj_color, obj_t, cordi_sigma, mod_vect = rc.obj_reconst_wrapper_3level(width=proj_width, 
                                                                                   height=proj_height, 
                                                                                   cam_width=cam_width,
                                                                                   cam_height=cam_height,
                                                                                   pitch_list=pitch_list, 
                                                                                   N_list=N_list,
                                                                                   limit=limit,
                                                                                   phase_st=phase_st,
                                                                                   direc=direc,
                                                                                   type_unwrap=type_unwrap, 
                                                                                   calib_path=calib_path, 
                                                                                   obj_path=savedir, 
                                                                                   sigma_path=sigma_path, 
                                                                                   temp=temp, 
                                                                                   data_type=data_type,
                                                                                   processing=processing,
                                                                                   kernel=1)  
end = perf_counter_ns()
print("Execution time %2.6f"%((end-start)/1e9))
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
#%%
