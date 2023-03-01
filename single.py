#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 10:23:26 2023

@author: Sreelakshmi

Single scan sigma x,y,z compared with ground truth (monte carlo) sigma x,y,z for plane and spherical surface.
"""

import numpy as np
import cv2
import os
import sys
sys.path.append(r'/Users/Sreelakshmi/Documents/Github/pyfringe')
import nstep_fringe as nstep
import reconstruction as rc
import matplotlib.pyplot as plt
import seaborn as sns

proj_width = 912  
proj_height = 1140 
cam_width = 1920 
cam_height = 1200
type_unwrap = 'multifreq'
surface = 'sphere'
sigma_path = r'/Volumes/My Passport/12Feb2023/reconst_test/mean_std_pixel.npy'
obj_path = r'/Users/Sreelakshmi/Documents/Raspberry/codes/22feb2023/single_scan/%s'%surface
calib_path = r'/Volumes/My Passport/12Feb2023/reconst_test'
pitch_list = [1000, 110, 16]
N_list = [3, 3, 9]
quantile_limit = 4.5
limit = nstep.B_cutoff_limit(sigma_path, quantile_limit, N_list, pitch_list)
#%%
reconst_inst = rc.Reconstruction(proj_width=proj_width,
                              proj_height=proj_height,
                              cam_width=cam_width,
                              cam_height=cam_height,
                              type_unwrap=type_unwrap,
                              limit=limit,
                              N_list=N_list,
                              pitch_list=pitch_list,
                              fringe_direc='v',
                              kernel=7,
                              data_type='jpeg',
                              processing='cpu',
                              calib_path=calib_path,
                              sigma_path=sigma_path,
                              object_path=obj_path,
                              temp=False,
                              save_ply=True,
                              probability=True)

obj_cordi, obj_color, cordi_sigma, mask, modulation_vector = reconst_inst.obj_reconst_wrapper()

#%%
np.save(os.path.join(obj_path,'single_cordi_std.npy'), cordi_sigma)
np.save(os.path.join(obj_path,'single_cordi.npy'), obj_cordi)
#%%
gt_std_img = np.load(r'/Volumes/My Passport/12Feb2023/Monte_carlo/%s/monte_std_cords_%s.npy'%(surface,surface))
gt_std_x = gt_std_img[0][mask]
gt_std_y = gt_std_img[1][mask]
gt_std_z = gt_std_img[2][mask]
fig,ax = plt.subplots(1,3)
sns.distplot(gt_std_x, label = 'Ground truth plane', hist = False, ax = ax[0])
sns.distplot(cordi_sigma[:,0], label = 'Test plane', hist = False, ax = ax[0])
ax[0].set_xlabel('$\sigma$ (mm)', fontsize = 15)
ax[0].set_ylabel('Density', fontsize = 15)
ax[0].tick_params(axis = 'both', labelsize = 15)
#ax[0].set_xlim(0,np.nanmax(sigma_z))
ax[0].set_title(' $\sigma_{x}$', fontsize = 20)
ax[0].legend(fontsize = 20)

sns.distplot(gt_std_y, label = 'Ground truth plane', hist = False, ax = ax[1])
sns.distplot(cordi_sigma[:,1], label = 'Test plane', hist = False, ax = ax[1])
ax[1].set_xlabel('$\sigma$ (mm)', fontsize = 15)
ax[1].set_ylabel('Density', fontsize = 15)
ax[1].tick_params(axis = 'both', labelsize = 15)
#ax[1].set_xlim(0,np.nanmax(sigma_z))
ax[1].set_title(' $\sigma_{y}$', fontsize = 20)
ax[1].legend(fontsize = 20)

sns.distplot(gt_std_z, label = 'Ground truth plane', hist = False, ax = ax[2])
sns.distplot(cordi_sigma[:,2], label = 'Test plane',  hist = False, ax = ax[2])
ax[2].set_xlabel('$\sigma$ (mm)', fontsize = 15)
ax[2].set_ylabel('Density', fontsize = 15)
ax[2].tick_params(axis = 'both', labelsize = 15)
#ax[2].set_xlim(0,np.nanmax(sigma_z))
ax[2].set_title(' $\sigma_{z}$', fontsize = 20)
ax[2].legend(fontsize = 20)
#%%
delta_sigma_x =  gt_std_x - cordi_sigma[:,0]
delta_sigma_y =gt_std_y - cordi_sigma[:,1]
delta_sigma_z = gt_std_z - cordi_sigma[:,2]
fig,ax = plt.subplots(1,3)
sns.distplot(delta_sigma_x, hist = False, ax = ax[0])
ax[0].set_xlabel('$\Delta\sigma$', fontsize = 15)
ax[0].set_ylabel('Density', fontsize = 15)
ax[0].tick_params(axis = 'both', labelsize = 15)
ax[0].set_title(' $\Delta\sigma_{x}$', fontsize = 20)
#ax[0].set_xlim(-4,4)


sns.distplot(delta_sigma_y, hist = False, ax = ax[1])
ax[1].set_xlabel('$\Delta\sigma$', fontsize = 15)
ax[1].set_ylabel('Density', fontsize = 15)
ax[1].tick_params(axis = 'both', labelsize = 15)
ax[1].set_title(' $\Delta\sigma_{y}$', fontsize = 20)
#ax[1].set_xlim(-4,4)

sns.distplot(delta_sigma_z, hist = False, ax = ax[2])
ax[2].set_xlabel('$\Delta\sigma$', fontsize = 15)
ax[2].set_ylabel('Density', fontsize = 15)
ax[2].tick_params(axis = 'both', labelsize = 15)
ax[2].set_title(' $\Delta\sigma_{z}$', fontsize = 20)
#ax[2].set_xlim(-10,10)