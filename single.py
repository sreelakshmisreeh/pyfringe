#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 10:23:26 2023

@author: Sreelakshmi

Single scan sigma x,y,z compared with ground truth (monte carlo) sigma x,y,z for plane and spherical surface.
"""

import numpy as np
import os
import sys
sys.path.append(r'C:\Users\kl001\pyfringe')
import reconstruction as rc
import nstep_fringe as nstep
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'Times New Roman',"mathtext.fontset":"cm"})
import seaborn as sns

proj_width = 912  
proj_height = 1140 
cam_width = 1920 
cam_height = 1200
limit = 80
type_unwrap = 'multifreq'
dark_bias_path = r"C:\Users\kl001\Documents\pyfringe_test\mean_pixel_std\exp_30_fp_42_retake\black_bias\avg_dark.npy"
calib_path = r'C:\Users\kl001\Documents\pyfringe_test\multifreq_calib_images'
obj_dir =  r"E:\green_small_sphere_single"  
model_path = r"C:\Users\kl001\Documents\pyfringe_test\mean_pixel_std\exp_30_fp_42_retake\lut_models.pkl"

pitch_list = [1200, 18]
N_list = [3, 3]

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
                                  data_type='tiff',
                                  processing='cpu',
                                  dark_bias_path=dark_bias_path,
                                  calib_path=calib_path,
                                  object_path=obj_dir,
                                  model_path=model_path,
                                  temp=False,
                                  save_ply=True,
                                  probability=True)

obj_cordi, obj_color, cordi_sigma, mask = reconst_inst.obj_reconst_wrapper(prob_up=True)

#%%
cord_image = np.array([nstep.recover_image(obj_cordi[:,i],mask,cam_height,cam_width) for i in range(3)])
cord_sigma_image = np.array([nstep.recover_image(cordi_sigma[:,i],mask,cam_height,cam_width) for i in range(3)])
np.save(os.path.join(obj_dir,'single_cord_image.npy'), cord_image)
np.save(os.path.join(obj_dir,'singlecord_sigma_image.npy'), cord_sigma_image)
np.save(os.path.join(obj_dir,'single_cordi_std.npy'), cordi_sigma)
np.save(os.path.join(obj_dir,'single_cordi.npy'), obj_cordi)
np.save(os.path.join(obj_dir,'single_mask.npy'), mask)
#%%
obj_dir2 = r"E:\green_freq_small_sphere\pitch_18"  
multi_img = np.load(os.path.join(obj_dir2,"cord_mean.npy"))
multi_var = np.load(os.path.join(obj_dir2,"cord_std.npy"))
multimask = np.load(os.path.join(obj_dir2,"mask.npy"))
#%%
single_vect = np.array([cord_image[i][multimask] for i in range(cord_image.shape[0])])
single_sigma_vect = np.array([cord_sigma_image[i][multimask] for i in range(cord_image.shape[0])])
multi_vect = np.array([multi_img[i][multimask] for i in range(cord_image.shape[0])])
multi_sigma_vect = np.array([multi_var[i][multimask] for i in range(cord_image.shape[0])])
#%%
fig, ax = plt.subplots(3,2)
for i in range(3):
    im = ax[i,0].imshow(np.sqrt(multi_var[i]),cmap="jet", vmin=np.sqrt(np.nanmin(multi_var[i])), vmax=np.nanmax(cord_image[i]))
    ax[i,0].tick_params(axis="both", which='major', labelsize=12)
    cbar0=fig.colorbar(im, ax=ax[i,0], extend="max")
    cbar0.ax.tick_params(labelsize=12)
    #cbar0.mappable.set_clim(0, 2)
    cbar0.set_label("radians", rotation=270, fontsize=15, labelpad=12)
    
    im1 = ax[i,1].imshow(cord_sigma_image[i],cmap="inferno",vmin=np.nanmin(multi_var[i]), vmax=np.nanmax(cord_image[i]))
    ax[i,1].tick_params(axis="both", which='major', labelsize=12)
    cbar1=fig.colorbar(im, ax=ax[i,1], extend="max")
    cbar1.ax.tick_params(labelsize=12)
    #cbar1.mappable.set_clim(0, 2)
    cbar1.set_label("radians", rotation=270, fontsize=15, labelpad=12)

#%%
title = [r"$\sigma_x$", r"$\sigma_y$", r"$\sigma_z$"]
fig, ax = plt.subplots(1,3)
for i in range(3):
    sns.histplot(cord_sigma_image[i].ravel(), ax=ax[i], label="single scan", color="red")
    sns.histplot(multi_var[i].ravel(), ax=ax[i], label="multi scan", color="blue")
    ax[i].set_xlabel("%s"%title[i], fontsize=20)
    ax[i].set_ylabel("Counts", fontsize=20)
    ax[i].set_title("Histogram %s"%title[i], fontsize=20)
    ax[i].tick_params(axis="both", which='major', labelsize=15)
    ax[i].legend(loc="upper right",fontsize=20)
#%%
title = [r"$\sigma_x$", r"$\sigma_y$", r"$\sigma_z$"]
camx = 750 #w410;900
camy = 420 #w300;400
deltax =390 
deltay = 380
diff_cord = cord_sigma_image-multi_var
relat_err = diff_cord/multi_var
fig, ax = plt.subplots(3,4)
for i in range(3):
    cmap0 = plt.cm.get_cmap("inferno").copy()
    cmap0.set_over('red')
    im = ax[i,0].imshow(multi_var[i,camy:(camy+deltay),camx:(camx+deltax)],cmap=cmap0,
                        vmin=np.nanquantile(multi_var[i],0.03), 
                        vmax=np.nanquantile(multi_var[i],0.97))
    ax[i,0].tick_params(axis="both", which='major', labelsize=12)
    ax[i,0].set_title("Empirical %s \nfrom multiple scans"%title[i], fontsize=20)
    cbar0=fig.colorbar(im, ax=ax[i,1],extend="max" )
    cbar0.ax.tick_params(labelsize=18)
    cbar0.ax.yaxis.get_offset_text().set(size=12)
    cbar0.set_label("mm", rotation=270, fontsize=15, labelpad=12)
    
    cmap1 = plt.cm.get_cmap("inferno").copy()
    cmap1.set_over('red')
    im1 = ax[i,1].imshow(cord_sigma_image[i, camy:(camy+deltay),camx:(camx+deltax)],cmap=cmap1,
                         vmin=np.nanquantile(multi_var[i],0.03), 
                         vmax=np.nanquantile(multi_var[i],0.97))
    ax[i,1].tick_params(axis="both", which='major', labelsize=12)
    ax[i,1].set_title("Predicted %s \nfrom a single scan"%title[i], fontsize=20)
    cbar1=fig.colorbar(im1, ax=ax[i,0],extend="max")
    cbar1.ax.tick_params(labelsize=18)
    cbar1.set_label("mm", rotation=270, fontsize=15, labelpad=12)
    
    cmap2 = plt.cm.get_cmap("BrBG").copy()
    cmap2.set_over('red')
    im2 = ax[i,2].imshow(diff_cord[i, camy:(camy+deltay),camx:(camx+deltax)],cmap=cmap2,
                         vmin=np.nanquantile(diff_cord[i],0.03), 
                         vmax=np.nanquantile(diff_cord[i],0.97))
    ax[i,2].tick_params(axis="both", which='major', labelsize=12)
    ax[i,2].set_title("Difference:\n (predicted-empirical)%s"%title[i], fontsize=20)
    cbar2=fig.colorbar(im2, ax=ax[i,2],extend="max")
    cbar2.ax.tick_params(labelsize=18)
    cbar2.set_label("mm", rotation=270, fontsize=15, labelpad=12)
    
    cmap3 = plt.cm.get_cmap("BrBG").copy()
    cmap3.set_over('red')
    im3 = ax[i,3].imshow(relat_err[i, camy:(camy+deltay),camx:(camx+deltax)]*100,cmap=cmap3,
                         vmin=-20, 
                         vmax=20)
    ax[i,3].tick_params(axis="both", which='major', labelsize=12)
    ax[i,3].set_title("Relative error in %s"%title[i], fontsize=20)
    cbar3=fig.colorbar(im3, ax=ax[i,3],extend="max")
    cbar3.ax.tick_params(labelsize=18)
    cbar3.set_label("mm", rotation=270, fontsize=15, labelpad=12)
    


#%%
