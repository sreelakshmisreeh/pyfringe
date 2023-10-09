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
from matplotlib.colors import Normalize
from matplotlib import ticker
plt.rcParams.update({'font.family':'Times New Roman',"mathtext.fontset":"cm"})
import seaborn as sns

def plot_indiv(plot,
               cmap, 
               extend,
               norm,
               title,
               save_title,
               save_path,
               formatter,
               unit,
               colorbar):
    
    fig, ax = plt.subplots(constrained_layout=True)
    cmap = plt.cm.get_cmap(cmap).copy()
    cmap.set_over('red')
    if extend == "both":
        cmap.set_under('blue')
    im = ax.imshow(plot,cmap=cmap, norm=norm)
    if colorbar:
        cbar=fig.colorbar(im, ax=ax, extend=extend, pad=0.0)
        ax.set_title(title, fontsize=25)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        cbar.ax.tick_params(labelsize=20)
        if formatter:
            cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True, 
                                                                     useOffset=True))
            cbar.ax.ticklabel_format(style='sci', scilimits=(0, 0))
            cbar.ax.yaxis.get_offset_text().set(size=20, va="bottom", ha="left")
            cbar.set_label("%s"%unit, rotation=270, fontsize=25, labelpad=20)
        else:
            cbar.set_label("%s"%unit, rotation=270, fontsize=25, labelpad=20)
    plt.savefig(os.path.join(save_path,"pdf\%s.pdf"%save_title))
    plt.savefig(os.path.join(save_path,"tiff\%s.tiff"%save_title))
       
    return

def plot_sigmas(cord_sigma_image, 
                emp_std, 
                fig_title, 
                camx, camy, 
                deltax, deltay,
                object_name,
                prob_up,
                save_indiv=False,save_path=None):
    title = [r"$\sigma_x$", r"$\sigma_y$", r"$\sigma_z$"]
    diff_cord = cord_sigma_image-emp_std
    relat_err = diff_cord/emp_std
    plt_lst = [emp_std[:,camy:(camy+deltay),camx:(camx+deltax)],
               cord_sigma_image[:, camy:(camy+deltay),camx:(camx+deltax)],
               diff_cord[:, camy:(camy+deltay),camx:(camx+deltax)],
               relat_err[:, camy:(camy+deltay),camx:(camx+deltax)]*100]
    cmap_lst = ["inferno", "inferno", "BrBG", "BrBG"]
    extend_lst = ["max","max","both","both"]
    unit_lst = ["mm","mm","mm","percentage %"]
    formatter_lst = [True, True, True, False]
    fig, ax = plt.subplots(3,4)
    fig.suptitle("%s"%fig_title, fontsize=20)
    if prob_up:
        cord_lst = ["xup","yup","zup"]
    else:
        cord_lst = ["xall","yall","zall"]
    
    for j in range(3):
        norm_lst = [Normalize(vmin=np.nanquantile(emp_std[j],0.03), vmax=np.nanquantile(emp_std[j],0.97)),
                    Normalize(vmin=np.nanquantile(emp_std[j],0.03), vmax=np.nanquantile(emp_std[j],0.97)),
                    Normalize(vmin=np.nanquantile(diff_cord[j],0.03), vmax=np.nanquantile(diff_cord[j],0.97)),
                    Normalize(vmin=-10,  vmax=10)]
        title_lst = ["Empirical %s \nfrom multiple scans"%title[j],
                     "Predicted %s \nfrom a single scan"%title[j],
                     "Predicted - Emperical", 
                     r"$\frac{Predicted - Emperical}{Emperical}$ in %"]
        save_title = ["%s_emp_%s_std"%(object_name,cord_lst[j]), 
                      "%s_pred_%s_std"%(object_name,cord_lst[j]), 
                      "%s_diff_%s_std"%(object_name,cord_lst[j]), 
                      "%s_rel_%s_std"%(object_name,cord_lst[j])]
        for i in range(4):
            if save_indiv:
                plot_indiv(plt_lst[i][j],
                           cmap_lst[i], 
                           extend_lst[i],
                           norm_lst[i],
                           title_lst[i],
                           save_title[i] ,
                           save_path,
                           formatter_lst[i],
                           unit_lst[i],
                           True)
            cmap = plt.cm.get_cmap(cmap_lst[i]).copy()
            cmap.set_over('red')
            if extend_lst[i] == "both":
                cmap.set_under('blue')
            im = ax[j,i].imshow(plt_lst[i][j],cmap=cmap, norm=norm_lst[i])
            ax[j,i].tick_params(axis="both", which='major', labelsize=12)
            ax[j,i].set_title(title_lst[i], fontsize=20)
            ax[j,i].axes.xaxis.set_visible(False)
            ax[j,i].axes.yaxis.set_visible(False)
            cbar=fig.colorbar(im, ax=ax[j,i],extend=extend_lst[i], pad=0.0)
            if formatter_lst[i]:
                cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True, useOffset=True))
                cbar.ax.ticklabel_format(style='sci', scilimits=(0, 0))
                cbar.ax.yaxis.get_offset_text().set(size=20, va="bottom", ha="left")
            cbar.set_label("%s"%unit_lst[i], rotation=270, fontsize=25, labelpad=20)
            cbar.ax.tick_params(labelsize=15)
    return

proj_width = 912  
proj_height = 1140 
cam_width = 1920 
cam_height = 1200
limit = 150
camx = 750 #c3:450#c2:500#c550#s750 #w410;900
camy = 420 #c3:200#c2:300#c300#s420 #w300;400
deltax =390#c3:1000#c2:1020#c800 #s390 #w1000;200
deltay = 380#c3:890#c2:890#c760 #s380 #w760;200
type_unwrap = 'multifreq'
dark_bias_path = r"C:\Users\kl001\Documents\pyfringe_test\mean_pixel_std\exp_30_fp_42_retake\black_bias\avg_dark.npy"
calib_path = r'C:\Users\kl001\Documents\pyfringe_test\multifreq_calib_images_bk'
obj_dir =  r"E:\green_small_sphere_single"
obj_dir2 =  r"E:\green_freq_small_sphere\pitch_18"  
model_path = r"C:\Users\kl001\Documents\pyfringe_test\mean_pixel_std\exp_30_fp_42_retake\const_tiff\calib_fringes\variance_model.npy"
object_name = "sphere"
pitch_list = [1200, 18]
N_list = [3, 3]
save_indiv= False
save_path= r"E:\paper_figures\%s"%object_name
#%%
prob_up=True
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

obj_cordi, obj_color, cordi_sigma = reconst_inst.obj_reconst_wrapper(prob_up=prob_up)
 
if prob_up:
    cord_image = np.array([nstep.recover_image(obj_cordi[:,i],reconst_inst.mask,cam_height,cam_width) for i in range(3)])
    cord_sigma_image = np.array([nstep.recover_image(cordi_sigma[:,i], reconst_inst.mask,cam_height,cam_width) for i in range(3)])
    np.save(os.path.join(obj_dir,'dist_single_cord_image.npy'), cord_image)
    np.save(os.path.join(obj_dir,'dist_singlecord_sigma_image.npy'), cord_sigma_image)
    np.save(os.path.join(obj_dir,'dist_single_cordi_std.npy'), cordi_sigma)
    np.save(os.path.join(obj_dir,'dist_single_cordi.npy'), obj_cordi)
    np.save(os.path.join(obj_dir,'dist_single_mask.npy'), reconst_inst.mask)
   
    multi_img = np.load(os.path.join(obj_dir2,"dist_corrected_cord_mean.npy"))
    multi_std = np.load(os.path.join(obj_dir2,"dist_corrected_cord_std.npy"))
    multimask = np.load(os.path.join(obj_dir2,"dist_corrected_mask.npy"))
    plot_sigmas(cord_sigma_image, multi_std, "Standard deviation with aleotry uncertainty",
                camx, camy, deltax, deltay, object_name,prob_up,
                save_indiv=save_indiv,save_path=save_path)
else:
    cord_image = np.array([nstep.recover_image(obj_cordi[:,i],reconst_inst.mask,cam_height,cam_width) for i in range(3)])
    cord_sigma_image = np.array([nstep.recover_image(cordi_sigma[:,i], reconst_inst.mask,cam_height,cam_width) for i in range(3)])
    np.save(os.path.join(obj_dir,'allparam_single_cord_image.npy'), cord_image)
    np.save(os.path.join(obj_dir,'allparam_singlecord_sigma_image.npy'), cord_sigma_image)
    np.save(os.path.join(obj_dir,'allparam_single_cordi_std.npy'), cordi_sigma)
    np.save(os.path.join(obj_dir,'allparam_single_cordi.npy'), obj_cordi)
    np.save(os.path.join(obj_dir,'allparam_single_mask.npy'), reconst_inst.mask)
    monte_img = np.load(os.path.join(obj_dir2,"monte_mean_cords_int_ext_concrete.npy"))
    monte_std = np.load(os.path.join(obj_dir2,"monte_std_cords_int_ext_concrete.npy"))
    montemask = np.load(os.path.join(obj_dir2,"monte_mask_int_ext_concrete.npy"))
    plot_sigmas(cord_sigma_image, monte_std, "Standard deviation with aleotry and epistemic uncertainty",
                camx, camy, deltax, deltay,
                object_name,prob_up,
                save_indiv=save_indiv,save_path=save_path )

#%%


