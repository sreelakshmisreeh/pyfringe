#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 17:36:17 2023

@author: Sreelakshmi

This file does analysis based on diffrent intrinsic and extrinsic at different Mahalanobis Distance.
"""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys
import shutil
#sys.path.append(r'C:\Users\kl001\pyfringe')
import reconstruction as rc
import nstep_fringe as nstep


def calculateMahalanobis(df):
    """
    Function to calculate Mahalanobis Distance
    """
    y_mu = df - df.mean()
    cov = np.cov(df.values.T)
    cov = np.diag(np.diag(cov))
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(y_mu, inv_covmat)
    mahal = np.dot(left, y_mu.T)
    return mahal.diagonal()

def reshape_mtx(param, md_param_path):
    """
    Sub function to reshape from column format to matrix format and save as npy.
    """
    c_mtx = param[:,0:9].reshape(3,3,3)
    c_dist = param[:,9:14].reshape(3,1,5)
    p_mtx = param[:,14:23].reshape(3,3,3)
    cp_rot_mtx = param[:,23:32].reshape(3,3,3)
    cp_trans_mtx = param[:,32:35].reshape(3,3,1)
    proj_h_mtx = param[:,35:47].reshape(3,3,4)
    cam_h_mtx = param[:,47:59].reshape(3,3,4)
    [np.savez(os.path.join(md_param_path,'md_param%d.npz'%i), 
              cam_mtx_mean=c_mtx[i], 
              cam_dist_mean=c_dist[i],
              proj_mtx_mean=p_mtx[i],
              proj_dist_mean=np.zeros(c_dist[i].shape), 
              st_rmat_mean=cp_rot_mtx[i],
              st_tvec_mean=cp_trans_mtx[i],
              cam_h_mtx_mean=cam_h_mtx[i],
              proj_h_mtx_mean=proj_h_mtx[i]) for i in range (0,param.shape[0])]
    return c_mtx, c_dist, p_mtx, cp_rot_mtx, cp_trans_mtx, proj_h_mtx, cam_h_mtx

def intrinsic_extrinsic_MD(cam_mtx_sample,
                           cam_dist_sample,
                           proj_mtx_sample,
                           st_rmat_sample,
                           st_tvec_sample,
                           proj_h_mtx_sample,
                           cam_h_mtx_sample,
                           md_quantile_list):
    """
    Function to get intrinsic extrinsic parameters at a given MD quantiles
    """
    data = np.concatenate((cam_mtx_sample.reshape(cam_mtx_sample.shape[0],cam_mtx_sample.shape[1] * cam_mtx_sample.shape[2]),
                           cam_dist_sample.reshape(cam_dist_sample.shape[0],cam_dist_sample.shape[1] * cam_dist_sample.shape[2]),
                           proj_mtx_sample.reshape(proj_mtx_sample.shape[0],proj_mtx_sample.shape[1] * proj_mtx_sample.shape[2]),
                           st_rmat_sample.reshape(st_rmat_sample.shape[0],st_rmat_sample.shape[1] * st_rmat_sample.shape[2]),
                           st_tvec_sample.reshape(st_tvec_sample.shape[0],st_tvec_sample.shape[1] * st_tvec_sample.shape[2]),
                           proj_h_mtx_sample.reshape(proj_h_mtx_sample.shape[0],proj_h_mtx_sample.shape[1] * proj_h_mtx_sample.shape[2]),
                           cam_h_mtx_sample.reshape(cam_h_mtx_sample.shape[0],cam_h_mtx_sample.shape[1] * cam_h_mtx_sample.shape[2])), 
                          axis=1)

    col_names = ['c_mtx_11','c_mtx_12','c_mtx_13','c_mtx_21','c_mtx_22','c_mtx_23','c_mtx_31','c_mtx_32','c_mtx_33',
                 'c_dist_1','c_dist_2','c_dist_3','c_dist_4','c_dist_5',
                 'p_mtx_11','p_mtx_12','p_mtx_13','p_mtx_21','p_mtx_22','p_mtx_23','p_mtx_31','p_mtx_32','p_mtx_33',
                 'st_r_11','st_r_12','st_r_13','st_r_21','st_r_22','st_r_23','st_r_31','st_r_32','st_r_33',
                 'st_t_1','st_t_2','st_t_3',
                 'hp_11','hp_12','hp_13','hp_14','hp_21','hp_22','hp_23','hp_24','hp_31','hp_32','hp_33','hp_34', 
                 'hc_11','hc_12','hc_13','hc_14','hc_21','hc_22','hc_23','hc_24','hc_31','hc_32','hc_33','hc_34']
    df = pd.DataFrame(data, columns = col_names)
    df_sub = df.iloc[:,-24:]
    df_sub = df_sub.loc[:, (df_sub != 0).any(axis=0)] # to drop all zero only columns
    df_sub = df_sub.loc[:,(df_sub != 1).any(axis = 0)]
    df_sub['MD_sq']= calculateMahalanobis(df_sub)
    df['MD_sq'] = df_sub['MD_sq']
    df['MD']= np.sqrt(df['MD_sq'])

    # Find parameters close to given MD
    md_list = df['MD'].quantile(md_quantile_list).to_list()
    idx = df['MD'].sub(md_list[0]).abs().idxmin()
    param_1 = np.array(df.loc[idx])
    idx = df['MD'].sub(md_list[1]).abs().idxmin()
    param_2 = np.array(df.loc[idx])
    idx = df['MD'].sub(md_list[2]).abs().idxmin()
    param_3 = np.array(df.loc[idx])
    param = np.stack((param_1,param_2,param_3), axis = 0)
    # Reshape as matrix and save
    up_c_mtx, up_c_dist, up_p_mtx, up_cp_rot_mtx, up_cp_trans_mtx, up_proj_h_mtx,up_cam_h_mtx = reshape_mtx(param, md_param_path)
    # Plot hisogram of MD values and show corresponding quantile
    fig, ax = plt.subplots()
    sns.distplot(df['MD'],ax= ax, label = 'Data', kde = False)
    ax.set_xlabel('Mahalanobis distance, D', fontsize = 15)
    ax.set_ylabel('Density', fontsize = 15)
    ax.tick_params(axis = 'both', labelsize = 15)
    ax.vlines(x = md_list[0] , ymin = 0, ymax =25, color = 'r', linestyle = '--', label = 'MD = %.3f (0.05 Q)'%md_list[0])
    ax.vlines(x =md_list[1] , ymin = 0, ymax =25, color = 'g', linestyle = '--', label = 'MD = %.3f (0.50 Q)'%md_list[1])
    ax.vlines(x = md_list[2] , ymin = 0, ymax =25, color = 'b', linestyle = '--', label = 'MD = %.3f (0.95 Q)'%md_list[2])
    plt.legend(fontsize = 20)
    return md_list
    
def md_obj_reconst_withstd(proj_width,
                           proj_height,
                           cam_width,
                           cam_height,
                           type_unwrap,
                           limit,
                           N_list,
                           pitch_list,
                           kernel,
                           data_type,
                           processing,
                           sigma_path,
                           object_path,
                           md_param_path):
    """
    Function to reconstruct object based on instrinsic and extrinsic at different MD quantiles
    """
    mask_lst = []
    md_cord_lst = []
    mod_lst = []
    for i in range(0,3):
        if  os.path.exists(md_param_path + '{}_mean_calibration_param.npz'.format(type_unwrap)):
            os.remove( md_param_path + '{}_mean_calibration_param.npz'.format(type_unwrap))
        source_path = os.path.join(md_param_path, 'md_param%d.npz'%i)  
        target_path = os.path.join(md_param_path, '{}_mean_calibration_param.npz'.format(type_unwrap))
        shutil.copy(source_path, target_path)

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
                                         data_type='npy',
                                         processing='cpu',
                                         calib_path=md_param_path,
                                         sigma_path=sigma_path,
                                         object_path=object_path,
                                         temp=False,
                                         save_ply=True,
                                         probability=False)
    
        obj_cordi, obj_color, cordi_sigma, mask, modulation_vector = reconst_inst.obj_reconst_wrapper() 
    
        if  os.path.exists( os.path.join(md_param_path, 'obj_param%d.ply'%i)):
            os.remove(os.path.join(md_param_path, 'obj_param%d.ply'%i))
        shutil.copy(os.path.join(object_path,'obj.ply') , os.path.join(md_param_path,'obj_param%d.ply'%i))
        mask_lst.append(mask)
        md_cord_lst.append(obj_cordi)
        mod_lst.append(modulation_vector)
    return md_cord_lst, mask_lst, mod_lst   

def boxplots_MD(z_list, mod_quantile, md_list):
    """
    """
    md_list = np.around(md_list, decimals=3)
    edge_color = ["green","blue","purple"]
    fig, ax = plt.subplots()
    fig.suptitle("Modulation = %f"%mod_quantile)
    bp_x = ax.boxplot(z_list[0])
    bp_y = ax.boxplot(z_list[1])
    bp_z = ax.boxplot(z_list[-1])
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp_x[element], color=edge_color[0])
            plt.setp(bp_y[element], color=edge_color[1])
            plt.setp(bp_z[element], color=edge_color[2])
    ax.set_xlabel("Mahalanobis distance, D", fontsize = 15)
    ax.set_ylabel("Z score", fontsize = 15)
    ax.legend([bp_x["boxes"][0], bp_y["boxes"][0], bp_z["boxes"][0]], ['x', 'y', 'z'], title = 'Boxplot for')
    ax.set_xticks([1,2,3],md_list)
    return

def z_score_plot(min_md_df, mean_md_df, max_md_df, mod_quantile, md_list):
    """
    find values close to given modulations
    """
    zscore_x = [min_md_df['Z_score_x'].loc[(min_md_df['modulation']- mod_quantile).abs()< 0.01].values.tolist(),
                mean_md_df['Z_score_x'].loc[(mean_md_df['modulation'] - mod_quantile).abs()<0.01].values.tolist(), 
                max_md_df['Z_score_x'].loc[(max_md_df['modulation'] - mod_quantile).abs()<0.01].values.tolist()]
    zscore_y = [min_md_df['Z_score_y'].loc[(min_md_df['modulation']- mod_quantile).abs()< 0.01].values.tolist(),
                mean_md_df['Z_score_y'].loc[(mean_md_df['modulation'] - mod_quantile).abs()<0.01].values.tolist(), 
                max_md_df['Z_score_y'].loc[(max_md_df['modulation'] - mod_quantile).abs()<0.01].values.tolist()]
    zscore_z = [min_md_df['Z_score_z'].loc[(min_md_df['modulation']- mod_quantile).abs()< 0.01].values.tolist(),
                mean_md_df['Z_score_z'].loc[(mean_md_df['modulation'] - mod_quantile).abs()<0.01].values.tolist(), 
                max_md_df['Z_score_z'].loc[(max_md_df['modulation'] - mod_quantile).abs()<0.01].values.tolist()]
    z_list = [zscore_x, zscore_y, zscore_z]
    boxplots_MD(z_list, mod_quantile, md_list)
    return
    
def z_score_cal(ground_truth_path, 
                md_cord_lst, 
                mask_lst, 
                mod_lst, 
                md_list, 
                mod_lower_quantile, 
                mod_mean_quantile, 
                mod_upper_quantile):
    """
    Function to calculate Z score for each cordinates generated at different MD intrinsic extrinsic.
    Z scores close to modulation quantile values are used in box plots.
    """
    #Ground truth cordinates (x*,y*,z*) and sigma 
    gt_std_img = np.load(os.path.join(ground_truth_path,'monte_std_cords_%s.npy'%surface))
    gt_mask1 = np.tile(mask_lst[0],(gt_std_img.shape[0],1,1))
    gt_std1 = gt_std_img[gt_mask1].reshape((gt_std_img.shape[0],-1)).T
    gt_cord_img = np.load(os.path.join(ground_truth_path,'monte_mean_cords_%s.npy'%surface))
    gt_cord1 = gt_cord_img[gt_mask1].reshape((gt_cord_img.shape[0], -1)).T
    col_head = ['x','y','z', 'x*', 'y*', 'z*', 'sigma_x*', 'sigma_y*', 'sigma_z*']
    min_md_data = np.concatenate((md_cord_lst[0], gt_cord1, gt_std1), axis = 1)
    min_md_df = pd.DataFrame(min_md_data,  columns = col_head)
    min_md_df['modulation'] = mod_lst[0]
    min_md_df['Z_score_x'] = (min_md_df['x'] - min_md_df['x*'])/min_md_df['sigma_x*']
    min_md_df['Z_score_y'] = (min_md_df['y'] - min_md_df['y*'])/min_md_df['sigma_y*']
    min_md_df['Z_score_z'] = (min_md_df['z'] - min_md_df['z*'])/min_md_df['sigma_z*']
    min_md_df.dropna(inplace=True)

    gt_mask2 = np.tile(mask_lst[1],(gt_std_img.shape[0],1,1))
    gt_std2 = gt_std_img[gt_mask2].reshape((gt_std_img.shape[0],-1)).T
    gt_cord2 = gt_cord_img[gt_mask2].reshape((gt_cord_img.shape[0], -1)).T
    mean_md_data = np.concatenate((md_cord_lst[1], gt_cord2, gt_std2), axis = 1)
    mean_md_df = pd.DataFrame(mean_md_data,  columns = col_head)
    mean_md_df['modulation'] = mod_lst[1]
    mean_md_df['Z_score_x'] = (mean_md_df['x'] - mean_md_df['x*'])/mean_md_df['sigma_x*']
    mean_md_df['Z_score_y'] = (mean_md_df['y'] - mean_md_df['y*'])/mean_md_df['sigma_y*']
    mean_md_df['Z_score_z'] = (mean_md_df['z'] - mean_md_df['z*'])/mean_md_df['sigma_z*']
    mean_md_df.dropna(inplace=True)

    gt_mask3 = np.tile(mask_lst[2],(gt_std_img.shape[0],1,1))
    gt_std3 = gt_std_img[gt_mask3].reshape((gt_std_img.shape[0],-1)).T
    gt_cord3 = gt_cord_img[gt_mask3].reshape((gt_cord_img.shape[0], -1)).T
    max_md_data = np.concatenate((md_cord_lst[2], gt_cord3, gt_std3), axis = 1)
    max_md_df = pd.DataFrame(max_md_data,  columns = col_head)
    max_md_df['modulation'] = mod_lst[2]
    max_md_df['Z_score_x'] = (max_md_df['x'] - max_md_df['x*'])/max_md_df['sigma_x*']
    max_md_df['Z_score_y'] = (max_md_df['y'] - max_md_df['y*'])/max_md_df['sigma_y*']
    max_md_df['Z_score_z'] = (max_md_df['z'] - max_md_df['z*'])/max_md_df['sigma_z*']
    max_md_df.dropna(inplace=True)
    
    #find values close to given modulations and plot
    mean_mod_zscore = z_score_plot(min_md_df, mean_md_df, max_md_df, mod_mean_quantile, md_list)
    lower_mod_zscore = z_score_plot(min_md_df, mean_md_df, max_md_df, mod_lower_quantile, md_list)
    upper_mod_zscore = z_score_plot(min_md_df, mean_md_df, max_md_df, mod_upper_quantile, md_list)
    
    return min_md_df, mean_md_df, max_md_df

    
#%%
type_unwrap =  'multifreq'
data_type = 'npy'
processing = 'cpu'
surface = 'plane'
root_dir = r'/Volumes/My Passport/12Feb2023/' 
bootstrap_path = os.path.join(root_dir, 'bootsrap_analysis')
samples = np.load(os.path.join(bootstrap_path,'{}_sample_calibration_param.npz'.format(type_unwrap)))
md_param_path = '/Users/Sreelakshmi/Documents/Raspberry/codes/22feb2023/MD' 
ground_truth_path =  '/Volumes/My Passport/12Feb2023/Monte_carlo/%s'%surface
sigma_path =  r'/Volumes/My Passport/12Feb2023/reconst_test/mean_std_pixel.npy'
proj_width = 912
proj_height = 1140
cam_width = 1920
cam_height = 1200
direc = 'v'
phase_st = 0
pitch_list =[1000,110,16]#[1375, 275, 55, 11] 
N_list = [3, 3, 9]
temp = False
kernel = 7
md_quantile_list = [0.05,0.5, 0.95]
quantile_limit = 4.5
limit = nstep.B_cutoff_limit(sigma_path, quantile_limit, N_list, pitch_list)
cam_mtx_sample = samples["cam_mtx_sample"][30]  
cam_dist_sample = samples["cam_dist_sample"][30]   
proj_mtx_sample = samples["proj_mtx_sample"][30]   
proj_dist_sample = samples["proj_dist_sample"][30]   
st_rmat_sample = samples["st_rmat_sample"][30]   
st_tvec_sample = samples["st_tvec_sample"][30]   
proj_h_mtx_sample =  samples["proj_h_mtx_sample"][30]   
cam_h_mtx_sample =  samples["cam_h_mtx_sample"][30] 
#%%
md_list = intrinsic_extrinsic_MD(cam_mtx_sample,
                                 cam_dist_sample,
                                 proj_mtx_sample,
                                 st_rmat_sample,
                                 st_tvec_sample,
                                 proj_h_mtx_sample,
                                 cam_h_mtx_sample,
                                 md_quantile_list)

#%%
md_cord_lst, mask_lst, mod_lst =  md_obj_reconst_withstd(proj_width,
                                                         proj_height,
                                                         cam_width,
                                                         cam_height,
                                                         type_unwrap,
                                                         limit,
                                                         N_list,
                                                         pitch_list,
                                                         kernel,
                                                         data_type,
                                                         processing,
                                                         sigma_path,
                                                         object_path=md_param_path,
                                                         md_param_path=md_param_path)
#%% Z score calculations
#Quantile value of modulation to select which Z score to be plotted
flat_mod_lst = [item for sublist in mod_lst for item in sublist]
mod_lower_quantile, mod_mean_quantile, mod_upper_quantile = np.nanquantile(flat_mod_lst,[0.05,0.5, 0.95])
fig, ax = plt.subplots(figsize=(16,9))
sns.distplot(flat_mod_lst,ax= ax, label = 'Modulation data', kde = False)
ax.set_xlabel('Modulation', fontsize = 15)
ax.set_ylabel('Density', fontsize = 15)
ax.tick_params(axis = 'both', labelsize = 15)
ax.axvline(x = mod_lower_quantile , color = 'r', linestyle = '--', label = 'Q1 = %.3f (0.05 Q)'%mod_lower_quantile)
ax.axvline(x =mod_mean_quantile , color = 'g', linestyle = '--', label = 'Q2 = %.3f (0.50 Q)'%mod_mean_quantile)
ax.axvline(x = mod_upper_quantile , color = 'b', linestyle = '--', label = 'Q3 = %.3f (0.95 Q)'%mod_upper_quantile)
plt.legend(fontsize = 20)
plt.tight_layout()
min_md_df, mean_md_df, max_md_df = z_score_cal(ground_truth_path, 
                                                md_cord_lst, 
                                                mask_lst, 
                                                mod_lst, 
                                                md_list, 
                                                mod_lower_quantile, 
                                                mod_mean_quantile, 
                                                mod_upper_quantile)

