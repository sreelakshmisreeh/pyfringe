#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:00:19 2022

@author: Sreelakshmi
"""
import numpy as np
import nstep_fringe as nstep
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
import os
from copy import deepcopy

EPSILON = -0.5
TAU = 5.5

def inv_mtx(a11,a12,a13,a21,a22,a23,a31,a32,a33):
    '''
    Function to calculate inversion matrix required for object reconstruction.
    Ref: S.Zhong, High-Speed 3D Imaging with Digital Fringe Projection Techniques, CRC Press, 2016.
    '''
   
    
    det = (a11 * a22 * a33) + (a12 * a23 * a31) + (a13 * a21 * a32) - (a13 * a22 * a31) - (a12 * a21 * a33) - (a11* a23* a32)    
    
    b11 = (a22 * a33 - a23 * a32) / det 
    b12 = -(a12 * a33 - a13 * a32) / det 
    b13 = (a12 * a23 - a13 * a22) / det
    
    b21 = -(a21 * a33 - a23 * a31) / det
    b22 = (a11 * a33 - a13 * a31) / det
    b23 = -(a11 * a23 - a13 * a21) / det
    
    b31 = (a21 * a32 - a22 * a31) / det
    b32 = -(a11 * a32 - a12 * a31) / det
    b33 = (a11 * a22 - a12 * a21) / det
    
    #b_mtx=np.stack((np.vstack((b11,b12,b13)).T,np.vstack((b21,b22,b23)).T,np.vstack((b31,b32,b33)).T),axis=1)
    
    return b11, b12, b13, b21, b22, b23, b31, b32, b33
    
    
    
    
def reconstruction_pts(uv_true, unwrapv, c_mtx, c_dist, p_mtx, cp_rot_mtx, cp_trans_mtx, phase_st, pitch):
    '''
    Function to reconstruct 3D point cordinates of 2D points. 

    Parameters
    ----------
    uv_true = type: float. 2D point cordinates
    unwrapv = type: float array. Unwrapped phase map of object.
    c_mtx = type: float array. Camera matrix from calibration.
    c_dist = type: float array. Camera distortion matrix from calibration.
    p_mtx = type: float array. Projector matrix from calibration.
    cp_rot_mtx = type: float array. Projector distortion matrix from calibration.
    cp_trans_mtx = type: float array. Camera-projector translational matrix from calibration.
    phase_st = type:float. Initial phase to be subtracted for phase to coordinate conversion.
    pitch  = type:float. Number of pixels per fringe period.

    Returns
    -------
    Coordinates array for given 2D points
    x = type: float. 
    y = type: float. 
    z = type: float. 

    '''
    no_pts = uv_true.shape[0]
    uv = cv2.undistortPoints(uv_true, c_mtx, c_dist, None, c_mtx )
    uv = uv.reshape(uv.shape[0],2)
    uv_true = uv_true.reshape(no_pts,2)
    #  Extract x and y coordinate of each point as uc, vc
    uc = uv[:,0].reshape(no_pts,1)
    vc = uv[:,1].reshape(no_pts,1)
    
    # Determinate 'up' from circle center
    up = np.array([(nstep.bilinear_interpolate(unwrapv,i) - phase_st) * (pitch / (2*np.pi)) for i in uv_true])
    up = up.reshape(no_pts,1)
    
    # Calculate H matrix for proj from intrinsics and extrinsics
    proj_h_mtx = np.dot(p_mtx, np.hstack((cp_rot_mtx, cp_trans_mtx)))
    #Calculate H matrix for camer
    cam_h_mtx = np.dot(c_mtx,np.hstack((np.identity(3), np.zeros((3,1)))))
    
    a11 = cam_h_mtx[0,0] - uc * cam_h_mtx[2,0]
    a12 = cam_h_mtx[0,1] - uc * cam_h_mtx[2,1]
    a13 = cam_h_mtx[0,2] - uc * cam_h_mtx[2,2]
    
    a21 = cam_h_mtx[1,0] - vc * cam_h_mtx[2,0]
    a22 = cam_h_mtx[1,1] - vc * cam_h_mtx[2,1]
    a23 = cam_h_mtx[1,2] - vc * cam_h_mtx[2,2]
    
    a31 = proj_h_mtx[0,0] - up * proj_h_mtx[2,0]
    a32 = proj_h_mtx[0,1] - up * proj_h_mtx[2,1]
    a33 = proj_h_mtx[0,2] - up * proj_h_mtx[2,2]
    
    b11, b12, b13, b21, b22, b23, b31, b32, b33 = inv_mtx(a11, a12, a13, a21, a22, a23, a31, a32,a33)
    
    c1 = uc * cam_h_mtx[2,3] - cam_h_mtx[0,3]
    c2 = vc * cam_h_mtx[2,3] - cam_h_mtx[1,3]
    c3 = up * proj_h_mtx[2,3] - proj_h_mtx[0,3]
   
    x = b11 * c1 + b12 * c2 + b13 * c3
    y = b21 * c1 + b22 * c2 + b23 * c3
    z = b31 * c1 + b32 * c2 + b33 * c3
    return x, y, z

def point_error(cord1,cord2):
    '''
    Function to plot error 

    '''
    
    delta = cord1 - cord2
    abs_delta = abs(delta)
    err_df =  pd.DataFrame(np.hstack((delta,abs_delta)) , columns = ['$\Delta x$','$\Delta y$','$\Delta z$','$abs(\Delta x)$', '$abs(\Delta y)$', '$abs(\Delta z)$'])
    plt.figure()
    gfg = sns.histplot(data = err_df[['$abs(\Delta x)$', '$abs(\Delta y)$', '$abs(\Delta z)$']])
    plt.xlabel('Absolute error mm',fontsize = 30)
    plt.ylabel('Count',fontsize = 30)
    plt.title('Reconstruction error',fontsize=30)
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.xlim(0,3)
    plt.setp(gfg.get_legend().get_texts(), fontsize='20') 
    return err_df
    
def reconstruction_obj(unwrapv, c_mtx, c_dist, p_mtx, cp_rot_mtx, cp_trans_mtx, phase_st, pitch):
    '''
    Sub function to reconstruct object from phase map

    Parameters
    ----------
    unwrapv = type: float array. Unwrapped phase map of object.
    c_mtx = type: float array. Camera matrix from calibration.
    c_dist = type: float array. Camera distortion matrix from calibration.
    p_mtx = type: float array. Projector matrix from calibration.
    cp_rot_mtx = type: float array. Projector distortion matrix from calibration.
    cp_trans_mtx = type: float array. Camera-projector translational matrix from calibration.
    phase_st = type: float. Initial phase to be subtracted for phase to coordinate conversion.
    pitch  = type:float. Number of pixels per fringe period.

    Returns
    -------
    Coordinates array for all points
    x = type: float array . 
    y = type: float array. 
    z = type: float array. 
    '''
    
    unwrap_dist = cv2.undistort(unwrapv, c_mtx, c_dist)
    u = np.arange(0,unwrap_dist.shape[1])
    v = np.arange(0,unwrap_dist.shape[0])
    uc, vc = np.meshgrid(u,v)
    up = (unwrap_dist - phase_st) * pitch / (2*np.pi) 
    # Calculate H matrix for proj from intrinsics and extrinsics
    proj_h_mtx = np.dot(p_mtx, np.hstack((cp_rot_mtx, cp_trans_mtx)))

    #Calculate H matrix for camera
    cam_h_mtx = np.dot(c_mtx,np.hstack((np.identity(3), np.zeros((3,1)))))

    a11 = cam_h_mtx[0,0] - uc * cam_h_mtx[2,0] 
    a12 = cam_h_mtx[0,1] - uc * cam_h_mtx[2,1]
    a13 = cam_h_mtx[0,2] - uc * cam_h_mtx[2,2]

    a21 = cam_h_mtx[1,0] - vc * cam_h_mtx[2,0]
    a22 = cam_h_mtx[1,1] - vc * cam_h_mtx[2,1]
    a23 = cam_h_mtx[1,2] - vc * cam_h_mtx[2,2]

    a31 = proj_h_mtx[0,0] - up * proj_h_mtx[2,0]
    a32 = proj_h_mtx[0,1] - up * proj_h_mtx[2,1]
    a33 = proj_h_mtx[0,2] - up * proj_h_mtx[2,2]

    b11, b12, b13, b21, b22, b23, b31, b32, b33 = inv_mtx(a11, a12, a13, a21, a22, a23, a31, a32, a33)
    
    c1 = uc * cam_h_mtx[2,3] - cam_h_mtx[0,3]
    c2 = vc * cam_h_mtx[2,3] - cam_h_mtx[1,3]
    c3 = up * proj_h_mtx[2,3] - proj_h_mtx[0,3]
    
    x = b11 * c1 + b12 * c2 + b13 * c3
    y = b21 * c1 + b22 * c2 + b23 * c3
    z = b31 * c1 + b32 * c2 + b33 * c3
    return x, y, z 

def complete_recon(unwrap, inte_rgb, modulation, recon_limit, dist,delta_dist, c_mtx, c_dist, p_mtx, cp_rot_mtx, cp_trans_mtx, phase_st, pitch, obj_path):
    '''
    Function to completely reconstruct object applying modulation mask to saving point cloud.

    Parameters
    ----------
    unwrap = type: float array. Unwrapped phase map of object.
    inte_rgb = type: float array. Texture image.
    modulation = type: float array. Intensity modulation image.
    limit = type: float. Intensity modulation limit for mask.
    dist = type: float. Distance at which object is placed in mm.
    delta_dist = type: float. Volumetric distance to remove outliers.
    c_mtx = type: float array. Camera matrix from calibration.
    c_dist = type: float array. Camera distortion matrix from calibration.
    p_mtx = type: float array. Projector matrix from calibration.
    cp_rot_mtx = type: float array. Projector distortion matrix from calibration.
    cp_trans_mtx = type: float array. Camera-projector translational matrix from calibration.
    phase_st = type: float. Initial phase to be subtracted for phase to coordinate conversion.
    pitch  = type:float. Number of pixels per fringe period.
    obj_path = type: string. Path to save point 3D reconstructed point cloud. 

    Returns
    -------
    cordi = type:  float array. x,y,z coordinate array of each object point.
    intensity = type: float array. Intensity (texture/ color) at each point.

    '''
    roi_mask = np.full(unwrap.shape, False)
    roi_mask[modulation > recon_limit] = True
    u_copy = deepcopy(unwrap)
    w_copy = deepcopy(inte_rgb)
    u_copy[~roi_mask] = np.nan
    w_copy[~roi_mask] = False
    obj_x, obj_y,obj_z = reconstruction_obj(u_copy, c_mtx, c_dist, p_mtx, cp_rot_mtx, cp_trans_mtx, phase_st, pitch)
    flag = (obj_z > (dist - delta_dist)) & (obj_z < (dist + delta_dist)) & roi_mask
    xt = obj_x[flag]
    yt = obj_y[flag]
    zt = obj_z[flag]
    intensity = w_copy[flag] / np.nanmax(w_copy[flag])
    cordi = np.vstack((xt, yt, zt)).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cordi)
    pcd.colors = o3d.utility.Vector3dVector(intensity)
    o3d.io.write_point_cloud(os.path.join(obj_path,'obj.ply'), pcd)
    return cordi,intensity

def obj_reconst_wrapper(width, height, pitch_list, N_list, limit, recon_limit, dist, delta_dist, phase_st, direc, type_unwrap, calib_path, obj_path, kernel = 1):
   '''
    Function for 3D reconstruction of object based on different unwrapping method.

    Parameters
    ----------
    width =type: float. Width of projector.
    height = type: float. Height of projector.
    pitch_list : TYPE
        DESCRIPTION.
    N_list = type: float array. The number of steps in phase shifting algorithm. If phase coded unwrapping method is used this is a single element array. For other methods corresponding to each pitch one element in the list.
    limit = type: float array. Array of number of pixels per fringe period.
    dist = type: float. Distance at which object is placed in mm.
    delta_dist = type: float. Volumetric distance to remove outliers.
    phase_st = type: float. Initial phase to be subtracted for phase to coordinate conversion.
    direc = type: string. Visually vertical (v) or horizontal(h) pattern.
    type_unwrap = type: string. Type of temporal unwrapping to be applied. 
                  'phase' = phase coded unwrapping method, 
                  'multifreq' = multifrequency unwrapping method
                  'multiwave' = multiwavelength unwrapping method.
    calib_path = type: string. Path to read calibration paraneters.
    obj_path = type: string. Path to read captured images
    kernel = type: int. Kernel size for median filter. The default is 1.

    Returns
    -------
    obj_cordi = type : float array. Array of reconstructed x,y,z coordinates of each points on the object
    obj_color = type: float array. Color (texture/ intensity) at each point.

    '''
   
   calibration = np.load(os.path.join(calib_path,'{}_calibration_param.npz'.format(type_unwrap)))
   c_mtx = calibration["arr_0"]
   c_dist = calibration["arr_1"]
   p_mtx = calibration["arr_2"]
   cp_rot_mtx = calibration["arr_3"]
   cp_trans_mtx = calibration["arr_4"]
   if type_unwrap == 'phase':
       object_cos, obj_cos_mod, obj_cos_avg, obj_cos_gamma, delta_deck_cos  = nstep.mask_img(np.array([cv2.imread(os.path.join(obj_path,'capt_%d.jpg'%i),0) for i in range(0, N_list[0])]), limit)
       object_step, obj_step_mod, obj_step_avg, obj_step_gamma, delta_deck_step = nstep.mask_img(np.array([cv2.imread(os.path.join(obj_path,'capt_%d.jpg'%i),0) for i in range(N_list[0],2 * N_list[0])]), limit)

       #wrapped phase
       phase_cos = nstep.phase_cal(object_cos, N_list, delta_deck_cos )
       phase_step = nstep.phase_cal(object_step, N_list, delta_deck_step )
       phase_step = nstep.step_rectification(phase_step,direc)
       #unwrapped phase
       unwrap0, k0 = nstep.unwrap_cal(phase_step, phase_cos, pitch_list[0], width, height, direc)
       unwrap, k = nstep.filt(unwrap0, kernel, direc)
       inte_img = cv2.imread(os.path.join(obj_path,'white.jpg'))
       inte_rgb = inte_img[...,::-1].copy()
       obj_cordi, obj_color = complete_recon(unwrap, inte_rgb, obj_cos_mod, recon_limit, dist, delta_dist, c_mtx, c_dist, p_mtx, cp_rot_mtx,cp_trans_mtx, phase_st, pitch_list[-1], obj_path)
       
   elif type_unwrap == 'multifreq':
       object_freq1, mod_freq1, avg_freq1, gamma_freq1, delta_deck_freq1  = nstep.mask_img(np.array([cv2.imread(os.path.join(obj_path,'capt_%d.jpg'%i),0) for i in range(0, N_list[0])]), limit)
       object_freq2, mod_freq2, avg_freq2, gamma_freq2, delta_deck_freq2 = nstep.mask_img(np.array([cv2.imread(os.path.join(obj_path,'capt_%d.jpg'%i),0) for i in range(N_list[0], N_list[0] + N_list[1])]), limit)
       object_freq3, mod_freq3, avg_freq3, gamma_freq3, delta_deck_freq3 = nstep.mask_img(np.array([cv2.imread(os.path.join(obj_path,'capt_%d.jpg'%i),0) for i in range( N_list[0] + N_list[1], N_list[0]+ N_list[1]+ N_list[2])]), limit)
       object_freq4, mod_freq4, avg_freq4, gamma_freq4, delta_deck_freq4 = nstep.mask_img(np.array([cv2.imread(os.path.join(obj_path,'capt_%d.jpg'%i),0) for i in range(N_list[0]+ N_list[1]+ N_list[2], N_list[0]+ N_list[1]+ N_list[2] + N_list[3])]), limit)

       #wrapped phase
       phase_freq1 = nstep.phase_cal(object_freq1, N_list[0], delta_deck_freq1 )
       phase_freq2 = nstep.phase_cal(object_freq2, N_list[1], delta_deck_freq2 )
       phase_freq3 = nstep.phase_cal(object_freq3, N_list[2], delta_deck_freq3 )
       phase_freq4 = nstep.phase_cal(object_freq4, N_list[3], delta_deck_freq4 )
       phase_freq1[phase_freq1 < EPSILON] = phase_freq1[phase_freq1 < EPSILON] + 2 * np.pi

       #unwrapped phase
       phase_arr = np.stack([phase_freq1, phase_freq2, phase_freq3, phase_freq4])
       unwrap, k = nstep.multifreq_unwrap(pitch_list, phase_arr, kernel, direc)
       inte_img = cv2.imread(os.path.join(obj_path,'white.jpg'))
       inte_rgb = inte_img[...,::-1].copy()
       obj_cordi, obj_color = complete_recon(unwrap, inte_rgb, mod_freq4, recon_limit, dist, delta_dist, c_mtx, c_dist, p_mtx, cp_rot_mtx,cp_trans_mtx, phase_st, pitch_list[-1], obj_path)
       
   elif type_unwrap == 'multiwave':
       eq_wav12 = (pitch_list[-1] * pitch_list[1]) / (pitch_list[1]-pitch_list[-1])
       eq_wav123 = pitch_list[0] * eq_wav12 / (pitch_list[0] - eq_wav12)

       pitch_list = np.insert(pitch_list, 0, eq_wav123)
       pitch_list = np.insert(pitch_list, 2, eq_wav12)
       
       object_wav3, mod_wav3, avg_wav3, gamma_wav1, delta_deck_wav3 = nstep.mask_img(np.array([cv2.imread(os.path.join(obj_path,'capt_%d.jpg'%i),0) for i in range(0, N_list[0])]), limit)
       object_wav2, mod_wav2, avg_wav2, gamma_wav2, delta_deck_wav2 = nstep.mask_img(np.array([cv2.imread(os.path.join(obj_path,'capt_%d.jpg'%i),0) for i in range(N_list[0], N_list[0] + N_list[1])]), limit)
       object_wav1, mod_wav1, avg_wav1, gamma_wav3, delta_deck_wav1 = nstep.mask_img(np.array([cv2.imread(os.path.join(obj_path,'capt_%d.jpg'%i),0) for i in range(N_list[0] + N_list[1], N_list[0]+ N_list[1]+ N_list[2])]), limit)

       #wrapped phase
       phase_wav1 = nstep.phase_cal(object_wav1, N_list[2], delta_deck_wav1 )
       phase_wav2 = nstep.phase_cal(object_wav2, N_list[1], delta_deck_wav2 )
       phase_wav3 = nstep.phase_cal(object_wav3, N_list[0], delta_deck_wav3 )
       phase_wav12 = np.mod(phase_wav1 - phase_wav2, 2 * np.pi)
       phase_wav123 = np.mod(phase_wav12 - phase_wav3, 2 * np.pi)
       # phase_wav123 = nstep.edge_rectification(phase_wav123, 'v')
       phase_wav123[phase_wav123 > TAU] = phase_wav123[phase_wav123 > TAU] - 2 * np.pi

       #unwrapped phase
       phase_arr = np.stack([phase_wav123, phase_wav3, phase_wav12, phase_wav2, phase_wav1])
       unwrap, k = nstep.multiwave_unwrap(pitch_list, phase_arr, kernel, direc)
       inte_img = cv2.imread(os.path.join(obj_path,'white.jpg'))
       inte_rgb = inte_img[...,::-1].copy()
       obj_cordi, obj_color = complete_recon(unwrap, inte_rgb, mod_wav3, recon_limit, dist, delta_dist, c_mtx, c_dist, p_mtx, cp_rot_mtx,cp_trans_mtx, phase_st, pitch_list[-1], obj_path)   
   return obj_cordi, obj_color
       
       
       
       
