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

def inv_mtx(a11,a12,a13,a21,a22,a23,a31,a32,a33):
   
    
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
    
    
    
    
def reconstruction_pts(uv_true,unwrapv,c_mtx,c_dist,p_mtx,cp_rot_mtx,cp_trans_mtx,phi0,pitch):
    no_pts = uv_true.shape[0]
    uv = cv2.undistortPoints(uv_true, c_mtx, c_dist, None, c_mtx )
    uv = uv.reshape(uv.shape[0],2)
    uv_true = uv_true.reshape(no_pts,2)
    #  Extract x and y coordinate of each point as uc, vc
    uc = uv[:,0].reshape(no_pts,1)
    vc = uv[:,1].reshape(no_pts,1)
    
    # Determinate 'up' from circle center
    up = np.array([(nstep.bilinear_interpolate(unwrapv,i) - phi0) * (pitch / (2*np.pi)) for i in uv_true])
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
    
def reconstruction_obj(unwrapv, c_mtx, c_dist, p_mtx, cp_rot_mtx, cp_trans_mtx, phi0, pitch):
    
    unwrap_dist = cv2.undistort(unwrapv, c_mtx, c_dist)
    u = np.arange(0,unwrap_dist.shape[1])
    v = np.arange(0,unwrap_dist.shape[0])
    uc, vc = np.meshgrid(u,v)
    up = (unwrap_dist - phi0) * pitch / (2*np.pi) 
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

def complete_recon(unwrap, inte_rgb, modulation, limit, dist,delta_dist, c_mtx, c_dist, p_mtx, cp_rot_mtx, cp_trans_mtx, phi0, pitch, obj_path):
    
    roi_mask = np.full(unwrap.shape, False)
    roi_mask[modulation > limit] = True
    unwrap[~roi_mask] = np.nan
    inte_rgb[~roi_mask] = False
    obj_x, obj_y,obj_z = reconstruction_obj(unwrap, c_mtx, c_dist, p_mtx, cp_rot_mtx, cp_trans_mtx, phi0, pitch)
    flag = (obj_z > (dist - delta_dist)) & (obj_z < (dist + delta_dist))
    xt = obj_x[flag]
    yt = obj_y[flag]
    zt = obj_z[flag]
    intensity = inte_rgb[flag] / np.nanmax(inte_rgb[flag])
    cordi = np.vstack((xt, yt, zt)).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cordi)
    pcd.colors = o3d.utility.Vector3dVector(intensity)
    o3d.io.write_point_cloud(os.path.join(obj_path,'obj.ply'), pcd)
    return cordi,intensity
    

