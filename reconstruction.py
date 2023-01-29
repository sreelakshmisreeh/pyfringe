#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:00:19 2022

@author: Sreelakshmi
"""
import numpy as np
import cupy as cp
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import os
from copy import deepcopy
from plyfile import PlyData, PlyElement
import nstep_fringe as nstep
import nstep_fringe_cp as nstep_cp

EPSILON = -0.5
TAU = 5.5
#TODO: Move out ply saving. Combine 3level, 4 level to general function for any level.
#TODO: Convert to pyqtgraph. Check sigma functions and convert to cupy. opencv cupy for undistort

def triangulation(uc, vc, up, cam_h_mtx, proj_h_mtx, processing):
    """
    Used for triangulation given camera coordinates uc vc and projector coordinates up, as well as two 'h' matrices

    Parameters
    ----------
    uc : n x 1 cupy.array
        u_c camera coordinate.
    vc : n x 1 cupy.array
        v_c camera coordinate.
    up : n x 1 cupy.array
        u_p projector coordinate
    cam_h_mtx : 3 x 3 cupy.array 
        camera h matrix.
    proj_h_mtx : 3 x 3 cupy.array
        projector h matrix.

    Returns
    -------
    coords:
        x = coords[:,0,0]
        y = coords[:,1,0]
        z = coords[:,2,0]

    """
    n_pixels = len(uc)
    if processing == 'gpu':
        A = cp.empty((n_pixels, 3, 3))
        c = cp.empty((n_pixels, 3, 1))
    else:
        A = np.empty((n_pixels, 3, 3))
        c = np.empty((n_pixels, 3, 1))
    
    A[:, 0, 0] = cam_h_mtx[0, 0] - uc * cam_h_mtx[2, 0]
    A[:, 0, 1] = cam_h_mtx[0, 1] - uc * cam_h_mtx[2, 1]
    A[:, 0, 2] = cam_h_mtx[0, 2] - uc * cam_h_mtx[2, 2]

    A[:, 1, 0] = cam_h_mtx[1, 0] - vc * cam_h_mtx[2, 0]
    A[:, 1, 1] = cam_h_mtx[1, 1] - vc * cam_h_mtx[2, 1]
    A[:, 1, 2] = cam_h_mtx[1, 2] - vc * cam_h_mtx[2, 2]

    A[:, 2, 0] = proj_h_mtx[0, 0] - up * proj_h_mtx[2, 0]
    A[:, 2, 1] = proj_h_mtx[0, 1] - up * proj_h_mtx[2, 1]
    A[:, 2, 2] = proj_h_mtx[0, 2] - up * proj_h_mtx[2, 2]
    if processing == 'gpu':
        A_inv = cp.linalg.inv(A)
    else:
        A_inv = np.linalg.inv(A)
    
    c[:, 0, 0] = uc * cam_h_mtx[2, 3] - cam_h_mtx[0, 3]
    c[:, 1, 0] = vc * cam_h_mtx[2, 3] - cam_h_mtx[1, 3]
    c[:, 2, 0] = up * proj_h_mtx[2, 3] - proj_h_mtx[0, 3]
    if processing == 'gpu':
        coords = cp.einsum('ijk,ikl->lij', A_inv, c)[0]
    else:
        coords = np.einsum('ijk,ikl->lij', A_inv, c)[0]
    return coords
def reconstruction_pts(uv_true, unwrapv, c_mtx, c_dist, p_mtx, cp_rot_mtx, cp_trans_mtx, phase_st, pitch, processing):
    """
    Function to reconstruct 3D point coordinates of 2D points.

    Parameters
    ----------
    :param uv_true: 2D point coordinates
    :param unwrapv: type: float array. Unwrapped phase map of object.
    :param c_mtx: type: float array. Camera matrix from calibration.
    :param c_dist: type: float array. Camera distortion matrix from calibration.
    :param p_mtx: type: float array. Projector matrix from calibration.
    :param cp_rot_mtx: type: float array. Projector distortion matrix from calibration.
    :param cp_trans_mtx: type: float array. Camera-projector translational matrix from calibration.
    :param phase_st: type:float. Initial phase to be subtracted for phase to coordinate conversion.
    :param pitch: type:float. Number of pixels per fringe period.
    :type uv_true: float
    :type unwrapv: float array
    :type c_mtx: float array
    :type c_dist:float array
    :type p_mtx:float array
    :type cp_rot_mtx:float array
    :type cp_trans_mtx:float array
    :type phase_st:float
    :type pitch:float
    :return x,y,z: cordinate arrays
    :rtype x,y,z: float
    """
    no_pts = uv_true.shape[0]
    uv = cv2.undistortPoints(uv_true, c_mtx, c_dist, None, c_mtx)
    uv = uv.reshape(uv.shape[0], 2)
    uv_true = uv_true.reshape(no_pts, 2)
    #  Extract x and y coordinate of each point as uc, vc
    uc = uv[:, 0]
    vc = uv[:, 1]
    # Determinate 'up' from circle center
    up = (nstep.bilinear_interpolate(unwrapv, uv_true) - phase_st) * pitch / (2*np.pi)
    up = up
    # Calculate H matrix for proj from intrinsics and extrinsic
    proj_h_mtx = np.dot(p_mtx, np.hstack((cp_rot_mtx, cp_trans_mtx)))
    #Calculate H matrix for camera
    cam_h_mtx = np.dot(c_mtx, np.hstack((np.identity(3), np.zeros((3, 1)))))
    if processing == 'gpu':
        uc = cp.asarray(uc)
        vc = cp.asarray(vc)
        up = cp.asarray(up)
        cam_h_mtx = cp.asarray(cam_h_mtx)
        proj_h_mtx = cp.asarray(proj_h_mtx)
    coordintes = triangulation(uc, vc, up, cam_h_mtx, proj_h_mtx, processing)
    return cp.asnumpy(coordintes)

def point_error(cord1, cord2):
    '''
    Function to plot error between two coordinate.
    :param cord1: coordinate 1
    :param cord2: coordinate 2
    :type cord1: float array
    :type cord2: float array
    :return err_df: error dataframe
    :rtype err_df: pandas dataframe
    '''
    
    delta = cord1 - cord2
    abs_delta = abs(delta)
    err_df = pd.DataFrame(np.hstack((delta, abs_delta)),
                          columns=['$\Delta x$', '$\Delta y$', '$\Delta z$', '$abs(\Delta x)$', '$abs(\Delta y)$', '$abs(\Delta z)$'])
    plt.figure()
    gfg = sns.histplot(data=err_df[['$abs(\Delta x)$', '$abs(\Delta y)$', '$abs(\Delta z)$']])
    plt.xlabel('Absolute error mm', fontsize=30)
    plt.ylabel('Count', fontsize=30)
    plt.title('Reconstruction error', fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlim(0, 3)
    plt.setp(gfg.get_legend().get_texts(), fontsize='20') 
    return err_df
#TODO: Remove this function if possible    
def reconstruction_obj(unwrapv, cam_mtx, cam_dist, proj_mtx, camproj_rot_mtx, camproj_trans_mtx, phase_st, pitch, processing):
    """
    Sub function to reconstruct object from phase map
    unwrapv = Unwrapped phase map of object.
    c_mtx = Camera matrix from calibration.
    c_dist = Camera distortion matrix from calibration.
    p_mtx = Projector matrix from calibration.
    cp_rot_mtx = Projector distortion matrix from calibration.
    cp_trans_mtx = Camera-projector translational matrix from calibration.
    phase_st = Initial phase to be subtracted for phase to coordinate conversion.
    pitch  = Number of pixels per fringe period.
    :type unwrapv:float array.
    :type cam_mtx: float array.
    :type cam_dist:float array.
    :type proj_mtx:float array.
    :type camproj_rot_mtx:float array.
    :type camproj_trans_mtx:float array.
    :type phase_st:float.
    :type pitch:float.
    :return x,y,z : coorninate arrays
    :rtpe x,y,z : float array
    """
    unwrap_dist = cv2.undistort(unwrapv, cam_mtx, cam_dist)
    u = np.arange(0, unwrap_dist.shape[1])
    v = np.arange(0, unwrap_dist.shape[0])
    uc, vc = np.meshgrid(u, v)
    up = (unwrap_dist - phase_st) * pitch / (2*np.pi) 
    uc = uc.ravel()
    vc = vc.ravel()
    up = up.ravel()
    nan_mask = np.isnan(up)
    uc_updated = uc[~nan_mask]
    vc_updated = vc[~nan_mask]
    up_updated = up[~nan_mask]
    # Calculate H matrix for proj from intrinsics and extrinsics
    proj_h_mtx = np.dot(proj_mtx, np.hstack((camproj_rot_mtx, camproj_trans_mtx)))
    #Calculate H matrix for camera
    cam_h_mtx = np.dot(cam_mtx, np.hstack((np.identity(3), np.zeros((3, 1)))))
    if processing == 'gpu':
        uc_updated = cp.asarray(uc_updated)
        vc_updated = cp.asarray(vc_updated)
        up_updated = cp.asarray(up_updated)
        cam_h_mtx = cp.asarray(cam_h_mtx)
        proj_h_mtx = cp.asarray(proj_h_mtx)
    coords = triangulation(uc_updated, vc_updated, up_updated, cam_h_mtx, proj_h_mtx, processing)
    return cp.asnumpy(coords), nan_mask 



def diff_funs_x(hc_11, hc_13, hc_22, hc_23, hc_33, hp_11, hp_12, hp_13, hp_14, hp_31, hp_32, hp_33, hp_34, det, x_num, uc, vc, up):
    """
    Sub function used to calculate x coordinate variance.
    Ref: S.Zhong, High-Speed 3D Imaging with Digital Fringe Projection Techniques, CRC Press, 2016.
    Chapter 7 :Digital Fringe Projection System Calibration, section:7.3.6
    :param hc_11, hc_13, hc_22, hc_23, hc_33: camera H matrix elements
    :param hp_11,hp_12, hp_13, hp_14, hp_31, hp_32, hp_33, hp_34 : projector H matrix elements
    :param det: determinate of inverse matrix to calculate coordinates.
    :param x_num: x_num/det gives the x cordinate.
    :param uc,vc : camera coordinates
    :param up: projector coordinate.
    :type hc_11, hc_13, hc_22, hc_23, hc_33: float
    :type hp_11,hp_12, hp_13, hp_14, hp_31, hp_32, hp_33, hp_34: float
    :type det:float array
    :type x_num:float array
    :type uc,vc: float array
    :type up:float array

    """
    df_dup = (det * (-hc_13 * hc_22 * hp_34 + uc * hc_22 * hc_33 * hp_34) - x_num * (-hc_11 * hc_22 * hp_33 + hc_13 * hc_22 * hp_31 - uc * hc_22 * hc_33 * hp_31 + hc_11 * hc_23 * hp_32 - vc * hc_11 * hc_33 * hp_32))/det**2
    df_dhc_11 = (- x_num * (hc_22 * hp_13 - up * hc_22 * hp_33 - hc_23 * hp_12 + up * hc_23 * hp_32 + vc * hc_33 * hp_12 - vc * up * hc_33 * hp_32))/det**2
    df_dhc_13 = (det * (-up * hc_22 * hp_34 + hc_22 * hp_14) - x_num * (-hc_22 * hp_11 + up * hc_22 * hp_31))/det**2
    df_dhc_22 = (det * (-up * hc_13 * hp_34 + hc_13 * hp_14 + uc * up * hc_33 * hp_34 - uc * hc_33 * hp_14) - x_num * (hc_11 * hp_13 - up * hc_11 * hp_33 - hc_13 * hp_11 + up * hc_13 * hp_31 + uc * hc_33 * hp_11 - uc * up * hc_33 * hp_31))/det**2
    df_dhc_23 = (- x_num * (-hc_11 * hp_12 + up * hc_11 * hp_32))/det**2
    df_dhc_33 = (det * (uc* up * hc_22 * hp_34 - uc * hc_22 * hp_14) - x_num * (uc * hc_22 * hp_11 - uc* up * hc_22 * hp_31 + vc * hc_11 * hp_12 - vc * up * hc_11 * hp_32)) / det**2
    df_dhp_11 = (- x_num *( -hc_13 * hc_22 + uc * hc_22 * hc_33))/det**2
    df_dhp_12 = (- x_num * (-hc_11*hc_23 + vc * hc_11 * hc_33))/det**2
    df_dhp_13 = (- x_num * (hc_11 * hc_22))/det**2
    df_dhp_14 = (det * (hc_13 * hc_22 - uc * hc_22 * hc_33))/det**2
    df_dhp_31 = (- x_num * (up * hc_22 * hc_13 - uc * up * hc_22 * hc_33))/det**2
    df_dhp_32 = (- x_num * (up * hc_11 * hc_23 - vc * up * hc_11 * hc_33))/det**2
    df_dhp_33 = (- x_num * (-up * hc_11 * hc_22))/det**2
    df_dhp_34 = (det * (-up * hc_13 * hc_22 + uc * up * hc_22 * hc_33))/det**2
    
    return df_dup, df_dhc_11, df_dhc_13, df_dhc_22, df_dhc_23, df_dhc_33, df_dhp_11, df_dhp_12, df_dhp_13, df_dhp_14, df_dhp_31, df_dhp_32, df_dhp_33, df_dhp_34

def diff_funs_y(hc_11, hc_13, hc_22, hc_23, hc_33, hp_11, hp_12, hp_13, hp_14, hp_31, hp_32, hp_33, hp_34, det, y_num, uc, vc, up):
    """
    Subfunction used to calculate y cordinate variance

    """
    df_dup = (det * (-hc_11 * hc_23 * hp_34 + vc * hc_11 * hc_33 * hp_34) - y_num * (-hc_11 * hc_22 * hp_33 + hc_13 * hc_22 * hp_31 - uc * hc_22 * hc_33 * hp_31 + hc_11 * hc_23 * hp_32 - vc * hc_11 * hc_33 * hp_32))/det**2
    df_dhc_11 = (det * (-up * hc_23 * hp_34 + hc_23 * hp_14 + vc * up * hc_33 * hp_34 - vc * hc_33 * hp_14) - y_num * (hc_22 * hp_13 - up * hc_22 * hp_33 - hc_23 * hp_12 + up * hc_23 * hp_32 + vc * hc_33 * hp_12 - vc * up * hc_33 * hp_32))/det**2
    df_dhc_13 = (- y_num * (-hc_22 * hp_11 + up * hc_22 * hp_31))/det**2
    df_dhc_22 = (- y_num * (hc_11 * hp_13 - up * hc_11 * hp_33 - hc_13 * hp_11 + up * hc_13 * hp_31 + uc * hc_33 * hp_11 - uc * up * hc_33 * hp_31))/det**2
    df_dhc_23 = (det * (-up * hc_11 * hp_34 + hc_11 * hp_14) - y_num * (-hc_11 * hp_12 + up * hc_11 * hp_32))/det**2
    df_dhc_33 = (det * (vc* up * hc_11 * hp_34 - vc * hc_11 * hp_14) - y_num * (uc * hc_22 * hp_11 - uc * up * hc_22 * hp_31 + vc * hc_11 * hp_12 - vc * up * hc_11 * hp_32)) / det**2
    df_dhp_11 = (- y_num * (-hc_13 * hc_22 + uc * hc_22 * hc_33))/det**2
    df_dhp_12 = (- y_num * (-hc_11 * hc_23 + vc * hc_11 * hc_33))/det**2
    df_dhp_13 = (- y_num * (hc_11 * hc_22))/det**2
    df_dhp_14 = (det * (hc_11 * hc_23 - vc * hc_11 * hc_33))/det**2
    df_dhp_31 = (- y_num * (up * hc_13 * hc_22 - uc * up * hc_22 * hc_33))/det**2
    df_dhp_32 = (- y_num * (up * hc_11 * hc_23 - vc * up * hc_11 * hc_33))/det**2
    df_dhp_33 = (- y_num * (-up * hc_11 * hc_22))/det**2
    df_dhp_34 = (det * (-up * hc_11 * hc_23 + vc * up * hc_11 * hc_33))/det**2
    
    return df_dup, df_dhc_11, df_dhc_13, df_dhc_22, df_dhc_23, df_dhc_33, df_dhp_11, df_dhp_12, df_dhp_13, df_dhp_14, df_dhp_31, df_dhp_32, df_dhp_33, df_dhp_34

def diff_funs_z(hc_11, hc_13, hc_22, hc_23, hc_33, hp_11, hp_12, hp_13, hp_14, hp_31, hp_32, hp_33, hp_34, det, z_num, uc, vc, up):
    """
    Subfunction used to calculate z cordinate variance

    """
    df_dup = (det * (hc_11 * hc_22 * hp_34) - z_num * (-hc_11 * hc_22 * hp_33 + hc_22 * hc_13 * hp_31 - uc * hc_22 * hc_33 * hp_31 + hc_11 * hc_23 * hp_32 - vc * hc_11 * hc_33 * hp_32))/det**2
    df_dhc_11 = (det * (up * hc_22 * hp_34 - hc_22 * hp_14) - z_num * (hc_22 * hp_13 - up * hc_22 * hp_33 - hc_23 * hp_12 + up * hc_23 * hp_32 + vc * hc_33 * hp_12 - vc * up * hc_33 * hp_32))/det**2
    df_dhc_13 = (- z_num * (-hc_22 * hp_11 + up * hc_22 * hp_31))/det**2
    df_dhc_22 = (det * (up * hc_11 * hp_34 - hc_11 * hp_14) - z_num * (hc_11 * hp_13 - up * hc_11 * hp_33 - hc_13 * hp_11 + up * hc_13 * hp_31 + uc * hc_33 * hp_11 - uc * up * hc_33 * hp_31))/det**2
    df_dhc_23 = (- z_num * (-hc_11 * hp_12 + up * hc_11 * hp_32))/det**2
    df_dhc_33 = (- z_num * (uc * hc_22 * hp_11 - uc * up * hc_22 * hp_31 + vc * hc_11 * hp_12 - vc * up * hc_11 * hp_32))/det**2
    df_dhp_11 = (- z_num * (-hc_13 * hc_22 + uc * hc_22 * hc_33))/det**2
    df_dhp_12 = (- z_num * (-hc_11 * hc_23 + vc * hc_11 * hc_33))/det**2
    df_dhp_13 = (- z_num * (hc_11 * hc_22))/det**2
    df_dhp_14 = (det * (-hc_11 * hc_22))/det**2
    df_dhp_31 = (- z_num * (up * hc_22 * hc_13 - uc * up * hc_22 * hc_33))/det**2
    df_dhp_32 = (- z_num * (up * hc_11 * hc_23 - vc * up * hc_11 * hc_33))/det**2
    df_dhp_33 = (- z_num * (-up * hc_11 * hc_22))/det**2
    df_dhp_34 = (det * (up * hc_11 * hc_22))/det**2
    
    return df_dup, df_dhc_11, df_dhc_13, df_dhc_22, df_dhc_23, df_dhc_33, df_dhp_11, df_dhp_12, df_dhp_13, df_dhp_14, df_dhp_31, df_dhp_32, df_dhp_33, df_dhp_34

def sigma_random(modulation, limit,  pitch, N, phase_st, unwrap, sigma_path, source_folder):
    '''
    Function to calculate variance of x,y,z coordinates

    '''
    sigma = np.load(sigma_path)
    mean_calibration_param = np.load(os.path.join(source_folder, 'mean_calibration_param.npz'))
    h_matrix_param = np.load(os.path.join(source_folder, 'h_matrix_param.npz'))
    cam_mtx = mean_calibration_param["arr_0"]
    cam_dist = mean_calibration_param["arr_2"]
    proj_h_mtx_mean = h_matrix_param["arr_2"]
    cam_h_mtx_mean = h_matrix_param["arr_0"]
    proj_h_mtx_std = h_matrix_param["arr_3"]
    cam_h_mtx_std = h_matrix_param["arr_1"]
    
    
    mod_copy = deepcopy(modulation)
    unwrap_copy = deepcopy(unwrap)
    roi_mask = np.full(mod_copy.shape, False)
    roi_mask[mod_copy > limit] = True
    mod_copy[~roi_mask] = np.nan
    unwrap_copy[~roi_mask] = np.nan
    unwrap_dist = cv2.undistort(unwrap, cam_mtx, cam_dist)
    u = np.arange(0, unwrap_dist.shape[1])
    v = np.arange(0, unwrap_dist.shape[0])
    uc, vc = np.meshgrid(u, v)
    up = (unwrap_dist - phase_st) * pitch / (2*np.pi)
    sigma_sq_phi = (2 * sigma**2) / (N * mod_copy**2)
    sigma_sq_up = sigma_sq_phi * pitch**2 / 4 * np.pi**2
    
    hc_11 = cam_h_mtx_mean[0, 0]
    sigmasq_hc_11 = cam_h_mtx_std[0, 0]**2
    hc_13 = cam_h_mtx_mean[0, 2]
    sigmasq_hc_13 = cam_h_mtx_std[0, 2]**2
    
    hc_22 = cam_h_mtx_mean[1, 1]
    sigmasq_hc_22 = cam_h_mtx_std[1, 1]**2
    hc_23 = cam_h_mtx_mean[1, 2]
    sigmasq_hc_23 = cam_h_mtx_std[1, 2]**2
    
    hc_33 = cam_h_mtx_mean[2, 2]
    sigmasq_hc_33 = cam_h_mtx_std[2, 2]**2
    
    hp_11 = proj_h_mtx_mean[0, 0]
    sigmasq_hp_11 = proj_h_mtx_std[0, 0]**2
    hp_12 = proj_h_mtx_mean[0, 1]
    sigmasq_hp_12 = proj_h_mtx_std[0, 1]**2
    hp_13 = proj_h_mtx_mean[0, 2]
    sigmasq_hp_13 = proj_h_mtx_std[0, 2]**2
    hp_14 = proj_h_mtx_mean[0, 3]
    sigmasq_hp_14 = proj_h_mtx_std[0, 3]**2
    
    hp_31 = proj_h_mtx_mean[2, 0]
    sigmasq_hp_31 = proj_h_mtx_std[2, 0]**2
    hp_32 = proj_h_mtx_mean[2, 1]
    sigmasq_hp_32 = proj_h_mtx_std[2, 1]**2
    hp_33 = proj_h_mtx_mean[2, 2]
    sigmasq_hp_33 = proj_h_mtx_std[2, 2]**2
    hp_34 = proj_h_mtx_mean[2, 3]
    sigmasq_hp_34 = proj_h_mtx_std[2, 3]**2
    
    det = (hc_11 * hc_22 * hp_13 - up * hc_11 * hc_22 * hp_33 - hc_13 * hc_22 * hp_11 + up * hc_13 *hc_22 * hp_31 + uc * hc_22 * hc_33 * hp_11
           - uc * up * hc_22 * hc_33 * hp_31 - hc_11 * hc_23 * hp_12 + up * hc_11 * hc_23 * hp_32 + vc * hc_11 * hc_33 * hp_12 - vc * up * hc_11 * hc_33 * hp_32)
           
    
    x_num = -up * hc_13 * hc_22 * hp_34 + hc_13 * hc_22 * hp_14 + uc * up * hc_22 * hc_33 * hp_34 - uc * hc_22 * hc_33 * hp_14
    df_dup_x, df_dhc_11_x, df_dhc_13_x, df_dhc_22_x, df_dhc_23_x, df_dhc_33_x, df_dhp_11_x, df_dhp_12_x, df_dhp_13_x, df_dhp_14_x, df_dhp_31_x, df_dhp_32_x, df_dhp_33_x, df_dhp_34_x = diff_funs_x(hc_11, hc_13, hc_22, hc_23, hc_33, hp_11, hp_12, hp_13, hp_14, hp_31, hp_32, hp_33, hp_34, det, x_num, uc, vc, up)
    sigmasq_x = ((df_dup_x**2 * sigma_sq_up) + (df_dhc_11_x**2 * sigmasq_hc_11) + (df_dhc_13_x**2 * sigmasq_hc_13) + (df_dhc_22_x**2 * sigmasq_hc_22) + (df_dhc_23_x**2 * sigmasq_hc_23) + (df_dhc_33_x**2 * sigmasq_hc_33)
                + (df_dhp_11_x**2 * sigmasq_hp_11) + (df_dhp_12_x**2 * sigmasq_hp_12) + (df_dhp_13_x**2 * sigmasq_hp_13) + (df_dhp_14_x**2 * sigmasq_hp_14) + (df_dhp_31_x**2 * sigmasq_hp_31) + (df_dhp_32_x**2 * sigmasq_hp_32) + (df_dhp_33_x**2 * sigmasq_hp_33) + (df_dhp_34_x**2 * sigmasq_hp_34))
    sigmasq_x[~roi_mask] = np.nan
    derv_x = np.stack((df_dup_x, df_dhc_11_x, df_dhc_13_x, df_dhc_22_x, df_dhc_23_x, df_dhc_33_x, df_dhp_11_x, df_dhp_12_x, df_dhp_13_x, df_dhp_14_x, df_dhp_31_x, df_dhp_32_x, df_dhp_33_x, df_dhp_34_x))
    
    y_num = -up * hc_11 * hc_23 * hp_34 + hc_11 * hc_23 * hp_14 + vc * up * hc_11 * hc_33 * hp_34 - vc * hc_11 * hc_33 * hp_14
    #y_num = (-hc_11 * (hc_23 - hc_33)*(up * hp_34 - hp_14))
    df_dup_y, df_dhc_11_y, df_dhc_13_y, df_dhc_22_y, df_dhc_23_y, df_dhc_33_y, df_dhp_11_y, df_dhp_12_y, df_dhp_13_y, df_dhp_14_y, df_dhp_31_y, df_dhp_32_y, df_dhp_33_y, df_dhp_34_y = diff_funs_y(hc_11, hc_13, hc_22, hc_23, hc_33, hp_11, hp_12, hp_13, hp_14, hp_31, hp_32, hp_33, hp_34, det, y_num, uc, vc, up)
    sigmasq_y = ((df_dup_y**2 * sigma_sq_up) + (df_dhc_11_y**2 * sigmasq_hc_11) + (df_dhc_13_y**2 * sigmasq_hc_13) + (df_dhc_22_y**2 * sigmasq_hc_22) + (df_dhc_23_y**2 * sigmasq_hc_23) + (df_dhc_33_y**2 * sigmasq_hc_33)
                + (df_dhp_11_y**2 * sigmasq_hp_11) + (df_dhp_12_y**2 * sigmasq_hp_12) + (df_dhp_13_y**2 * sigmasq_hp_13) + (df_dhp_14_y**2 * sigmasq_hp_14) + (df_dhp_31_y**2 * sigmasq_hp_31) + (df_dhp_32_y**2 * sigmasq_hp_32) + (df_dhp_33_y**2 * sigmasq_hp_33) + (df_dhp_34_y**2 * sigmasq_hp_34))
    sigmasq_y[~roi_mask] = np.nan
    derv_y = np.stack((df_dup_y, df_dhc_11_y, df_dhc_13_y, df_dhc_22_y, df_dhc_23_y, df_dhc_33_y, df_dhp_11_y, df_dhp_12_y, df_dhp_13_y, df_dhp_14_y, df_dhp_31_y, df_dhp_32_y, df_dhp_33_y, df_dhp_34_y))
    
    z_num = up * hc_11 * hc_22 * hp_34 - hc_11 * hc_22 * hp_14 
    df_dup_z, df_dhc_11_z, df_dhc_13_z, df_dhc_22_z, df_dhc_23_z, df_dhc_33_z, df_dhp_11_z, df_dhp_12_z, df_dhp_13_z, df_dhp_14_z, df_dhp_31_z, df_dhp_32_z, df_dhp_33_z, df_dhp_34_z = diff_funs_z(hc_11, hc_13, hc_22, hc_23, hc_33, hp_11, hp_12, hp_13, hp_14, hp_31, hp_32, hp_33, hp_34, det, z_num, uc, vc, up)
    sigmasq_z = ((df_dup_z**2 * sigma_sq_up) + (df_dhc_11_z**2 * sigmasq_hc_11) + (df_dhc_13_z**2 * sigmasq_hc_13) + (df_dhc_22_z**2 * sigmasq_hc_22) + (df_dhc_23_z**2 * sigmasq_hc_23) + (df_dhc_33_z**2 * sigmasq_hc_33)
                + (df_dhp_11_z**2 * sigmasq_hp_11) + (df_dhp_12_z**2 * sigmasq_hp_12) + (df_dhp_13_z**2 * sigmasq_hp_13) + (df_dhp_14_z**2 * sigmasq_hp_14) + (df_dhp_31_z**2 * sigmasq_hp_31) + (df_dhp_32_z**2 * sigmasq_hp_32) + (df_dhp_33_z**2 * sigmasq_hp_33) + (df_dhp_34_z**2 * sigmasq_hp_34))
    sigmasq_z[~roi_mask] = np.nan
    derv_z = np.stack((df_dup_z, df_dhc_11_z, df_dhc_13_z, df_dhc_22_z, df_dhc_23_z, df_dhc_33_z, df_dhp_11_z, df_dhp_12_z, df_dhp_13_z, df_dhp_14_z, df_dhp_31_z, df_dhp_32_z, df_dhp_33_z, df_dhp_34_z))
    
    return sigmasq_x, sigmasq_y, sigmasq_z, derv_x, derv_y, derv_z

def complete_recon(unwrap, inte_rgb, modulation, limit, calib_path, sigma_path, phase_st, pitch, N, obj_path, temp, processing, temperature=None):
    '''
    Function to completely reconstruct object applying modulation mask to saving point cloud.

    Parameters
    ----------
    unwrap = type: float array. Unwrapped phase map of object.
    inte_rgb = type: float array. Texture image.
    modulation = type: float array. Intensity modulation image.
    limit = type: float. Intensity modulation limit for mask.
    cam_mtx = type: float array. Camera matrix from calibration.
    cam_dist = type: float array. Camera distortion matrix from calibration.
    proj_mtx = type: float array. Projector matrix from calibration.
    camproj_rot_mtx = type: float array. Projector distortion matrix from calibration.
    camproj_trans_mtx = type: float array. Camera-projector translational matrix from calibration.
    phase_st = type: float. Initial phase to be subtracted for phase to coordinate conversion.
    pitch  = type:float. Number of pixels per fringe period.
    N = type: int. No. of images
    obj_path = type: string. Path to save point 3D reconstructed point cloud. 
    temp = type: bool. True if temperature information is available else False
    temperature = type:Array of floats. Temperature values for each pixel

    Returns
    -------
    cordi = type:  float array. x,y,z coordinate array of each object point.
    intensity = type: float array. Intensity (texture/ color) at each point.

    '''
    calibration = np.load(os.path.join(calib_path, 'mean_calibration_param.npz'))
    cam_mtx = calibration["arr_0"]
    cam_dist = calibration["arr_2"]
    proj_mtx = calibration["arr_4"]
    camproj_rot_mtx = calibration["arr_8"]
    camproj_trans_mtx = calibration["arr_10"]
    u_copy = deepcopy(unwrap)
    w_copy = deepcopy(inte_rgb)
    mod = deepcopy(modulation)
    unwrap_dist = cv2.undistort(u_copy, cam_mtx, cam_dist)
    u = np.arange(0, unwrap_dist.shape[1])
    v = np.arange(0, unwrap_dist.shape[0])
    uc, vc = np.meshgrid(u, v)
    up = (unwrap_dist - phase_st) * pitch / (2*np.pi) 
    uc = uc.ravel()
    vc = vc.ravel()
    up = up.ravel()
    nan_mask = np.isnan(up)
    uc_updated = uc[~nan_mask]
    vc_updated = vc[~nan_mask]
    up_updated = up[~nan_mask]
    # Calculate H matrix for proj from intrinsics and extrinsics
    proj_h_mtx = np.dot(proj_mtx, np.hstack((camproj_rot_mtx, camproj_trans_mtx)))
    #Calculate H matrix for camera
    cam_h_mtx = np.dot(cam_mtx, np.hstack((np.identity(3), np.zeros((3, 1)))))
    if processing == 'gpu':
        uc_updated = cp.asarray(uc_updated)
        vc_updated = cp.asarray(vc_updated)
        up_updated = cp.asarray(up_updated)
        cam_h_mtx = cp.asarray(cam_h_mtx)
        proj_h_mtx = cp.asarray(proj_h_mtx)
    cordi = triangulation(uc_updated, vc_updated, up_updated, cam_h_mtx, proj_h_mtx, processing)
    xyz = list(map(tuple, cordi)) 
    inte_img = (w_copy / np.nanmax(w_copy)).ravel()
    inte_img = inte_img[~nan_mask]
    inte_rgb = np.stack((inte_img, inte_img, inte_img), axis=-1)
    color = list(map(tuple, inte_rgb))
    sigmasq_x, sigmasq_y, sigmasq_z, derv_x, derv_y, derv_z = sigma_random(modulation, limit,  pitch, N, phase_st, unwrap, sigma_path, calib_path)
    sigma_x = np.sqrt(sigmasq_x)
    sigma_y = np.sqrt(sigmasq_y)
    sigma_z = np.sqrt(sigmasq_z)
    cordi_sigma = np.vstack((sigma_x.ravel(), sigma_y.ravel(), sigma_z.ravel())).T
    up_cordi_sigma = cordi_sigma[~nan_mask]
    xyz_sigma = list(map(tuple, up_cordi_sigma))
    mod[~nan_mask] = np.nan
    mod_vect = np.array(mod.ravel(), dtype=[('modulation', 'f4')])
    if temp:
        t_vect = np.array(temperature.ravel(), dtype=[('temperature', 'f4')])
        up_t_vect = t_vect[~nan_mask]
        PlyData(
            [
                PlyElement.describe(np.array(xyz, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'points'),
                PlyElement.describe(np.array(color, dtype=[('r', 'f4'), ('g', 'f4'), ('b', 'f4')]), 'color'),
                PlyElement.describe(np.array(xyz_sigma, dtype=[('dx', 'f4'), ('dy', 'f4'), ('dz', 'f4')]), 'std'),
                PlyElement.describe(np.array(up_t_vect, dtype=[('temperature', 'f4')]), 'temperature'),
                PlyElement.describe(np.array(mod_vect, dtype=[('modulation', 'f4')]), 'modulation')
                
            ]).write(os.path.join(obj_path, 'obj.ply'))
    else:
        t_vect = None
        PlyData(
            [
                PlyElement.describe(np.array(xyz, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'points'),
                PlyElement.describe(np.array(color, dtype=[('r', 'f4'), ('g', 'f4'), ('b', 'f4')]), 'color'),
                PlyElement.describe(np.array(xyz_sigma, dtype=[('dx', 'f4'), ('dy', 'f4'), ('dz', 'f4')]), 'std'),
                PlyElement.describe(np.array(mod_vect, dtype=[('modulation', 'f4')]), 'modulation')
                
            ]).write(os.path.join(obj_path, 'obj.ply'))
      
    return cordi, inte_rgb, t_vect, cordi_sigma, mod_vect

def obj_reconst_wrapper(width, 
                        height,
                        pitch_list, 
                        N_list,
                        limit,
                        phase_st,
                        direc,
                        type_unwrap, 
                        calib_path, 
                        obj_path, 
                        sigma_path, 
                        temp, 
                        data_type,
                        processing,
                        kernel=1):
    """
    Function for 3D reconstruction of object based on different unwrapping method.

    Parameters
    ----------
    width =type: float. Width of projector.
    height = type: float. Height of projector.
    cam_width,
    cam_height
    pitch_list :
    N_list = type: float array. The number of steps in phase shifting algorithm. If phase coded unwrapping method is used this is a single element array. For other methods corresponding to each pitch one element in the list.
    limit = type: float array. Array of number of pixels per fringe period.
    
    phase_st = type: float. Initial phase to be subtracted for phase to coordinate conversion.
    direc = type: string. Visually vertical (v) or horizontal(h) pattern.
    type_unwrap = type: string. Type of temporal unwrapping to be applied. 
                  'phase' = phase coded unwrapping method, 
                  'multifreq' = multi frequency unwrapping method
                  'multiwave' = multi wavelength unwrapping method.
    calib_path = type: string. Path to read mean calibration parameters.
    obj_path = type: string. Path to read captured images
    kernel = type: int. Kernel size for median filter. The default is 1.

    Returns
    -------
    obj_cordi = type : float array. Array of reconstructed x,y,z coordinates of each points on the object
    obj_color = type: float array. Color (texture/ intensity) at each point.

    """
   
    if data_type == 'jpeg':
        img_path = sorted(glob.glob(os.path.join(obj_path, 'capt_*.jpg')), key=os.path.getmtime)
        images_arr = [cv2.imread(file, 0) for file in img_path]
        if processing == 'cpu':
            images_arr = np.array(images_arr).astype(np.float64)
        else:
            images_arr = cp.asarray(images_arr).astype(cp.float64)
    elif(data_type == 'npy') & (processing == 'cpu'):
        images_arr = np.load(os.path.join(obj_path, 'capt_*.npy'))
    elif(data_type == 'npy') & (processing == 'gpu'):
        images_arr = cp.load(os.path.join(obj_path, 'capt_*.npy'))

    if type_unwrap == 'phase':
      if processing == 'cpu':
          obj_cos_mod, obj_cos_avg, phase_cos = nstep.phase_cal(images_arr[0:N_list[0]],
                                                                limit)
          obj_step_mod, obj_step_avg, phase_step = nstep.phase_cal(images_arr[N_list[0]:2*N_list[0]],
                                                                   limit)
          phase_step = nstep.step_rectification(phase_step, direc)
          #unwrapped phase
          unwrap0, k0 = nstep.unwrap_cal(phase_step, phase_cos, pitch_list[0], width, height, direc)
          unwrap, k = nstep.filt(unwrap0, kernel, direc)
          mod_freq4 = obj_cos_mod
       
    elif type_unwrap == 'multifreq':
        if processing == 'cpu':
            mod_freq1, avg_freq1, phase_freq1 = nstep.phase_cal(images_arr[0: N_list[0]],
                                                                limit)
            mod_freq2, avg_freq2, phase_freq2 = nstep.phase_cal(images_arr[N_list[0]: N_list[0] + N_list[1]],
                                                                limit)
            mod_freq3, avg_freq3, phase_freq3 = nstep.phase_cal(images_arr[N_list[0] + N_list[1]: N_list[0]+ N_list[1]+ N_list[2]],
                                                                limit)
            mod_freq4, avg_freq4, phase_freq4 = nstep.phase_cal(images_arr[N_list[0]+ N_list[1]+ N_list[2]: N_list[0]+ N_list[1]+ N_list[2] + N_list[3]],
                                                                limit)
            phase_freq1[phase_freq1 < EPSILON] = phase_freq1[phase_freq1 < EPSILON] + 2 * np.pi
            #unwrapped phase
            phase_arr = np.stack([phase_freq1, phase_freq2, phase_freq3, phase_freq4])
            unwrap, k = nstep.multifreq_unwrap(pitch_list, phase_arr, kernel, direc)
           
        elif processing == 'gpu':
            mod_freq1, avg_freq1, phase_freq1 = nstep_cp.phase_cal_cp(images_arr[0: N_list[0]],
                                                                      limit)
            mod_freq2, avg_freq2, phase_freq2 = nstep_cp.phase_cal_cp(images_arr[N_list[0]: N_list[0] + N_list[1]],
                                                                      limit)
            mod_freq3, avg_freq3, phase_freq3 = nstep_cp.phase_cal_cp(images_arr[N_list[0] + N_list[1]: N_list[0] + N_list[1] + N_list[2]],
                                                                      limit)
            mod_freq4, avg_freq4, phase_freq4 = nstep_cp.phase_cal_cp(images_arr[N_list[0] + N_list[1] + N_list[2]: N_list[0] + N_list[1] + N_list[2] + N_list[3]],
                                                                      limit)
            phase_freq1[phase_freq1 < EPSILON] = phase_freq1[phase_freq1 < EPSILON] + 2 * np.pi
            # unwrapped phase
            phase_arr = np.stack([phase_freq1, phase_freq2, phase_freq3, phase_freq4])
            unwrap, k = nstep_cp.multifreq_unwrap(pitch_list, phase_arr, kernel, direc)
            unwrap = cp.asnumpy(unwrap)
            mod_freq4 = cp.asnumpy(mod_freq4)
       
    elif type_unwrap == 'multiwave':
        eq_wav12 = (pitch_list[-1] * pitch_list[1]) / (pitch_list[1]-pitch_list[-1])
        eq_wav123 = pitch_list[0] * eq_wav12 / (pitch_list[0] - eq_wav12)
        pitch_list = np.insert(pitch_list, 0, eq_wav123)
        pitch_list = np.insert(pitch_list, 2, eq_wav12)
        mod_wav3, avg_wav3, phase_wav1 = nstep.phase_cal(images_arr[0, N_list[0]],
                                                         limit)
        mod_wav2, avg_wav2, phase_wav2 = nstep.phase_cal(images_arr[N_list[0], N_list[0] + N_list[1]],
                                                         limit)
        mod_wav1, avg_wav1, phase_wav3 = nstep.phase_cal(images_arr[N_list[0] + N_list[1], N_list[0]+ N_list[1]+ N_list[2]],
                                                         limit)
        phase_wav12 = np.mod(phase_wav1 - phase_wav2, 2 * np.pi)
        phase_wav123 = np.mod(phase_wav12 - phase_wav3, 2 * np.pi)
        phase_wav123[phase_wav123 > TAU] = phase_wav123[phase_wav123 > TAU] - 2 * np.pi
        #unwrapped phase
        phase_arr = np.stack([phase_wav123, phase_wav3, phase_wav12, phase_wav2, phase_wav1])
        unwrap, k = nstep.multiwave_unwrap(pitch_list, phase_arr, kernel, direc)
        mod_freq4 = mod_wav3
    inte_img = cv2.imread(os.path.join(obj_path, 'white.jpg'))
    if temp:
        temperature = np.load(os.path.join(obj_path, 'temperature.npy'))
    else:
        temperature = 0
    inte_rgb = inte_img[..., ::-1].copy()
    np.save(os.path.join(obj_path, '{}_obj_mod.npy'.format(type_unwrap)), mod_freq4)
    np.save(os.path.join(obj_path, '{}_unwrap.npy'.format(type_unwrap)), unwrap)
    obj_cordi, obj_color, obj_t, cordi_sigma, mod_vect = complete_recon(unwrap, 
                                                                        inte_rgb, 
                                                                        mod_freq4, 
                                                                        limit, 
                                                                        calib_path, 
                                                                        sigma_path, 
                                                                        phase_st, 
                                                                        pitch_list[-1], 
                                                                        N_list[-1], 
                                                                        obj_path, 
                                                                        temp, 
                                                                        temperature)
    return obj_cordi, obj_color, obj_t, cordi_sigma, mod_vect
       
def obj_reconst_wrapper_3level(width, 
                               height,
                               pitch_list, 
                               N_list,
                               limit,
                               phase_st,
                               direc,
                               type_unwrap, 
                               calib_path, 
                               obj_path, 
                               sigma_path, 
                               temp, 
                               data_type,
                               processing,
                               kernel=1):
    """
    Function for 3D reconstruction of object based on different unwrapping method.

    Parameters
    ----------
    width =type: float. Width of projector.
    height = type: float. Height of projector.
    pitch_list = type: list.
    N_list = type: float array. The number of steps in phase shifting algorithm. If phase coded unwrapping method is used this is a single element array. For other methods corresponding to each pitch one element in the list.
    limit = type: float array. Array of number of pixels per fringe period.
    
    phase_st = type: float. Initial phase to be subtracted for phase to coordinate conversion.
    direc = type: string. Visually vertical (v) or horizontal(h) pattern.
    type_unwrap = type: string. Type of temporal unwrapping to be applied. 
                  'phase' = phase coded unwrapping method, 
                  'multifreq' = multifrequency unwrapping method
                  'multiwave' = multiwavelength unwrapping method.
    calib_path = type: string. Path to read mean calibration paraneters. 
    obj_path = type: string. Path to read captured images
    kernel = type: int. Kernel size for median filter. The default is 1.

    Returns
    -------
    obj_cordi = type : float array. Array of reconstructed x,y,z coordinates of each points on the object
    obj_color = type: float array. Color (texture/ intensity) at each point.

    """
    if data_type == 'jpeg':
        img_path = sorted(glob.glob(os.path.join(obj_path, 'capt_*.jpg')), key=os.path.getmtime)
        images_arr = [cv2.imread(file, 0) for file in img_path]
        if processing == 'cpu':
            images_arr = np.array(images_arr).astype(np.float64)
        else:
            images_arr = cp.asarray(images_arr).astype(cp.float64)
    elif (data_type == 'npy') & (processing == 'cpu'):
        images_arr = np.load(os.path.join(obj_path, 'capt_*.npy'))
    elif (data_type == 'npy') & (processing == 'gpu'):
        images_arr = cp.load(os.path.join(obj_path, 'capt_*.npy'))
        
    if type_unwrap == 'phase':
       
        obj_cos_mod, obj_cos_avg, phase_cos = nstep.phase_cal(images_arr[0, N_list[0]],
                                                              limit)
        obj_step_mod, obj_step_avg, phase_step = nstep.phase_cal(images_arr[N_list[0], 2 * N_list[0]],
                                                                limit)

        #wrapped phase
        phase_step = nstep.step_rectification(phase_step, direc)
        #unwrapped phase
        unwrap0, k0 = nstep.unwrap_cal(phase_step, phase_cos, pitch_list[0], width, height, direc)
        unwrap, k = nstep.filt(unwrap0, kernel, direc)
        mod_freq3 = obj_cos_mod
       
    elif type_unwrap == 'multifreq':
        if processing == 'cpu':
            mod_freq1, avg_freq1, phase_freq1 = nstep.phase_cal(images_arr[0: N_list[0]],
                                                                limit)
            mod_freq2, avg_freq2, phase_freq2 = nstep.phase_cal(images_arr[N_list[0]: N_list[0] + N_list[1]],
                                                                limit)
            mod_freq3, avg_freq3, phase_freq3 = nstep.phase_cal(images_arr[N_list[0] + N_list[1]: N_list[0]+ N_list[1]+ N_list[2]],
                                                                limit)
            #wrapped phase
            phase_freq1[phase_freq1 < EPSILON] = phase_freq1[phase_freq1 < EPSILON] + 2 * np.pi
            #unwrapped phase
            phase_arr = np.stack([phase_freq1, phase_freq2, phase_freq3])
            unwrap, k = nstep.multifreq_unwrap(pitch_list, phase_arr, kernel, direc)
        elif processing == 'gpu':
            mod_freq1, avg_freq1, phase_freq1 = nstep_cp.phase_cal_cp(images_arr[0: N_list[0]],
                                                                      limit)
            mod_freq2, avg_freq2, phase_freq2 = nstep_cp.phase_cal_cp(images_arr[N_list[0]: N_list[0] + N_list[1]],
                                                                      limit)
            mod_freq3, avg_freq3, phase_freq3 = nstep_cp.phase_cal_cp(images_arr[N_list[0] + N_list[1]: N_list[0]+ N_list[1]+ N_list[2]],
                                                                      limit)
            #wrapped phase
            phase_freq1[phase_freq1 < EPSILON] = phase_freq1[phase_freq1 < EPSILON] + 2 * np.pi
            #unwrapped phase
            phase_arr = [phase_freq1, phase_freq2, phase_freq3]
            unwrap, k = nstep_cp.multifreq_unwrap_cp(pitch_list, phase_arr, kernel, direc)
            unwrap = cp.asnumpy(unwrap)
            mod_freq3 = cp.asnumpy(mod_freq3)
       
    elif type_unwrap == 'multiwave':
        eq_wav12 = (pitch_list[-1] * pitch_list[1]) / (pitch_list[1]-pitch_list[-1])
        eq_wav123 = pitch_list[0] * eq_wav12 / (pitch_list[0] - eq_wav12)
        pitch_list = np.insert(pitch_list, 0, eq_wav123)
        pitch_list = np.insert(pitch_list, 2, eq_wav12)
        mod_wav3, avg_wav3, phase_wav1 = nstep.phase_cal(images_arr[0: N_list[0]],
                                                         limit)
        mod_wav2, avg_wav2, phase_wav2 = nstep.phase_cal(images_arr[N_list[0]: N_list[0] + N_list[1]],
                                                         limit)
        mod_wav1, avg_wav1, phase_wav3 = nstep.phase_cal(images_arr[N_list[0] + N_list[1]: N_list[0]+ N_list[1]+ N_list[2]],
                                                         limit)
        phase_wav12 = np.mod(phase_wav1 - phase_wav2, 2 * np.pi)
        phase_wav123 = np.mod(phase_wav12 - phase_wav3, 2 * np.pi)
        phase_wav123[phase_wav123 > TAU] = phase_wav123[phase_wav123 > TAU] - 2 * np.pi
        #unwrapped phase
        phase_arr = np.stack([phase_wav123, phase_wav3, phase_wav12, phase_wav2, phase_wav1])
        unwrap, k = nstep.multiwave_unwrap(pitch_list, phase_arr, kernel, direc)
        mod_freq3 = mod_wav3
       
    inte_img = cv2.imread(os.path.join(obj_path, 'white.jpg'))
    if temp:
        temperature = np.load(os.path.join(obj_path, 'temperature.npy'))
    else:
        temperature = 0
    inte_rgb = inte_img[..., ::-1].copy()
    np.save(os.path.join(obj_path, '{}_obj_mod.npy'.format(type_unwrap)), mod_freq3)
    np.save(os.path.join(obj_path, '{}_unwrap.npy'.format(type_unwrap)), unwrap)
    obj_cordi, obj_color, obj_t, cordi_sigma, mod_vect = complete_recon(unwrap, 
                                                                        inte_rgb, 
                                                                        mod_freq3, 
                                                                        limit, 
                                                                        calib_path, 
                                                                        sigma_path, 
                                                                        phase_st, pitch_list[-1], 
                                                                        N_list[-1], 
                                                                        obj_path, 
                                                                        temp, temperature)
    return obj_cordi, obj_color, obj_t, cordi_sigma, mod_vect
       
