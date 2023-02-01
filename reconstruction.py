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
#TODO: Move out ply saving. 
#TODO: Convert to pyqtgraph. 

class Reconstruction:
    """
    Reconstruction class is used for complete reconstruction of 3D object with per coordinate uncertainity. 
    If temperature is available it also integrated into the point cloud.
    """
    def __init__(self,
                 proj_width,
                 proj_height,
                 cam_width,
                 cam_height,
                 type_unwrap,
                 limit,
                 N_list,
                 pitch_list,
                 fringe_direc,
                 kernel,
                 data_type,
                 processing,
                 calib_path,
                 sigma_path,
                 object_path,
                 temp,
                 bootstrap):
        self.proj_width = proj_width
        self.proj_height = proj_height
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.type_unwrap = type_unwrap
        self.limit = limit
        self.N_list = N_list
        self.pitch_list = pitch_list
        self.fringe_direc = fringe_direc
        self.kernel = kernel
        self.calib_path = calib_path
        self.object_path = object_path
        self.temp = temp
        
        if (self.type_unwrap == 'multifreq') or (self.type_unwrap == 'multiwave'):
            self.phase_st = 0
        else:
            print('ERROR: Invalid type_unwrap')
            return
        if not os.path.exists(self.calib_path):
            print('ERROR:calibration parameter path  %s does not exist' % self.calib_path)
        if not os.path.exists(sigma_path):
            print('ERROR:Path for noise error  %s does not exist' % self.calib_path)
        else:
            self.sigma_path = sigma_path
        if not object_path:
            self.object_path = calib_path
        if not os.path.exists(object_path):
            print('ERROR:Path for noise error  %s does not exist' % self.calib_path)
        else:
            self.object_path = object_path
    
        if self.processing == 'cpu':
            calibration = np.load(os.path.join(self.calib_path, 'mean_calibration_param.npz'))
            self.cam_mtx = calibration["arr_0"]
            self.cam_dist = calibration["arr_2"]
            self.proj_mtx = calibration["arr_4"]
            self.camproj_rot_mtx = calibration["arr_8"]
            self.camproj_trans_mtx = calibration["arr_10"]
            h_matrix_param = np.load(os.path.join(self.calib_path, 'h_matrix_param.npz'))
            self.proj_h_mtx = h_matrix_param["arr_2"]
            self.cam_h_mtx = h_matrix_param["arr_0"]
            self.proj_h_mtx_std = h_matrix_param["arr_3"]
            self.cam_h_mtx_std = h_matrix_param["arr_1"]
            self.sigma = np.load(self.sigma_path)
        elif self.processing == 'gpu': 
            calibration = cp.load(os.path.join(self.calib_path, 'mean_calibration_param.npz'))
            self.cam_mtx = cp.asarray(calibration["arr_0"])
            self.cam_dist = cp.asarray(calibration["arr_2"])
            self.proj_mtx = cp.asarray(calibration["arr_4"])
            self.camproj_rot_mtx = cp.asarray(calibration["arr_8"])
            self.camproj_trans_mtx = cp.asarray(calibration["arr_10"])
            h_matrix_param = cp.load(os.path.join(self.calib_path, 'h_matrix_param.npz'))
            self.proj_h_mtx = cp.asarray(h_matrix_param["arr_2"])
            self.cam_h_mtx = cp.asarray(h_matrix_param["arr_0"])
            self.proj_h_mtx_std = cp.asarray(h_matrix_param["arr_3"])
            self.cam_h_mtx_std = cp.asarray(h_matrix_param["arr_1"])
            self.sigma = cp.load(self.sigma_path)
        else: 
            print("ERROR: Invalid processing type.")
            
    def triangulation(self, uc, vc, up):
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
        if self.processing == 'gpu':
            A = cp.empty((n_pixels, 3, 3))
            c = cp.empty((n_pixels, 3, 1))
        else:
            A = np.empty((n_pixels, 3, 3))
            c = np.empty((n_pixels, 3, 1))
        
        A[:, 0, 0] = self.cam_h_mtx[0, 0] - uc * self.cam_h_mtx[2, 0]
        A[:, 0, 1] = self.cam_h_mtx[0, 1] - uc * self.cam_h_mtx[2, 1]
        A[:, 0, 2] = self.cam_h_mtx[0, 2] - uc * self.cam_h_mtx[2, 2]
    
        A[:, 1, 0] = self.cam_h_mtx[1, 0] - vc * self.cam_h_mtx[2, 0]
        A[:, 1, 1] = self.cam_h_mtx[1, 1] - vc * self.cam_h_mtx[2, 1]
        A[:, 1, 2] = self.cam_h_mtx[1, 2] - vc * self.cam_h_mtx[2, 2]
    
        A[:, 2, 0] = self.proj_h_mtx[0, 0] - up * self.proj_h_mtx[2, 0]
        A[:, 2, 1] = self.proj_h_mtx[0, 1] - up * self.proj_h_mtx[2, 1]
        A[:, 2, 2] = self.proj_h_mtx[0, 2] - up * self.proj_h_mtx[2, 2]
        if self.processing == 'gpu':
            A_inv = cp.linalg.inv(A)
        else:
            A_inv = np.linalg.inv(A)
        
        c[:, 0, 0] = uc * self.cam_h_mtx[2, 3] - self.cam_h_mtx[0, 3]
        c[:, 1, 0] = vc * self.cam_h_mtx[2, 3] - self.cam_h_mtx[1, 3]
        c[:, 2, 0] = up * self.proj_h_mtx[2, 3] - self.proj_h_mtx[0, 3]
        if self.processing == 'gpu':
            coords = cp.einsum('ijk,ikl->lij', A_inv, c)[0]
            coords = cp.asnumpy(coords)
        else:
            coords = np.einsum('ijk,ikl->lij', A_inv, c)[0]
        return coords
    
    def reconstruction_pts(self, uv_true, unwrap_vector, mask):
        """
        Function to reconstruct 3D point coordinates of 2D points.
        """
        no_pts = uv_true.shape[0]
        unwrap_image = nstep.recover_image(unwrap_vector, mask, self.cam_height, self.cam_width)
        uv = cv2.undistortPoints(uv_true, self.cam_mtx, self.cam_dist, None, self.cam_mtx)
        uv = uv.reshape(uv.shape[0], 2)
        uv_true = uv_true.reshape(no_pts, 2)
        #  Extract x and y coordinate of each point as uc, vc
        uc = uv[:, 0]
        vc = uv[:, 1]
        # Determinate 'up' from circle center
        up = (nstep.bilinear_interpolate(unwrap_image, uv_true[:,0], uv_true[:,1]) - self.phase_st) * self.pitch_list[-1] / (2*np.pi)
        if self.processing == 'gpu':
            uc = cp.asarray(uv[:,0])
            vc = cp.asarray(uv[:,1])
            up = cp.asarray(up)
        coordintes = self.triangulation(uc, vc, up) #return is numpy
        return coordintes

    def point_error(self, cord1, cord2):
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
                              columns=['$\Delta x$', '$\Delta y$', '$\Delta z$', 
                                       '$abs(\Delta x)$', '$abs(\Delta y)$', '$abs(\Delta z)$'])
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
   
    def reconstruction_obj(self,
                           unwrap_vector,
                           mask):
        """
        Sub function to reconstruct object from phase map
        """
        if self.processing == 'cpu':
            unwrap_image = nstep.recover_image(unwrap_vector, mask, self.cam_height, self.cam_width)
            unwrap_dist = nstep.undistort(unwrap_image, self.cam_mtx, self.cam_dist)
            u = np.arange(0, unwrap_dist.shape[1])
            v = np.arange(0, unwrap_dist.shape[0])
            uc, vc = np.meshgrid(u, v)
            uc = uc[mask]
            vc = vc[mask]
            up = (unwrap_dist - self.phase_st) * self.pitch[-1] / (2 * np.pi)
            up = up[mask]
        elif self.processing == 'gpu':
            unwrap_image = nstep_cp.recover_image_cp(unwrap_vector, mask, self.cam_height, self.cam_width)
            unwrap_dist = nstep_cp.undistort_cp(unwrap_image, self.cam_mtx, self.cam_dist)
            u = cp.arange(0, unwrap_dist.shape[1])
            v = cp.arange(0, unwrap_dist.shape[0])
            uc, vc = cp.meshgrid(u, v)
            uc = uc[mask]
            vc = vc[mask]
            up = (unwrap_dist - self.phase_st) * self.pitch[-1] / (2 * cp.pi)
            up = up[mask]
        coords = self.triangulation(uc, vc, up) #return is numpy
        return coords

    def diff_funs_x(self,hc_11, hc_13, hc_22, hc_23, hc_33, hp_11, hp_12, hp_13,
                    hp_14, hp_31, hp_32, hp_33, hp_34, det, x_num, uc, vc, up):
        """
        Sub function used to calculate x coordinate variance.
        Ref: S.Zhong, High-Speed 3D Imaging with Digital Fringe Projection Techniques, CRC Press, 2016.
        Chapter 7 :Digital Fringe Projection System Calibration, section:7.3.6
    
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

    def diff_funs_y(self, hc_11, hc_13, hc_22, hc_23, hc_33, hp_11, hp_12, hp_13, 
                    hp_14, hp_31, hp_32, hp_33, hp_34, det, y_num, uc, vc, up):
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

    def diff_funs_z(self, hc_11, hc_13, hc_22, hc_23, hc_33, hp_11, hp_12, hp_13, 
                    hp_14, hp_31, hp_32, hp_33, hp_34, det, z_num, uc, vc, up):
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

    def sigma_random(self, modulation_vector, unwrap_vector, mask):
        '''
        Function to calculate variance of x,y,z coordinates
    
        '''
        if self.processing == 'cpu':
            unwrap_image = nstep.recover_image(unwrap_vector, mask, self.cam_height, self.cam_width)
            unwrap_dist = nstep.undistort(unwrap_image, self.cam_mtx, self.cam_dist)
            u = np.arange(0, unwrap_dist.shape[1])
            v = np.arange(0, unwrap_dist.shape[0])
            uc, vc = np.meshgrid(u, v)
            uc = uc[mask]
            vc = vc[mask]
            up = (unwrap_dist - self.phase_st) * self.pitch[-1] / (2 * np.pi)
            up = up[mask] 
        elif self.processing == 'gpu':
            unwrap_image = nstep_cp.recover_image_cp(unwrap_vector, mask, self.cam_height, self.cam_width)
            unwrap_dist = nstep_cp.undistort_cp(unwrap_image, self.cam_mtx, self.cam_dist)
            u = cp.arange(0, unwrap_dist.shape[1])
            v = cp.arange(0, unwrap_dist.shape[0])
            uc, vc = cp.meshgrid(u, v)
            uc = uc[mask]
            vc = vc[mask]
            up = (unwrap_dist - self.phase_st) * self.pitch[-1] / (2 * cp.pi)
            up = up[mask]
        
        sigma_sq_phi = (2 * self.sigma**2) / (self.N_list[-1] * modulation_vector**2)
        sigma_sq_up = sigma_sq_phi * self.pitch_list[-1]**2 / 4 * np.pi**2
        
        hc_11 = self.cam_h_mtx[0, 0]
        sigmasq_hc_11 = self.cam_h_mtx_std[0, 0]**2
        hc_13 = self.cam_h_mtx[0, 2]
        sigmasq_hc_13 = self.cam_h_mtx_std[0, 2]**2
        
        hc_22 = self.cam_h_mtx[1, 1]
        sigmasq_hc_22 = self.cam_h_mtx_std[1, 1]**2
        hc_23 = self.cam_h_mtx[1, 2]
        sigmasq_hc_23 = self.cam_h_mtx_std[1, 2]**2
        
        hc_33 = self.cam_h_mtx[2, 2]
        sigmasq_hc_33 = self.cam_h_mtx_std[2, 2]**2
        
        hp_11 = self.proj_h_mtx[0, 0]
        sigmasq_hp_11 = self.proj_h_mtx_std[0, 0]**2
        hp_12 = self.proj_h_mtx[0, 1]
        sigmasq_hp_12 = self.proj_h_mtx_std[0, 1]**2
        hp_13 = self.proj_h_mtx[0, 2]
        sigmasq_hp_13 = self.proj_h_mtx_std[0, 2]**2
        hp_14 = self.proj_h_mtx[0, 3]
        sigmasq_hp_14 = self.proj_h_mtx_std[0, 3]**2
        
        hp_31 = self.proj_h_mtx[2, 0]
        sigmasq_hp_31 = self.proj_h_mtx_std[2, 0]**2
        hp_32 = self.proj_h_mtx[2, 1]
        sigmasq_hp_32 = self.proj_h_mtx_std[2, 1]**2
        hp_33 = self.proj_h_mtx[2, 2]
        sigmasq_hp_33 = self.proj_h_mtx_std[2, 2]**2
        hp_34 = self.proj_h_mtx[2, 3]
        sigmasq_hp_34 = self.proj_h_mtx_std[2, 3]**2
        
        det = (hc_11 * hc_22 * hp_13 - up * hc_11 * hc_22 * hp_33 - hc_13 * hc_22 * hp_11 + up * hc_13 *hc_22 * hp_31 + uc * hc_22 * hc_33 * hp_11
               - uc * up * hc_22 * hc_33 * hp_31 - hc_11 * hc_23 * hp_12 + up * hc_11 * hc_23 * hp_32 + vc * hc_11 * hc_33 * hp_12 - vc * up * hc_11 * hc_33 * hp_32)
               
        
        x_num = -up * hc_13 * hc_22 * hp_34 + hc_13 * hc_22 * hp_14 + uc * up * hc_22 * hc_33 * hp_34 - uc * hc_22 * hc_33 * hp_14
        df_dup_x, df_dhc_11_x, df_dhc_13_x, df_dhc_22_x, df_dhc_23_x, df_dhc_33_x, df_dhp_11_x, df_dhp_12_x, df_dhp_13_x, df_dhp_14_x, df_dhp_31_x, df_dhp_32_x, df_dhp_33_x, df_dhp_34_x = self.diff_funs_x(hc_11, hc_13, hc_22, hc_23, hc_33, hp_11, hp_12, hp_13, hp_14, hp_31, hp_32, hp_33, hp_34, det, x_num, uc, vc, up)
        sigmasq_x = ((df_dup_x**2 * sigma_sq_up) + (df_dhc_11_x**2 * sigmasq_hc_11) + (df_dhc_13_x**2 * sigmasq_hc_13) + (df_dhc_22_x**2 * sigmasq_hc_22) + (df_dhc_23_x**2 * sigmasq_hc_23) + (df_dhc_33_x**2 * sigmasq_hc_33)
                    + (df_dhp_11_x**2 * sigmasq_hp_11) + (df_dhp_12_x**2 * sigmasq_hp_12) + (df_dhp_13_x**2 * sigmasq_hp_13) + (df_dhp_14_x**2 * sigmasq_hp_14) + (df_dhp_31_x**2 * sigmasq_hp_31) + (df_dhp_32_x**2 * sigmasq_hp_32) + (df_dhp_33_x**2 * sigmasq_hp_33) + (df_dhp_34_x**2 * sigmasq_hp_34))
        derv_x = np.stack((df_dup_x, df_dhc_11_x, df_dhc_13_x, df_dhc_22_x, df_dhc_23_x, df_dhc_33_x, df_dhp_11_x, df_dhp_12_x, df_dhp_13_x, df_dhp_14_x, df_dhp_31_x, df_dhp_32_x, df_dhp_33_x, df_dhp_34_x))
        
        y_num = -up * hc_11 * hc_23 * hp_34 + hc_11 * hc_23 * hp_14 + vc * up * hc_11 * hc_33 * hp_34 - vc * hc_11 * hc_33 * hp_14
        #y_num = (-hc_11 * (hc_23 - hc_33)*(up * hp_34 - hp_14))
        df_dup_y, df_dhc_11_y, df_dhc_13_y, df_dhc_22_y, df_dhc_23_y, df_dhc_33_y, df_dhp_11_y, df_dhp_12_y, df_dhp_13_y, df_dhp_14_y, df_dhp_31_y, df_dhp_32_y, df_dhp_33_y, df_dhp_34_y = self.diff_funs_y(hc_11, hc_13, hc_22, hc_23, hc_33, hp_11, hp_12, hp_13, hp_14, hp_31, hp_32, hp_33, hp_34, det, y_num, uc, vc, up)
        sigmasq_y = ((df_dup_y**2 * sigma_sq_up) + (df_dhc_11_y**2 * sigmasq_hc_11) + (df_dhc_13_y**2 * sigmasq_hc_13) + (df_dhc_22_y**2 * sigmasq_hc_22) + (df_dhc_23_y**2 * sigmasq_hc_23) + (df_dhc_33_y**2 * sigmasq_hc_33)
                    + (df_dhp_11_y**2 * sigmasq_hp_11) + (df_dhp_12_y**2 * sigmasq_hp_12) + (df_dhp_13_y**2 * sigmasq_hp_13) + (df_dhp_14_y**2 * sigmasq_hp_14) + (df_dhp_31_y**2 * sigmasq_hp_31) + (df_dhp_32_y**2 * sigmasq_hp_32) + (df_dhp_33_y**2 * sigmasq_hp_33) + (df_dhp_34_y**2 * sigmasq_hp_34))
        derv_y = np.stack((df_dup_y, df_dhc_11_y, df_dhc_13_y, df_dhc_22_y, df_dhc_23_y, df_dhc_33_y, df_dhp_11_y, df_dhp_12_y, df_dhp_13_y, df_dhp_14_y, df_dhp_31_y, df_dhp_32_y, df_dhp_33_y, df_dhp_34_y))
        
        z_num = up * hc_11 * hc_22 * hp_34 - hc_11 * hc_22 * hp_14 
        df_dup_z, df_dhc_11_z, df_dhc_13_z, df_dhc_22_z, df_dhc_23_z, df_dhc_33_z, df_dhp_11_z, df_dhp_12_z, df_dhp_13_z, df_dhp_14_z, df_dhp_31_z, df_dhp_32_z, df_dhp_33_z, df_dhp_34_z = self.diff_funs_z(hc_11, hc_13, hc_22, hc_23, hc_33, hp_11, hp_12, hp_13, hp_14, hp_31, hp_32, hp_33, hp_34, det, z_num, uc, vc, up)
        sigmasq_z = ((df_dup_z**2 * sigma_sq_up) + (df_dhc_11_z**2 * sigmasq_hc_11) + (df_dhc_13_z**2 * sigmasq_hc_13) + (df_dhc_22_z**2 * sigmasq_hc_22) + (df_dhc_23_z**2 * sigmasq_hc_23) + (df_dhc_33_z**2 * sigmasq_hc_33)
                    + (df_dhp_11_z**2 * sigmasq_hp_11) + (df_dhp_12_z**2 * sigmasq_hp_12) + (df_dhp_13_z**2 * sigmasq_hp_13) + (df_dhp_14_z**2 * sigmasq_hp_14) + (df_dhp_31_z**2 * sigmasq_hp_31) + (df_dhp_32_z**2 * sigmasq_hp_32) + (df_dhp_33_z**2 * sigmasq_hp_33) + (df_dhp_34_z**2 * sigmasq_hp_34))
        derv_z = np.stack((df_dup_z, df_dhc_11_z, df_dhc_13_z, df_dhc_22_z, df_dhc_23_z, df_dhc_33_z, df_dhp_11_z, df_dhp_12_z, df_dhp_13_z, df_dhp_14_z, df_dhp_31_z, df_dhp_32_z, df_dhp_33_z, df_dhp_34_z))
        
        return sigmasq_x, sigmasq_y, sigmasq_z, derv_x, derv_y, derv_z

    def complete_recon(self,
                       unwrap_vector, 
                       mask,
                       inte_rgb_vector, 
                       modulation_vector,    
                       temperature_vector):
        """
        Function to completely reconstruct object applying modulation mask to saving point cloud.
    
        Parameters
        ----------
        unwrap_vector: np.ndarray/cp.ndarray. 
                       Unwrapped phase map vector of object.
        mask: np.ndarray/cp.ndarray.
                        Masked used to convert between image and vector format of data.
        modulation_vector: np.ndarray/cp.ndarray.
                           Intensity modulation image.
        
        temperature_vector: np.ndarray/cp.ndarray.
                            Temperature data of object.
        Returns
        -------
        coords: np.ndarray/cp.ndarray.
                x,y,z coordinate array of each object point.
        inte_rgb: np.ndarray/cp.ndarray. 
                  Intensity (texture/ color) at each point.
        cordi_sigma: np.ndarray/cp.ndarray. 
                    Standard deviation of each pixel.
        """
        coords = self.reconstruction_obj(unwrap_vector, mask)
        sigmasq_x, sigmasq_y, sigmasq_z, derv_x, derv_y, derv_z = self.sigma_random( modulation_vector, unwrap_vector, mask)
        sigma_x = np.sqrt(sigmasq_x)
        sigma_y = np.sqrt(sigmasq_y)
        sigma_z = np.sqrt(sigmasq_z)
        cordi_sigma = np.vstack((sigma_x, sigma_y, sigma_z)).T
        xyz = list(map(tuple, coords)) 
        inte_img = inte_rgb_vector / np.nanmax(inte_rgb_vector)
        inte_rgb = np.stack((inte_img, inte_img, inte_img), axis=-1)
        color = list(map(tuple, inte_rgb))
        xyz_sigma = list(map(tuple, cordi_sigma))
        mod_vect = np.array(modulation_vector, dtype=[('modulation', 'f4')])
        if self.temp:
            temperature_vector = np.array(temperature_vector, dtype=[('temperature', 'f4')])
            PlyData(
                [
                    PlyElement.describe(np.array(xyz, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'points'),
                    PlyElement.describe(np.array(color, dtype=[('r', 'f4'), ('g', 'f4'), ('b', 'f4')]), 'color'),
                    PlyElement.describe(np.array(xyz_sigma, dtype=[('dx', 'f4'), ('dy', 'f4'), ('dz', 'f4')]), 'std'),
                    PlyElement.describe(np.array(temperature_vector, dtype=[('temperature', 'f4')]), 'temperature'),
                    PlyElement.describe(np.array(modulation_vector, dtype=[('modulation', 'f4')]), 'modulation')
                    
                ]).write(os.path.join(self.obj_path, 'obj.ply'))
        else:
            PlyData(
                [
                    PlyElement.describe(np.array(xyz, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'points'),
                    PlyElement.describe(np.array(color, dtype=[('r', 'f4'), ('g', 'f4'), ('b', 'f4')]), 'color'),
                    PlyElement.describe(np.array(xyz_sigma, dtype=[('dx', 'f4'), ('dy', 'f4'), ('dz', 'f4')]), 'std'),
                    PlyElement.describe(np.array(mod_vect, dtype=[('modulation', 'f4')]), 'modulation')
                    
                ]).write(os.path.join(self.obj_path, 'obj.ply'))
          
        return coords, inte_rgb, cordi_sigma

    def obj_reconst_wrapper(self):
        """
        Function for 3D reconstruction of object based on different unwrapping method.
        Returns
        -------
        obj_cordi: np.ndarray.
                    Array of reconstructed x,y,z coordinates of each points on the object
        obj_color: np.ndarray. 
                   Color (texture/ intensity) at each point.
    
        """
       
        if self.data_type == 'jpeg':
            if os.path.exists(os.path.join(self.path, 'capt_*.jpg')):
                img_path = sorted(glob.glob(os.path.join(self.obj_path, 'capt_*.jpg')), key=os.path.getmtime)
                images_arr = [cv2.imread(file, 0) for file in img_path]
            else:
                print("ERROR:Data path does not exist!")
                return
        elif self.data_type == 'npy':
            if os.path.exists(os.path.join(self.path, 'capt_*.npy')):
                images_arr = np.array(images_arr).astype(np.float64)
            else:
                print("ERROR:Data path does not exist!")
                images_arr = None
        else:
            print("ERROR: data type is not supported, must be '.jpeg' or '.npy'.")
            images_arr = None
        
        if self.type_unwrap == 'multifreq':
            if self.processing == 'cpu':
                modulation_vector, orig_img, phase_map, mask = nstep.phase_cal(images_arr, self.limit, self.N_list, False )
                phase_map[0][phase_map[0] < EPSILON] = phase_map[0][phase_map[0] < EPSILON] + 2 * np.pi
                unwrap_vector, k_arr = nstep.multifreq_unwrap(self.pitch_list, phase_map, self.kernel, 'v')
               
            elif self.processing == 'gpu':
               images_arr = cp.asarray(images_arr) 
               modulation_vector, orig_img, phase_map, mask = nstep_cp.phase_cal_cp(images_arr, self.limit, self.N_list, False )
               phase_map[0][phase_map[0] < EPSILON] = phase_map[0][phase_map[0] < EPSILON] + 2 * np.pi
               unwrap_vector, k_arr = nstep_cp.multifreq_unwrap_cp(self.pitch_list, phase_map, self.kernel, self.fringe_direc)
           
        elif self.type_unwrap == 'multiwave':
            eq_wav12 = (self.pitch_list[-1] * self.pitch_list[1]) / (self.pitch_list[1] - self.pitch_list[-1])
            eq_wav123 = self.pitch_list[0] * eq_wav12 / (self.pitch_list[0] - eq_wav12)
            self.pitch_list = np.insert(self.pitch_list, 0, eq_wav123)
            self.pitch_list = np.insert(self.pitch_list, 2, eq_wav12)
            modulation_vector, orig_img, phase_map, mask = nstep.phase_cal(images_arr, self.limit, self.N_list, False)
            phase_wav12 = np.mod(phase_map[0] - phase_map[1], 2 * np.pi)
            phase_wav123 = np.mod(phase_wav12 - phase_map[2], 2 * np.pi)
            phase_wav123[phase_wav123 > TAU] = phase_wav123[phase_wav123 > TAU] - 2 * np.pi
            #unwrapped phase
            phase_arr = np.stack([phase_wav123, phase_map[2], phase_wav12, phase_map[1], phase_map[0]])
            unwrap_vector, k = nstep.multiwave_unwrap(self.pitch_list, phase_arr, self.kernel, self.fringe_direc)
        inte_img = cv2.imread(os.path.join(self.obj_path, 'white.jpg'))
        if self.temp:
            temperature = np.load(os.path.join(self.obj_path, 'temperature.npy'))
            temperature_vector = temperature[mask]
        else:
            temperature = None
        inte_rgb = inte_img[..., ::-1].copy()
        inte_rgb_vector = inte_rgb[mask]
        np.save(os.path.join(self.obj_path, '{}_obj_mod.npy'.format(self.type_unwrap)), modulation_vector)
        np.save(os.path.join(self.obj_path, '{}_unwrap.npy'.format(self.type_unwrap)), unwrap_vector)
        obj_cordi, obj_color, cordi_sigma = self.complete_recon(unwrap_vector,
                                                                mask,
                                                                inte_rgb_vector,
                                                                modulation_vector,
                                                                temperature_vector)
        return obj_cordi, obj_color, temperature_vector, cordi_sigma, modulation_vector
       
       
