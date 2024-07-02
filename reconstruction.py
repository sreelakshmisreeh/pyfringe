#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:00:19 2022

@author: Sreelakshmi
"""
import numpy as np
import cupy as cp
import glob
import cv2
import os
from plyfile import PlyData, PlyElement
import nstep_fringe as nstep
import nstep_fringe_cp as nstep_cp
import matplotlib.pyplot as plt
import pickle

EPSILON = -0.5
TAU = 5.5
#TODO: Convert to pyqtgraph. 
#Note: Befor running probabilistic reconstruction make sure the calibration parameter have single mean and std.
#Since bootstrap now performs multiple number of poses the mean file will have mean of each number of poses.
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
                 dark_bias_path,
                 calib_path,
                 object_path,
                 model_path=None,
                 temp=False,
                 save_ply=True,
                 probability=False,
                 prob_up=True):
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
        self.data_type = data_type
        self.save_ply = save_ply
        self.probability = probability
        self.prob_up=prob_up
        
        self.mask = None
        if (self.type_unwrap == 'multifreq') or (self.type_unwrap == 'multiwave'):
            self.phase_st = 0
        else:
            print('ERROR: Invalid type_unwrap')
            return
        if not os.path.exists(self.calib_path):
            print('ERROR:calibration parameter path  %s does not exist' % self.calib_path)
        if not object_path:
            self.object_path = calib_path
        if not os.path.exists(object_path):
            print('ERROR:Path for noise error  %s does not exist' % self.calib_path)
        else:
            self.object_path = object_path
            
        if not os.path.exists(dark_bias_path):
             print('ERROR:Path for dark bias  %s does not exist' % self.calib_path)
        else:
            self.dark_bias = np.load(dark_bias_path)
    
        if processing == 'cpu':
            self.processing = processing
            calibration_mean = np.load(os.path.join(self.calib_path, '{}_mean_calibration_param.npz'.format(self.type_unwrap)))
            self.cam_mtx = calibration_mean["cam_mtx_mean"]
            self.cam_dist = calibration_mean["cam_dist_mean"]
            self.proj_mtx = calibration_mean["proj_mtx_mean"]
            self.proj_dist = calibration_mean["proj_dist_mean"]
            self.camproj_rot_mtx = calibration_mean["st_rmat_mean"]
            self.camproj_trans_mtx = calibration_mean["st_tvec_mean"]
            self.cam_h_mtx = calibration_mean["cam_h_mtx_mean"]
            self.proj_h_mtx = calibration_mean["proj_h_mtx_mean"]
            self.uc_img = np.load(os.path.join(self.calib_path,"uc_img.npy"))
            self.vc_img = np.load(os.path.join(self.calib_path,"vc_img.npy"))
            if not os.path.exists(model_path):
                 print('ERROR:Path for noise error  %s does not exist' % self.calib_path)
            else:
                self.model = np.load(model_path)
            if  ((probability == True) & (prob_up == False)):
                calibration_std = np.load(os.path.join(self.calib_path, '{}_std_calibration_param.npz'.format(self.type_unwrap)))
                self.cam_h_mtx_std = calibration_std["cam_h_mtx_std"]
                self.proj_h_mtx_std = calibration_std["proj_h_mtx_std"]
                
            else:
                self.proj_h_mtx_std = np.zeros((3,4))
                self.cam_h_mtx_std = np.zeros((3,4))
               
        elif processing == 'gpu':
            self.processing = processing
            calibration_mean = cp.load(os.path.join(self.calib_path, '{}_mean_calibration_param.npz'.format(self.type_unwrap)))
            self.cam_mtx = cp.asarray(calibration_mean["cam_mtx_mean"])
            self.cam_dist = cp.asarray(calibration_mean["cam_dist_mean"])
            self.proj_mtx = cp.asarray(calibration_mean["proj_mtx_mean"])
            self.proj_dist = cp.asarray(calibration_mean["proj_dist_mean"])
            self.camproj_rot_mtx = cp.asarray(calibration_mean["st_rmat_mean"])
            self.camproj_trans_mtx = cp.asarray(calibration_mean["st_tvec_mean"])
            self.cam_h_mtx = cp.asarray(calibration_mean["cam_h_mtx_mean"])
            self.proj_h_mtx = cp.asarray(calibration_mean["proj_h_mtx_mean"])
            self.uc_img = cp.load(os.path.join(self.calib_path,"uc_img.npy"))
            self.vc_img = cp.load(os.path.join(self.calib_path,"vc_img.npy"))
            if not os.path.exists(model_path):
                 print('ERROR:Path for noise error  %s does not exist' % self.calib_path)
            else:
                self.model = cp.load(model_path)
                
            if ((probability == True) & (prob_up == False)):
                calibration_std = cp.load(os.path.join(self.calib_path, '{}_std_calibration_param.npz'.format(self.type_unwrap)))
                self.cam_h_mtx_std = cp.asarray(calibration_std["cam_h_mtx_std"])
                self.proj_h_mtx_std = cp.asarray(calibration_std["proj_h_mtx_std"])
               
            else:
                self.proj_h_mtx_std = cp.zeros((3,4))
                self.cam_h_mtx_std = cp.zeros((3,4))
                
        else:
            self.processing = None
            print("ERROR: Invalid processing type.")
            return
            
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
    
    def reconstruction_pts(self, uv_true, unwrap_vector):
        """
        Function to reconstruct 3D point coordinates of 2D points.
        """
        no_pts = uv_true.shape[0]
        unwrap_image = nstep.recover_image(unwrap_vector, self.mask, self.cam_height, self.cam_width)
        if self.processing == "gpu":
            c_mtx = cp.asnumpy(self.cam_mtx)
            c_dist = cp.asnumpy(self.cam_dist)
        uv = cv2.undistortPoints(uv_true, c_mtx, c_dist, None, c_mtx)
        uv = uv.reshape(uv.shape[0], 2)
        uv_true = uv_true.reshape(no_pts, 2)
        #  Extract x and y coordinate of each point as uc, vc
        uc = uv[:, 0]
        vc = uv[:, 1]
        # Determinate 'up' from circle center
        up = (nstep.bilinear_interpolate(unwrap_image, uv_true[:, 0], uv_true[:, 1]) - self.phase_st) * self.pitch_list[-1] / (2*np.pi)
        if self.processing == 'gpu':
            uc = cp.asarray(uv[:, 0])
            vc = cp.asarray(uv[:, 1])
            up = cp.asarray(up)
        coordintes = self.triangulation(uc, vc, up) #return is numpy
        return coordintes
   
    def reconstruction_obj(self,
                           unwrap_vector, sigma_sq_phi):
        """
        Sub function to reconstruct object from phase map
        """
        if self.processing == 'cpu':
            unwrap_image = nstep.recover_image(unwrap_vector, self.mask, self.cam_height, self.cam_width)
            unwrap_dist, unwrap_var = nstep.undistort(unwrap_image, self.cam_mtx, self.cam_dist, 
                                                      sigmasq_image=sigma_sq_phi)
            self.mask = ~np.isnan(unwrap_dist)
            u = np.arange(0, self.cam_width)
            v = np.arange(0, self.cam_height)
            uc_grid, vc_grid = np.meshgrid(u, v)
            # cordinates = np.stack((vc_grid.ravel(),uc_grid.ravel()),axis=1).astype("float64")
            # uv = cv2.undistortPoints(cordinates, self.cam_mtx, self.cam_dist, None, self.cam_mtx).reshape((self.cam_width*self.cam_height,2))
            # uc = uv[:,1]
            # vc = uv[:,0]
            # uc = uc.reshape(self.cam_height, self.cam_width)[self.mask]
            # vc = vc.reshape(self.cam_height, self.cam_width)[self.mask]
            uc = uc_grid[self.mask]
            vc = vc_grid[self.mask]
            up = (unwrap_dist - self.phase_st) * self.pitch_list[-1] / (2 * np.pi)
            up = up[self.mask]
        else:
            unwrap_image = nstep_cp.recover_image_cp(unwrap_vector, self.mask, self.cam_height, self.cam_width)
            
            unwrap_dist, unwrap_var = nstep_cp.undistort_cp(unwrap_image, self.cam_mtx, self.cam_dist,
                                                sigmasq_image=sigma_sq_phi)
            self.mask = ~cp.isnan(unwrap_dist)
            u = cp.arange(0,self.cam_width)
            v = cp.arange(0, self.cam_height)
            uc, vc = cp.meshgrid(u, v)
            uc = self.uc_img[self.mask]
            vc = self.vc_img[self.mask]
            up = (unwrap_dist - self.phase_st) * self.pitch_list[-1] / (2 * cp.pi)
         
            up = up[self.mask]
            self.mask = cp.asnumpy(self.mask)
        
        coords = self.triangulation(uc, vc, up) #return is numpy
        return coords, uc, vc, up, unwrap_var

    @staticmethod
    def diff_funs_x(hc_11, hc_13, hc_22, hc_23, hc_33, hp_11, hp_12, hp_13,
                    hp_14, hp_31, hp_32, hp_33, hp_34, det, x_num, uc, vc, up):
        """
        Sub function used to calculate x coordinate variance.
        Ref: S.Zhong, High-Speed 3D Imaging with Digital Fringe Projection Techniques, CRC Press, 2016.
        Chapter 7 :Digital Fringe Projection System Calibration, section:7.3.6
    
        """
        df_dup = (det * (-hc_13 * hc_22 * hp_34 + uc * hc_22 * hc_33 * hp_34) - 
                  x_num * (-hc_11 * hc_22 * hp_33 + hc_13 * hc_22 * hp_31 - uc * hc_22 * hc_33 * hp_31 + hc_11 * hc_23 * hp_32 - vc * hc_11 * hc_33 * hp_32))/det**2
        df_dhc_11 = (- x_num * (hc_22 * hp_13 - up * hc_22 * hp_33 - hc_23 * hp_12 + up * hc_23 * hp_32 + vc * hc_33 * hp_12 - vc * up * hc_33 * hp_32))/det**2
        df_dhc_13 = (det * (-up * hc_22 * hp_34 + hc_22 * hp_14) - x_num * (-hc_22 * hp_11 + up * hc_22 * hp_31))/det**2
        df_dhc_22 = (det * (-up * hc_13 * hp_34 + hc_13 * hp_14 + uc * up * hc_33 * hp_34 - uc * hc_33 * hp_14) - 
                     x_num * (hc_11 * hp_13 - up * hc_11 * hp_33 - hc_13 * hp_11 + up * hc_13 * hp_31 + uc * hc_33 * hp_11 - uc * up * hc_33 * hp_31))/det**2
        df_dhc_23 = (- x_num * (-hc_11 * hp_12 + up * hc_11 * hp_32))/det**2
        df_dhc_33 = (det * (uc* up * hc_22 * hp_34 - uc * hc_22 * hp_14) - 
                     x_num * (uc * hc_22 * hp_11 - uc* up * hc_22 * hp_31 + vc * hc_11 * hp_12 - vc * up * hc_11 * hp_32)) / det**2
        df_dhp_11 = (- x_num *(-hc_13 * hc_22 + uc * hc_22 * hc_33))/det**2
        df_dhp_12 = (- x_num * (-hc_11*hc_23 + vc * hc_11 * hc_33))/det**2
        df_dhp_13 = (- x_num * (hc_11 * hc_22))/det**2
        df_dhp_14 = (det * (hc_13 * hc_22 - uc * hc_22 * hc_33))/det**2
        df_dhp_31 = (- x_num * (up * hc_22 * hc_13 - uc * up * hc_22 * hc_33))/det**2
        df_dhp_32 = (- x_num * (up * hc_11 * hc_23 - vc * up * hc_11 * hc_33))/det**2
        df_dhp_33 = (- x_num * (-up * hc_11 * hc_22))/det**2
        df_dhp_34 = (det * (-up * hc_13 * hc_22 + uc * up * hc_22 * hc_33))/det**2
        
        return df_dup, df_dhc_11, df_dhc_13, df_dhc_22, df_dhc_23, df_dhc_33, df_dhp_11, df_dhp_12, df_dhp_13, df_dhp_14, df_dhp_31, df_dhp_32, df_dhp_33, df_dhp_34

    @staticmethod
    def diff_funs_y(hc_11, hc_13, hc_22, hc_23, hc_33, hp_11, hp_12, hp_13,
                    hp_14, hp_31, hp_32, hp_33, hp_34, det, y_num, uc, vc, up):
        """
        Subfunction used to calculate y cordinate variance
    
        """
        df_dup = (det * (-hc_11 * hc_23 * hp_34 + vc * hc_11 * hc_33 * hp_34) - 
                  y_num * (-hc_11 * hc_22 * hp_33 + hc_13 * hc_22 * hp_31 - uc * hc_22 * hc_33 * hp_31 + hc_11 * hc_23 * hp_32 - vc * hc_11 * hc_33 * hp_32))/det**2
        df_dhc_11 = (det * (-up * hc_23 * hp_34 + hc_23 * hp_14 + vc * up * hc_33 * hp_34 - vc * hc_33 * hp_14) -
                     y_num * (hc_22 * hp_13 - up * hc_22 * hp_33 - hc_23 * hp_12 + up * hc_23 * hp_32 + vc * hc_33 * hp_12 - vc * up * hc_33 * hp_32))/det**2
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

    @staticmethod
    def diff_funs_z(hc_11, hc_13, hc_22, hc_23, hc_33, hp_11, hp_12, hp_13,
                    hp_14, hp_31, hp_32, hp_33, hp_34, det, z_num, uc, vc, up):
        """
        Sub function used to calculate z coordinate variance
    
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

    def sigma_random(self, sigma_sq_phi, uc, vc, up):
        """
        Function to calculate variance of x,y,z coordinates
        """
        
        sigma_sq_up = sigma_sq_phi * self.pitch_list[-1]**2 / (4 * np.pi**2)
        
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
        df_dup_x, df_dhc_11_x, df_dhc_13_x, df_dhc_22_x, df_dhc_23_x, df_dhc_33_x, df_dhp_11_x, df_dhp_12_x, df_dhp_13_x, df_dhp_14_x, df_dhp_31_x, df_dhp_32_x, df_dhp_33_x, df_dhp_34_x = Reconstruction.diff_funs_x(hc_11, hc_13, hc_22, hc_23, hc_33, hp_11, hp_12, hp_13, hp_14, hp_31, hp_32, hp_33, hp_34, det, x_num, uc, vc, up)
        
        
        y_num = -up * hc_11 * hc_23 * hp_34 + hc_11 * hc_23 * hp_14 + vc * up * hc_11 * hc_33 * hp_34 - vc * hc_11 * hc_33 * hp_14
        #y_num = (-hc_11 * (hc_23 - hc_33)*(up * hp_34 - hp_14))
        df_dup_y, df_dhc_11_y, df_dhc_13_y, df_dhc_22_y, df_dhc_23_y, df_dhc_33_y, df_dhp_11_y, df_dhp_12_y, df_dhp_13_y, df_dhp_14_y, df_dhp_31_y, df_dhp_32_y, df_dhp_33_y, df_dhp_34_y = Reconstruction.diff_funs_y(hc_11, hc_13, hc_22, hc_23, hc_33, hp_11, hp_12, hp_13, hp_14, hp_31, hp_32, hp_33, hp_34, det, y_num, uc, vc, up)
        
        z_num = up * hc_11 * hc_22 * hp_34 - hc_11 * hc_22 * hp_14 
        df_dup_z, df_dhc_11_z, df_dhc_13_z, df_dhc_22_z, df_dhc_23_z, df_dhc_33_z, df_dhp_11_z, df_dhp_12_z, df_dhp_13_z, df_dhp_14_z, df_dhp_31_z, df_dhp_32_z, df_dhp_33_z, df_dhp_34_z = Reconstruction.diff_funs_z(hc_11, hc_13, hc_22, hc_23, hc_33, hp_11, hp_12, hp_13, hp_14, hp_31, hp_32, hp_33, hp_34, det, z_num, uc, vc, up)
        if self.prob_up:
            sigmasq_x = (df_dup_x**2 * sigma_sq_up)
            sigmasq_y = (df_dup_y**2 * sigma_sq_up)
            sigmasq_z = df_dup_z**2 * sigma_sq_up
            derv_x = df_dup_x
            derv_y = df_dup_y
            derv_z = df_dup_z
        else:
            sigmasq_x = ((df_dup_x**2 * sigma_sq_up) + (df_dhc_11_x**2 * sigmasq_hc_11) + 
                         (df_dhc_13_x**2 * sigmasq_hc_13) + (df_dhc_22_x**2 * sigmasq_hc_22) + 
                         (df_dhc_23_x**2 * sigmasq_hc_23) + (df_dhc_33_x**2 * sigmasq_hc_33) +
                         (df_dhp_11_x**2 * sigmasq_hp_11) + (df_dhp_12_x**2 * sigmasq_hp_12) + 
                         (df_dhp_13_x**2 * sigmasq_hp_13) + (df_dhp_14_x**2 * sigmasq_hp_14) + 
                         (df_dhp_31_x**2 * sigmasq_hp_31) + (df_dhp_32_x**2 * sigmasq_hp_32) + 
                         (df_dhp_33_x**2 * sigmasq_hp_33) + (df_dhp_34_x**2 * sigmasq_hp_34))
            sigmasq_y = ((df_dup_y**2 * sigma_sq_up) + (df_dhc_11_y**2 * sigmasq_hc_11) + 
                         (df_dhc_13_y**2 * sigmasq_hc_13) + (df_dhc_22_y**2 * sigmasq_hc_22) + 
                         (df_dhc_23_y**2 * sigmasq_hc_23) + (df_dhc_33_y**2 * sigmasq_hc_33) +
                         (df_dhp_11_y**2 * sigmasq_hp_11) + (df_dhp_12_y**2 * sigmasq_hp_12) + 
                         (df_dhp_13_y**2 * sigmasq_hp_13) + (df_dhp_14_y**2 * sigmasq_hp_14) + 
                         (df_dhp_31_y**2 * sigmasq_hp_31) + (df_dhp_32_y**2 * sigmasq_hp_32) + 
                         (df_dhp_33_y**2 * sigmasq_hp_33) + (df_dhp_34_y**2 * sigmasq_hp_34))
            sigmasq_z = ((df_dup_z**2 * sigma_sq_up) + (df_dhc_11_z**2 * sigmasq_hc_11) + 
                         (df_dhc_13_z**2 * sigmasq_hc_13) + (df_dhc_22_z**2 * sigmasq_hc_22) + 
                         (df_dhc_23_z**2 * sigmasq_hc_23) + (df_dhc_33_z**2 * sigmasq_hc_33) +
                         (df_dhp_11_z**2 * sigmasq_hp_11) + (df_dhp_12_z**2 * sigmasq_hp_12) + 
                         (df_dhp_13_z**2 * sigmasq_hp_13) + (df_dhp_14_z**2 * sigmasq_hp_14) + 
                         (df_dhp_31_z**2 * sigmasq_hp_31) + (df_dhp_32_z**2 * sigmasq_hp_32) + 
                         (df_dhp_33_z**2 * sigmasq_hp_33) + (df_dhp_34_z**2 * sigmasq_hp_34))
            
            derv_x = np.stack((df_dup_x, df_dhc_11_x, df_dhc_13_x, df_dhc_22_x, df_dhc_23_x, df_dhc_33_x, df_dhp_11_x, df_dhp_12_x, df_dhp_13_x, df_dhp_14_x, df_dhp_31_x, df_dhp_32_x, df_dhp_33_x, df_dhp_34_x))
            derv_y = np.stack((df_dup_y, df_dhc_11_y, df_dhc_13_y, df_dhc_22_y, df_dhc_23_y, df_dhc_33_y, df_dhp_11_y, df_dhp_12_y, df_dhp_13_y, df_dhp_14_y, df_dhp_31_y, df_dhp_32_y, df_dhp_33_y, df_dhp_34_y))
            derv_z = np.stack((df_dup_z, df_dhc_11_z, df_dhc_13_z, df_dhc_22_z, df_dhc_23_z, df_dhc_33_z, df_dhp_11_z, df_dhp_12_z, df_dhp_13_z, df_dhp_14_z, df_dhp_31_z, df_dhp_32_z, df_dhp_33_z, df_dhp_34_z))
        if self.processing == 'gpu':
            sigmasq_x = cp.asnumpy(sigmasq_x)
            sigmasq_y = cp.asnumpy(sigmasq_y)
            sigmasq_z = cp.asnumpy(sigmasq_z)
            derv_x = cp.asnumpy(derv_x)
            derv_y = cp.asnumpy(derv_y)
            derv_z = cp.asnumpy(derv_z)
            
        return sigmasq_x, sigmasq_y, sigmasq_z, derv_x, derv_y, derv_z    
        
    # This will be optional once instant display is setup
    def cloud_save(self):
        
        xyz = list(map(tuple, self.coords)) 
        color = list(map(tuple, self.inte_rgb))
        if self.temp:
            temperature_vector = np.array(self.temperature_vector, dtype=[('temperature', 'f4')])
        else:
            temperature_vector = [None]
        
        if self.probability:
            xyz_sigma = list(map(tuple, self.cordi_sigma))
            xyz_quality = np.array(self.quality_vector, dtype=[('quality', 'f4')])
        else:
            xyz_sigma = [None]
            xyz_quality = [None]
            
        PlyData(
            [
                PlyElement.describe(np.array(xyz, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'points'),
                PlyElement.describe(np.array(color, dtype=[('r', 'f4'), ('g', 'f4'), ('b', 'f4')]), 'color'),
                PlyElement.describe(np.array(xyz_sigma, dtype=[('dx', 'f4'), ('dy', 'f4'), ('dz', 'f4')]), 'std'),
                PlyElement.describe(np.array(temperature_vector, dtype=[('temperature', 'f4')]), 'temperature'),
                PlyElement.describe(np.array(xyz_quality, dtype=[('quality', 'f4')]), 'quality'),
            ]).write(os.path.join(self.object_path, 'obj.ply'))
        print("\n Point cloud saved at %s"% (os.path.join(self.object_path, 'obj.ply')))
        return
    

    def complete_recon(self,
                       unwrap_vector, 
                       inte_rgb_image,  
                       temperature_image,
                       sigma_sq_phi,
                       quality):
        """
        Function to completely reconstruct object applying modulation mask to saving point cloud.
    
        Parameters
        ----------
        unwrap_vector: np.ndarray/cp.ndarray. 
                       Unwrapped phase map vector of object.
        inte_rgb_image: np.ndarray/cp.ndarray.
                         Object texture image.
        temperature_image: np.ndarray/cp.ndarray.
                            Temperature data of object.
        sigma_sq_phi: np.ndarray/cp.ndarray.
                        Phase variance map.
        prob_up: bool.
                  When probability is true, if prob_up is true consider only up standard deviation for calculating coordinate standard deviation
        Returns
        -------
        coords: np.ndarray/cp.ndarray.
                x,y,z coordinate array of each object point.
        inte_rgb: np.ndarray/cp.ndarray. 
                  Intensity (texture/ color) at each point.
        cordi_sigma: np.ndarray/cp.ndarray. 
                    Standard deviation of each pixel.
        """
        coords, uc, vc, up, sigmasq_phi_dist = self.reconstruction_obj(unwrap_vector, sigma_sq_phi)
        inte_img = inte_rgb_image[self.mask] / np.nanmax(inte_rgb_image[self.mask])
        inte_rgb = np.stack((inte_img, inte_img, inte_img), axis=-1)
        if self.probability:
            sigma_sq_low_phi_vect = sigmasq_phi_dist[self.mask]
            sigmasq_x, sigmasq_y, sigmasq_z, derv_x, derv_y, derv_z = self.sigma_random(sigma_sq_low_phi_vect, uc, vc, up)
            sigma_x = np.sqrt(sigmasq_x)
            sigma_y = np.sqrt(sigmasq_y)
            sigma_z = np.sqrt(sigmasq_z)
            cordi_sigma = np.vstack((sigma_x, sigma_y, sigma_z)).T
            quality_vector = quality[self.mask]
        else:
            cordi_sigma = None
            quality_vector = None
            sigma_sq_low_phi_vect = None
        
        if self.temp:
            temperature_vector = temperature_image[self.mask]
        else:
            temperature_vector = [None]
        self.coords = coords
        self.inte_rgb = inte_rgb
        self.cordi_sigma = cordi_sigma
        self.temperature_vector = temperature_vector
        self.sigma_sq_low_phi_vect = sigma_sq_low_phi_vect
        self.quality_vector = quality_vector
        if self.save_ply: 
            self.cloud_save()  
        return coords, inte_rgb, cordi_sigma

    def obj_reconst_wrapper(self):
        """
        Function for 3D reconstruction of object based on different unwrapping method.
        Parameters
        ----------
        prob_up: bool.
                 When probability is true, if prob_up is true consider only up standard deviation for calculating coordinate standard deviation
        Returns
        -------
        obj_cordi: np.ndarray.
                    Array of reconstructed x,y,z coordinates of each points on the object
        obj_color: np.ndarray. 
                   Color (texture/ intensity) at each point.
    
        """
        if self.data_type == 'tiff':
            if os.path.exists(os.path.join(self.object_path, 'capt_000_000000.tiff')):
                img_path = sorted(glob.glob(os.path.join(self.object_path, 'capt_*')), key=lambda x:int(os.path.basename(x)[-11:-5]))
                images_arr = np.array([cv2.imread(file, 0) for file in img_path])- self.dark_bias
                
                
            else:
                print("ERROR:Data path does not exist!")
                return
            if self.temp:
                if not os.path.exists(os.path.join(self.object_path, 'temperature.tiff')):
                    print("ERROR: Temperature data path %s does not exist"% (os.path.join(self.object_path, 'temperature.tiff')))
                else:
                    temperature_image = np.load(os.path.join(self.object_path, 'temperature.tiff'))
            else:
                temperature_image = None
        elif self.data_type == 'npy':
            if os.path.exists(os.path.join(self.object_path, 'capt_000_000000.npy')):
                images_arr = np.load(os.path.join(self.object_path, 'capt_000_000000.npy')).astype(np.float64) - self.dark_bias
                
            else:
                print("ERROR:Data path does not exist!")
                images_arr = None
            if self.temp:
                if not os.path.exists(os.path.join(self.object_path, 'temperature.npy')):
                    print("ERROR: Temperature data path %s does not exist"% (os.path.join(self.object_path, 'temperature.npy')))
                else:
                    temperature_image = np.load(os.path.join(self.object_path, 'temperature.npy'))
            else:
                temperature_image = None
        else:
            print("ERROR: data type is not supported, must be '.tiff' or '.npy'.")
            images_arr = None
            
        if self.type_unwrap == 'multifreq':
            if self.processing == 'cpu':
                modulation_vector, orig_img, phase_map, mask = nstep.phase_cal(images_arr,
                                                                               self.limit, 
                                                                               self.N_list,
                                                                               False)
                self.mask = mask
                phase_map[0][phase_map[0] < EPSILON] = phase_map[0][phase_map[0] < EPSILON] + 2 * np.pi
                unwrap_vector, k_arr, mask = nstep.multifreq_unwrap(self.pitch_list,
                                                              phase_map,
                                                              self.kernel,
                                                              self.fringe_direc,
                                                              self.mask,
                                                              self.cam_width,
                                                              self.cam_height)
                orig_img = orig_img[-1] 
                self.mask = mask
                if self.probability:
                    cov_arr_l,_ = nstep.pred_var_fn(images_arr[-(self.N_list[-2]+self.N_list[-1]): -self.N_list[-1]], self.model)
                    
                    sigma_sq_phi_l = nstep.var_func(images_arr[-(self.N_list[-2]+self.N_list[-1]): -self.N_list[-1]],
                                                  self.mask,
                                                  self.N_list[-2],
                                                  cov_arr_l)
                    cov_arr_h,_ = nstep.pred_var_fn(images_arr[-self.N_list[-1]:], self.model)
                    sigma_sq_phi = nstep.var_func(images_arr[-self.N_list[-1]:],
                                                  self.mask,
                                                  self.N_list[-1],
                                                  cov_arr_h)
                    sigma_sq_delta_phi = ((self.pitch_list[-2]/self.pitch_list[-1])**2 * sigma_sq_phi_l) + sigma_sq_phi
                    quality = np.pi/np.sqrt(sigma_sq_delta_phi)
                    
                else:
                    sigma_sq_phi = None
                    quality = None
            elif self.processing == 'gpu':
                images_arr_cp = cp.asarray(images_arr)
                modulation_vector, orig_img, phase_map, mask = nstep_cp.phase_cal_cp(images_arr_cp,
                                                                                     self.limit,
                                                                                     self.N_list,
                                                                                     False)
                phase_map[0][phase_map[0] < EPSILON] = phase_map[0][phase_map[0] < EPSILON] + 2 * np.pi
                self.mask = mask
                unwrap_vector, k_arr, mask = nstep_cp.multifreq_unwrap_cp(self.pitch_list,
                                                                    phase_map,
                                                                    self.kernel,
                                                                    self.fringe_direc,
                                                                    self.mask,
                                                                    self.cam_width,
                                                                    self.cam_height)
                orig_img = cp.asnumpy(orig_img[-1])
                self.mask = mask
                if self.probability:
                    
                    cov_arr_l,_ = nstep_cp.pred_var_fn(images_arr_cp[-(self.N_list[-2]+self.N_list[-1]): -self.N_list[-1]], self.model)
                    
                    sigma_sq_phi_l = nstep_cp.var_func(images_arr_cp[-(self.N_list[-2]+self.N_list[-1]): -self.N_list[-1]],
                                                  self.mask,
                                                  self.N_list[-2],
                                                  cov_arr_l)
                    cov_arr_h,_ = nstep_cp.pred_var_fn(images_arr_cp[-self.N_list[-1]:], self.model)
                    sigma_sq_phi = nstep_cp.var_func(images_arr_cp[-self.N_list[-1]:],
                                                  self.mask,
                                                  self.N_list[-1],
                                                  cov_arr_h)
                    sigma_sq_delta_phi = ((self.pitch_list[-2]/self.pitch_list[-1])**2 * sigma_sq_phi_l) + sigma_sq_phi
                    quality = np.pi/np.sqrt(sigma_sq_delta_phi)
                    quality = cp.asnumpy(quality)
                else:
                    sigma_sq_phi = None
                    quality = None
        elif self.type_unwrap == 'multiwave':
            eq_wav12 = (self.pitch_list[-1] * self.pitch_list[1]) / (self.pitch_list[1] - self.pitch_list[-1])
            eq_wav123 = self.pitch_list[0] * eq_wav12 / (self.pitch_list[0] - eq_wav12)
            self.pitch_list = np.insert(self.pitch_list, 0, eq_wav123)
            self.pitch_list = np.insert(self.pitch_list, 2, eq_wav12)
            modulation_vector, orig_img, phase_map, mask = nstep.phase_cal(images_arr, 
                                                                           self.limit, 
                                                                           self.N_list,
                                                                           False)
            phase_wav12 = np.mod(phase_map[0] - phase_map[1], 2 * np.pi)
            phase_wav123 = np.mod(phase_wav12 - phase_map[2], 2 * np.pi)
            phase_wav123[phase_wav123 > TAU] = phase_wav123[phase_wav123 > TAU] - 2 * np.pi
            #unwrapped phase
            phase_arr = np.stack([phase_wav123, phase_map[2], phase_wav12, phase_map[1], phase_map[0]])
            unwrap_vector, k = nstep.multiwave_unwrap(self.pitch_list,
                                                      phase_arr,
                                                      self.kernel,
                                                      self.fringe_direc,
                                                      mask,
                                                      self.cam_width,
                                                      self.cam_height)
            self.mask = mask
            
        if os.path.exists(os.path.join(self.object_path, 'white.tiff')):
            inte_img = cv2.imread(os.path.join(self.object_path, 'white.tiff'))
            inte_rgb_image = inte_img[..., ::-1].copy()
        else:
            inte_rgb_image = orig_img
        obj_cordi, obj_color, cordi_sigma, = self.complete_recon(unwrap_vector,                                                
                                                                 inte_rgb_image,
                                                                 temperature_image,
                                                                 sigma_sq_phi,
                                                                 quality)
        
        return obj_cordi, obj_color, cordi_sigma
    
def undistort_point(xc_yc, camera_dist):
    r_sq = xc_yc[0]**2 + xc_yc[1]**2
    undist_point = xc_yc * (1 + camera_dist[0, 0] * r_sq + camera_dist[0, 1] * r_sq**2)
    undist_point[2] = 1
    return undist_point

def device_cord(world_cord, device_matrix, device_distortion, rotation_transl_matrix):
    device_cordinate_xyz = np.dot(rotation_transl_matrix, world_cord.T)
    device_xyz_norm = device_cordinate_xyz/device_cordinate_xyz[2]
    device_dist = undistort_point(device_xyz_norm, device_distortion)
    device_points = np.matmul(device_matrix, device_dist)
    device_uv = device_points[:-1]
    return device_points, device_uv

def reconst_test(savedir):
    
    pitch_list = [1375, 275, 55, 11]
    #savedir = r'test_data\reconst_toydata'
    calibration = np.load(os.path.join(savedir, 'multifreq_mean_calibration_param.npz'))
    proj_matrix = calibration['proj_mtx_mean']
    proj_dist = calibration["proj_dist_mean"]
    proj_cam_rotation = calibration['st_rmat_mean']
    proj_cam_trans = calibration['st_tvec_mean']
    camera_matrix = calibration['cam_mtx_mean']
    camera_dist = calibration["cam_dist_mean"]
    proj_rotation_trans_mtx = np.concatenate((proj_cam_rotation, proj_cam_trans), axis=1)
    cam_rot_trans_mtx = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
    # Point cloud data
    cordinates = np.load(os.path.join(savedir, "cloud_coordinates.npy"))
    color_index = np.load(os.path.join(savedir, "cloud_intensity.npy"))
    # Data obtained from forward calculation
    cam_white_stack = np.load(os.path.join(savedir, "cam_white.npy"))
    cam_unwrap = np.load(os.path.join(savedir, "cam_unwrap.npy"))
    proj_unwrap = np.load(os.path.join(savedir, "proj_unwrap.npy"))

    world_cord = np.concatenate((cordinates, np.ones((len(cordinates), 1))), axis=1)
    #world to device coordinates
    proj_point, proj_uv = device_cord(world_cord, proj_matrix, proj_dist, proj_rotation_trans_mtx)
    cam_point, cam_uv = device_cord(world_cord, camera_matrix, camera_dist, cam_rot_trans_mtx)

    plt.figure()
    plt.imshow(proj_unwrap, cmap='gray')
    plt.scatter(proj_uv[0, :], proj_uv[1, :], color='r', s=10)
    plt.title('Projector unwrap phase', fontsize=20)

    plt.figure()
    plt.imshow(cam_white_stack, cmap='gray')
    plt.scatter(cam_uv[0, :], cam_uv[1, :], color='r', s=10)
    plt.title('Camera unwrap phase', fontsize=20)
    #Intensity from camera image based on derived cloud coordinates
    cam_int = nstep.bilinear_interpolate(cam_white_stack/np.max(cam_white_stack), cam_uv[0, :], cam_uv[1, :])
    intensity_diff = np.diff(color_index[:, 0] - cam_int)
    #
    proj_phase = nstep.bilinear_interpolate(cam_unwrap, cam_uv[0, :], cam_uv[1, :])
    proj_uv_phase = proj_uv*(2*np.pi/pitch_list[-1])
    phase_dif = proj_phase - proj_uv_phase[0]

    plt.figure()
    plt.hist(intensity_diff, bins=5)
    plt.title("Intensity difference", fontsize=20)
    plt.xlabel("Count", fontsize=15)
    plt.show()
    plt.figure()
    plt.hist(phase_dif, bins=5)
    plt.title("Phase difference", fontsize=20)
    plt.xlabel("Count", fontsize=15)
    plt.show()
            
def main():
    
    print("\nPlease Choose")
    option = input("\n1:Reconstruction test \n2: 2 level reconstruction \n3: 3 level reconstruction \n4: 4 level reconstruction")
    if option == "1":
        savedir = input("Enter the path for data or enter None:")
        if savedir == "None":
            savedir = r'test_data\reconst_toydata'
        reconst_test(savedir)
        return
    elif option == "2":
        pitch_list =[1200, 18]
       # N_list = [3, 3]
        N_list = [3, 3]
    elif option == "3":
        pitch_list = [1200, 120, 12]
        N_list = [3, 3, 9]
    elif option == "4":
        pitch_list =[1375, 275, 55, 11] 
        N_list = [3, 3, 3, 9]
    else:
        print("ERROR: Invalid entry for number of levels")
        return
    limit = float(input("\nEnter background limit:"))
    save_option = input("\nDo you want to save as .ply?(y/n):")
    if save_option == "y":
        save_ply = True
    elif save_option == "n":
        save_ply = False
    else:
        print("\n ERROR: Invalid entry")
        return
    prob = input("Do you need a model with pixel uncertainty?(y/n):")
    prob_up = True
    if prob =="y":
        probability = True
        pup = input("1:Aleatoric uncertainty \n2: Full uncertainty:")
        if pup == "1":
            prob_up = True
        else:
            prob_up = False
    elif prob == "n":
        probability = False
    else:
        print("\n ERROR: Invalid uncertainity entry")
        return
    temp_option = input("Is temperature data available?(y/n):")
    if temp_option == "y":
        temp = True
    elif temp_option == "n":
        temp = False
    else:
        print("ERROR: Invalid entry")
        return
    proj_width = 912  
    proj_height = 1140 
    cam_width = 1920 
    cam_height = 1200
    type_unwrap = 'multifreq'
    dark_bias_path = r"C:\Users\kl001\Documents\pyfringe_test\mean_pixel_std\exp_30_fp_42_retake\black_bias\avg_dark.npy"
    #obj_path = r'C:\Users\kl001\Documents\grasshopper3_python\images'
    obj_path = r"E:\test2"
    calib_path = r"G:\My Drive\Epistemic_newdata\calibration_100"
    model_path = r"G:\My Drive\Epistemic_newdata\variance_model.npy"
    reconst_inst = Reconstruction(proj_width=proj_width,
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
                                  processing='gpu',
                                  dark_bias_path=dark_bias_path,
                                  calib_path=calib_path,
                                  object_path=obj_path,
                                  model_path=model_path,
                                  temp=temp,
                                  save_ply=save_ply,
                                  probability=probability,
                                  prob_up=prob_up)
    
    obj_cordi, obj_color, cordi_sigma = reconst_inst.obj_reconst_wrapper()
    # np.save(os.path.join(obj_path,"accuracy_corrected_cord_std.npy"),cordi_sigma)
    np.save(os.path.join(obj_path,"accuracy_corrected_cord_mean.npy"),obj_cordi)
    np.save(os.path.join(obj_path,"accuracy_corrected_mask.npy"),reconst_inst.mask)
    return


if __name__ == '__main__':
    main()
    
        