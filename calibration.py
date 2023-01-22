#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cupy as cp
import nstep_fringe as nstep
import nstep_fringe_cp as nstep_cp
import cv2
import os
import glob
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import reconstruction as rc
from plyfile import PlyData, PlyElement
from scipy.optimize import leastsq
from scipy.spatial import distance
from copy import deepcopy
import shutil

EPSILON = -0.5
TAU = 5.5
class Calibration:
    """
    Calibration class is used to calibrate camera and projector setting. User can choose between phase coded , multifrequency and multiwavelength temporal unwrapping.
    After calibration the camera and projector parameters are saved as npz file at the given calibration image path.
    """
    def __init__(self,
                 proj_width,
                 proj_height,
                 cam_width,
                 cam_height,
                 no_pose,
                 acquisition_index,
                 mask_limit,
                 type_unwrap,
                 N_list,
                 pitch_list,
                 board_gridrows,
                 board_gridcolumns,
                 dist_betw_circle,
                 bobdetect_areamin,
                 bobdetect_convexity,
                 kernel_v,
                 kernel_h,
                 path,
                 data_type,
                 processing):
        """
        Parameters
        ----------
        proj_width: int.
                    Width of projector.
        proj_height: int.
                     Height of projector.
        cam_width: int.
                   Width of camera.
        cam_height: int.
                    Height of camera.
        mask_limit: float.
                    Modulation limit for applying mask to captured images.
        no_pose: int.
                 Number of calibration poses.
        acquisition_index: int.
                           Starting acquisition index
        type_unwrap: string.
                     Type of temporal unwrapping to be applied.
                     'phase' = phase coded unwrapping method,
                     'multifreq' = multi frequency unwrapping method
                     'multiwave' = multi wavelength unwrapping method.
        N_list: list.
                The number of steps in phase shifting algorithm.
                If phase coded unwrapping method is used this is a single element list.
                For other methods corresponding to each pitch one element in the list.
        pitch_list: list.
                    Array of number of pixels per fringe period.
        board_gridrows: int.
                        Number of rows in the asymmetric circle pattern.
        board_gridcolumns: int.
                           Number of columns in the asymmetric circle pattern.
        dist_betw_circle: float.
                          Distance between circle centers.
        kernel_v: int.
                  Kernel for vertical filter
        kernel_h: int.
                  Kernel for horizontal filter.
        path: str.
              Path to read captured calibration images and save calibration results.
        data_type:str
                  Calibration image data can be either .jpeg or .npy.
        processing:str.
                   Type of data processing. Use 'cpu' for desktop computation and 'gpu' for gpu.

        """
        self.proj_width = proj_width
        self.proj_height = proj_height
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.no_pose = no_pose
        self.acquisition_index = acquisition_index
        self.limit = mask_limit
        self.type_unwrap = type_unwrap
        self.N = N_list
        self.pitch = pitch_list
        self.path = path
        self.board_gridrows = board_gridrows
        self.board_gridcolumns = board_gridcolumns
        self.dist_betw_circle = dist_betw_circle
        self.bobdetect_areamin = bobdetect_areamin
        self.bobdetect_convexity = bobdetect_convexity
        self.kernel_v = kernel_v
        self.kernel_h = kernel_h
        if self.type_unwrap == 'phase':
            self.phase_st = -np.pi
        elif (self.type_unwrap == 'multifreq') or (self.type_unwrap == 'multiwave'):
            self.phase_st = 0
        else:
            print('ERROR: Invalid type_unwrap')
            return
        if not os.path.exists(self.path):
            print('ERROR: %s does not exist' % self.path)
        if (data_type != 'jpeg') and (data_type != 'npy'):
            print('ERROR: Invalid data type. Data type should be \'jpeg\' or \'npy\'')
        else:
            self.data_type = data_type
        if (processing != 'cpu') and (processing != 'gpu'):
            print('ERROR: Invalid processing type. Processing type should be \'cpu\' or \'gpu\'')
        else:
            self.processing = processing
        
    def calib(self, fx, fy):
        """
        Function to calibrate camera and projector and save npz file of calibration parameter based on user choice 
        of temporal phase unwrapping.
        Returns
        -------
        unwrapv_lst: list.
                     List of unwrapped phase maps obtained from horizontally varying intensity patterns.
        unwraph_lst: list.
                    List of unwrapped phase maps obtained from vertically varying intensity patterns..
        white_lst: list.
                  List of true images for each calibration pose.
        mod_lst: list.
                 List of modulation intensity images for each calibration pose for intensity
                 varying both horizontally and vertically.
        proj_img_lst: list.
                      List of projector images.
        cam_objpts: list.
                    List of world coordinates used for camera calibration for each pose.
        cam_imgpts: list.
                    List of circle centers grid for each calibration pose.
        proj_imgpts: list.
                     List of circle center grid coordinates for each pose of projector calibration.
        euler_angles : np.array.
                       Array of roll,pitch and yaw angles between camera and projector in degrees.
        cam_mean_error: list.
                        List of camera mean error per calibration pose.
        cam_delta: np.ndarray.
                   Array of camera re projection error.
        cam_df1: pandas dataframe.
                 Dataframe of camera absolute error in x and y directions of all poses.
        proj_mean_error: list.
                         List of projector mean error per calibration pose.
        proj_delta: np.ndarray.
                    Array of projector re projection error.
        proj_df1: pandas dataframe.
                  Dataframe of projector absolute error in x and y directions of all poses.
        """
        objp = self.world_points()
        if self.type_unwrap == 'phase':
            unwrapv_lst, unwraph_lst, white_lst, avg_lst, mod_lst, wrapped_phase_lst = self.projcam_calib_img_phase()
        elif self.type_unwrap == 'multifreq':
            unwrapv_lst, unwraph_lst, white_lst, avg_lst, mod_lst, wrapped_phase_lst = self.projcam_calib_img_multifreq()
        elif self.type_unwrap == 'multiwave':
            unwrapv_lst, unwraph_lst, white_lst, avg_lst, mod_lst, wrapped_phase_lst = self.projcam_calib_img_multiwave()
            
        # Projector images
        proj_img_lst = self.projector_img(unwrapv_lst, unwraph_lst, white_lst, fx, fy)
        # Camera calibration
        camr_error, cam_objpts, cam_imgpts, cam_mtx, cam_dist, cam_rvecs, cam_tvecs = self.camera_calib(objp, white_lst)
        
        # Projector calibration
        proj_ret, proj_imgpts, proj_mtx, proj_dist, proj_rvecs, proj_tvecs = self.proj_calib(cam_objpts,
                                                                                             cam_imgpts,
                                                                                             unwrapv_lst,
                                                                                             unwraph_lst,
                                                                                             proj_img_lst)
        # Camera calibration error analysis
        cam_mean_error, cam_delta, cam_df1 = self.intrinsic_error_analysis(cam_objpts, 
                                                                           cam_imgpts, 
                                                                           cam_mtx, 
                                                                           cam_dist, 
                                                                           cam_rvecs, 
                                                                           cam_tvecs)
        # Projector calibration error analysis
        proj_mean_error, proj_delta, proj_df1 = self.intrinsic_error_analysis(cam_objpts,
                                                                              proj_imgpts,
                                                                              proj_mtx,
                                                                              proj_dist,
                                                                              proj_rvecs,
                                                                              proj_tvecs)
        # Stereo calibration
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.0001)
        stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC+cv2.CALIB_ZERO_TANGENT_DIST+cv2.CALIB_FIX_K3+cv2.CALIB_FIX_K4+cv2.CALIB_FIX_K5+cv2.CALIB_FIX_K6

        st_retu, st_cam_mtx, st_cam_dist, st_proj_mtx, st_proj_dist, st_cam_proj_rmat, st_cam_proj_tvec, E, F = cv2.stereoCalibrate(cam_objpts,
                                                                                                                                    cam_imgpts,
                                                                                                                                    proj_imgpts,
                                                                                                                                    cam_mtx,
                                                                                                                                    cam_dist,
                                                                                                                                    proj_mtx,
                                                                                                                                    proj_dist,
                                                                                                                                    white_lst[0].shape[::-1],
                                                                                                                                    flags=stereocalibration_flags,
                                                                                                                                    criteria=criteria)
        project_mat = np.hstack((st_cam_proj_rmat, st_cam_proj_tvec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(project_mat)
        
        np.savez(os.path.join(self.path, '{}_calibration_param.npz'.format(self.type_unwrap)), st_cam_mtx, st_cam_dist, st_proj_mtx, st_cam_proj_rmat, st_cam_proj_tvec)
        np.savez(os.path.join(self.path, '{}_cam_rot_tvecs.npz'.format(self.type_unwrap)), cam_rvecs, cam_tvecs)
        
        return unwrapv_lst, unwraph_lst, white_lst, mod_lst, proj_img_lst, cam_objpts, cam_imgpts, proj_imgpts, euler_angles, cam_mean_error, cam_delta, cam_df1, proj_mean_error, proj_delta, proj_df1
    
    def update_list_calib(self, proj_df1, unwrapv_lst, unwraph_lst, white_lst, mod_lst, proj_img_lst, reproj_criteria):
        """
        Function to remove outlier calibration poses.

        Parameters
        ----------
        proj_df1: pandas dataframe.
                  Dataframe of projector absolute error in x and y directions of all poses.
        unwrapv_lst: list.
                     List of unwrapped phase maps obtained from horizontally varying intensity patterns.
        unwraph_lst:list.
                    List of unwrapped phase maps obtained from vertically varying intensity patterns.
        white_lst:list.
                  List of true images for each calibration pose.
        mod_lst: list.
                 List of modulation intensity images for each calibration pose for intensity
                 varying both horizontally and vertically.
        proj_img_lst: list.
                      List of circle center grid coordinates for each pose of projector calibration.
        reproj_criteria: float.
                         Criteria to remove outlier poses.
        Returns
        -------
        up_unwrapv_lst: list.
                        Updated list of unwrapped phase maps obtained from horizontally
                        varying intensity patterns.
        up_unwraph_lst: list of float.
                        Updated list of unwrapped phase maps obtained from vertically
                        varying intensity patterns.
        up_white_lst: list.
                      Updated list of true images for each calibration pose.
        up_mod_lst: list.
                    Updated list of modulation intensity images for each
                    calibration pose for intensity varying both horizontally and vertically.
        up_proj_img_lst: list.
                         Updated list of modulation intensity images for each
                         calibration pose for intensity varying both horizontally and vertically.
        cam_objpts: list.
                    Updated list of world coordinates used for camera calibration for each pose.
        cam_imgpts: list.
                    Updated list of circle centers grid for each calibration pose.
        proj_imgpts: float.
                     Updated list of circle center grid coordinates for each pose of projector calibration.
        euler_angles:float.
                     Array of roll,pitch and yaw angles between camera and projector in degrees
        cam_mean_error: list.
                        List of camera mean error per calibration pose.
        cam_delta: list.
                   List of camera re projection error.
        cam_df1: pandas dataframe.
                 Dataframe of camera absolute error in x and y directions
                 of updated list of poses.
        proj_mean_error: list. List of projector mean error per calibration pose.
        proj_delta: list. List of projector re projection error.
        proj_df1: pandas dataframe.
                  Dataframe of projector absolute error in x and y directions
                  of updated list of poses.
        """
        up_lst = list(set(proj_df1[proj_df1['absdelta_x'] > reproj_criteria]['image'].to_list() + proj_df1[proj_df1['absdelta_y'] > reproj_criteria]['image'].to_list()))
        up_white_lst = []
        up_unwrapv_lst = []
        up_unwraph_lst = []
        up_mod_lst = []
        up_proj_img_lst = []
        for index, element in enumerate(white_lst):
            if index not in up_lst:
                up_white_lst.append(element)
                up_unwrapv_lst.append(unwrapv_lst[index])
                up_unwraph_lst.append(unwraph_lst[index])
                up_mod_lst.append(mod_lst[index])
                up_proj_img_lst.append(proj_img_lst[index])
        objp = self.world_points()
        camr_error, cam_objpts, cam_imgpts, cam_mtx, cam_dist, cam_rvecs, cam_tvecs = self.camera_calib(objp, up_white_lst)
        
        # Projector calibration
        proj_ret, proj_imgpts, proj_mtx, proj_dist, proj_rvecs, proj_tvecs = self.proj_calib(cam_objpts,
                                                                                             cam_imgpts,
                                                                                             up_unwrapv_lst,
                                                                                             up_unwraph_lst,
                                                                                             up_proj_img_lst)
        # Camera calibration error analysis
        cam_mean_error, cam_delta, cam_df1 = self.intrinsic_error_analysis(cam_objpts,
                                                                           cam_imgpts,
                                                                           cam_mtx,
                                                                           cam_dist,
                                                                           cam_rvecs,
                                                                           cam_tvecs)
        # Projector calibration error analysis
        proj_mean_error, proj_delta, proj_df1 = self.intrinsic_error_analysis(cam_objpts,
                                                                              proj_imgpts,
                                                                              proj_mtx,
                                                                              proj_dist,
                                                                              proj_rvecs,
                                                                              proj_tvecs)
        # Stereo calibration
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.0001)
        stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC+cv2.CALIB_ZERO_TANGENT_DIST+cv2.CALIB_FIX_K3+cv2.CALIB_FIX_K4+cv2.CALIB_FIX_K5+cv2.CALIB_FIX_K6

        st_retu, st_cam_mtx, st_cam_dist, st_proj_mtx, st_proj_dist, st_cam_proj_rmat, st_cam_proj_tvec, E, F = cv2.stereoCalibrate(cam_objpts,
                                                                                                                                    cam_imgpts,
                                                                                                                                    proj_imgpts,
                                                                                                                                    cam_mtx,
                                                                                                                                    cam_dist,
                                                                                                                                    proj_mtx,
                                                                                                                                    proj_dist,
                                                                                                                                    white_lst[0].shape[::-1],
                                                                                                                                    flags=stereocalibration_flags,
                                                                                                                                    criteria=criteria)
        project_mat = np.hstack((st_cam_proj_rmat, st_cam_proj_tvec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(project_mat)
        
        np.savez(os.path.join(self.path, '{}_calibration_param.npz'.format(self.type_unwrap)), st_cam_mtx, st_cam_dist, st_proj_mtx, st_cam_proj_rmat, st_cam_proj_tvec)
        np.savez(os.path.join(self.path, '{}_cam_rot_tvecs.npz'.format(self.type_unwrap)), cam_rvecs, cam_tvecs)
        return up_unwrapv_lst, up_unwraph_lst, up_white_lst, up_mod_lst, up_proj_img_lst, cam_objpts, cam_imgpts, proj_imgpts, euler_angles, cam_mean_error, cam_delta, cam_df1, proj_mean_error, proj_delta, proj_df1 

    def calib_center_reconstruction(self, cam_imgpts, unwrap_phase):
        """
        This function is a wrapper function to reconstruct circle centers for each camera pose and compute error 
        with computed world projective coordinates in camera coordinate system.

        Parameters
        ----------
        cam_imgpts: list.
                    List of detected circle centers in camera images.
        unwrap_phase: np.ndarray.
                      Unwrapped phase map

        Returns
        -------
        delta_df: pandas dataframe.
                  Data frame of error for each pose all circle centers.
        abs_delta_df: pandas dataframe.
                      Data frame of absolute error for each pose all circle centers.
        center_cordi_lst: list.
                          List of x,y,z coordinates of detected circle centers in each calibration pose.

        """
        calibration = np.load(os.path.join(self.path, '{}_calibration_param.npz'.format(self.type_unwrap)))
        c_mtx = calibration["arr_0"]
        c_dist = calibration["arr_1"]
        p_mtx = calibration["arr_2"]
        cp_rot_mtx = calibration["arr_3"]
        cp_trans_mtx = calibration["arr_4"]
        vectors = np.load(os.path.join(self.path, '{}_cam_rot_tvecs.npz'.format(self.type_unwrap)))
        rvec = vectors["arr_0"]
        tvec = vectors["arr_1"]
        # Function call to get all circle center x,y,z coordinates
        center_cordi_lst = self.center_xyz(cam_imgpts, 
                                           unwrap_phase, 
                                           c_mtx, 
                                           c_dist, 
                                           p_mtx, 
                                           cp_rot_mtx, 
                                           cp_trans_mtx)
        true_coordinates = self.world_points()
        
        # Function call to get projective xyz for each pose
        proj_xyz_arr = self.project_validation(rvec, tvec, true_coordinates)
        # Error dataframes
        delta_df, abs_delta_df = self.center_err_analysis(center_cordi_lst, proj_xyz_arr)
        
        return delta_df, abs_delta_df, center_cordi_lst

    def world_points(self):
        """
        Function to generate world coordinate for asymmetric circle center calibration of camera and projector.
        Returns
        -------
        coord = type: float. Array of world coordinates.
        """
        col1 = np.append(np.tile([0, 0.5], int((self.board_gridcolumns-1) / 2)), 0).reshape(self.board_gridcolumns, 1)
        col2 = np.ones((self.board_gridcolumns, self.board_gridrows)) * np.arange(0, self.board_gridrows)
        col_mat = col1 + col2
    
        row_mat = (0.5 * np.arange(0, self.board_gridcolumns).reshape(self.board_gridcolumns, 1))@np.ones((1, self.board_gridrows))
        zer = np.zeros((self.board_gridrows * self.board_gridcolumns))
        coord = np.column_stack((row_mat.ravel(), col_mat.ravel(), zer)) * self.dist_betw_circle
        return coord.astype('float32')

    def projcam_calib_img_phase(self):
        """
        Function is used to generate absolute phase maps and true (single channel gray) images 
        (object image without fringe patterns) from fringe image for camera and projector calibration 
        from raw captured images using phase coded temporal unwrapping method.
        The function generates 'no_pose' number of absolute phase maps and true images.
        Returns
        -------
        unwrapv_lst: list.
                     List of unwrapped phase maps obtained from horizontally varying intensity patterns.
        unwraph_lst: list.
                     List of unwrapped phase maps obtained from vertically varying intensity patterns.
        white_lst: list.
                   List of true images for each calibration pose.
        avg_lst: list.
                  List of average intensity images for each calibration pose.
        mod_lst: list.
                  List of modulation intensity images for each calibration pose.
        wrapped_phase_lst: dictionary.
                            List of vertical and horizontal phase maps
        """
        unwrap_v_lst = []
        unwrap_h_lst = []
        white_lst = []
        avg_lst = []
        mod_lst = []
        kv_lst = []
        kh_lst = []
        coswrapv_lst = []
        coswraph_lst = []
        stepwrapv_lst = []
        stepwraph_lst = []
        delta_deck_lst, delta_index = self.delta_deck_calculation()
        for x in tqdm(range(self.acquisition_index, (self.acquisition_index + self.no_pose)), desc='generating unwrapped phases map for {} images'.format(self.no_pose)):
            if os.path.exists(os.path.join(self.path, 'capt_%d_0.jpg' % x)):
                # Read and apply mask to each captured images for cosine and stair patterns
                img_path = sorted(glob.glob(os.path.join(self.path, 'capt_%d_*.jpg' % x)), key=os.path.getmtime)
                images_arr = np.array([cv2.imread(file, 0) for file in img_path]).astype(np.float64)
                cos_v_int8,  mod1, avg1, phase_cosv = nstep.phase_cal(images_arr[0:self.N[0]], delta_deck_lst[0], self.limit)
                cos_h_int8,  mod2, avg2, phase_cosh = nstep.phase_cal(images_arr[self.N[0]:2*self.N[0]], delta_deck_lst[0], self.limit)
                step_v_int8, mod3, avg3, phase_stepv = nstep.phase_cal(images_arr[2*self.N[0]:3*self.N[0]], delta_deck_lst[0], self.limit)
                step_h_int8, mod4, avg4, phase_steph = nstep.phase_cal(images_arr[3*self.N[0]:4*self.N[0]], delta_deck_lst[0], self.limit)
                unwrap_v, unwrap_h, k0_v, k0_h, cos_wrap_v, cos_wrap_h, step_wrap_v, step_wrap_h = nstep.ph_temp_unwrap(phase_cosv, 
                                                                                                                        phase_cosh,
                                                                                                                        phase_stepv, 
                                                                                                                        phase_steph,
                                                                                                                        self.pitch[-1], 
                                                                                                                        self.proj_height, 
                                                                                                                        self.proj_width,
                                                                                                                        self.kernel_v, 
                                                                                                                        self.kernel_h)
                # True image for a given pose  
                orig_img = avg2 + mod2
                unwrap_v_lst.append(unwrap_v)
                unwrap_h_lst.append(unwrap_h)
                white_lst.append(orig_img)
                avg_lst.append(np.array([avg1, avg2, avg3, avg4]))
                mod_lst.append(np.array([mod1, mod2, mod3, mod4]))
                kv_lst.append(k0_v)
                kh_lst.append(k0_h)
                coswrapv_lst.append(cos_wrap_v)
                coswraph_lst.append(cos_wrap_h)
                stepwrapv_lst.append(step_wrap_v)
                stepwraph_lst.append(step_wrap_h)

        wrapped_phase_lst = {"wrapv": coswrapv_lst,
                             "wraph": coswraph_lst,
                             "stepwrapv": stepwrapv_lst,
                             "stepwraph": stepwraph_lst}
        return unwrap_v_lst, unwrap_h_lst, white_lst, avg_lst, mod_lst, wrapped_phase_lst
    
    def delta_deck_calculation(self):
        """
        Function computes phase shift δ  values used in N-step phase shifting algorithm for each unique N values
        compatible to type of data processing chosen.
        Returns
        -------
        delta_deck_lst: list.
                        List of delta arrays for each unique N values.
        delta_index: list.
                     List indicating which delta_deck to use.
        """
        unique_N_list = list(dict.fromkeys(self.N))
        delta_deck_lst = []
        for n in unique_N_list:
            if self.processing == 'cpu':
                delta_deck = nstep.delta_deck_gen(n, self.cam_height, self.cam_width)
            else:
                delta_deck = nstep_cp.delta_deck_gen_cp(n, self.cam_height, self.cam_width)
            delta_deck_lst.append(delta_deck)
        delta_index = []
        N_list_array = np.array(self.N)
        for i, n in enumerate(unique_N_list):
            count = np.sum(N_list_array == n)
            delta_index.extend([i] * count)
        return delta_deck_lst, delta_index
    
    def multifreq_analysis(self, data_array, delta_deck_lst, delta_index):
        """
        Helper function to compute unwrapped phase maps using multi frequency unwrapping on CPU.
        Parameters
        ----------
        data_array: np.ndarray:float64.
                    Array of images used in 4 level phase unwrapping.
        delta_deck_lst: List of np.ndarray
                        Delta images for each N.
        delta_index: list.
                     List indicating which delta_deck to use.
        Returns
        -------
        multifreq_unwrap_v: np.ndarray.
                            Unwrapped phase map for vertical fringes.
        multifreq_unwrap_h: np.ndarray.
                            Unwrapped phase map for horizontal fringes.
        phase_arr_v: list.
                     Wrapped phase maps for each pitch in vertical direction.
        phase_arr_h: list.
                     Wrapped phase maps for each pitch in horizontal direction.
        orig_img: np.ndarray.
                  True image without fringes.
        avg_arr: np.ndarray.
                 Average intensity image of each pitch.
        mod_arr: np.ndarray.
                 Modulation intensity image of each pitch.
        """
        multi_cos_v_int1, multi_mod_v1, multi_avg_v1, multi_phase_v1 = nstep.phase_cal(data_array[0:self.N[0]],
                                                                                       delta_deck_lst[delta_index[0]], 
                                                                                       self.limit)
        multi_cos_h_int1, multi_mod_h1, multi_avg_h1, multi_phase_h1 = nstep.phase_cal(data_array[self.N[0]:2 * self.N[0]], 
                                                                                       delta_deck_lst[delta_index[0]],
                                                                                       self.limit)
        multi_cos_v_int2, multi_mod_v2, multi_avg_v2, multi_phase_v2 = nstep.phase_cal(data_array[2 * self.N[0]:2 * self.N[0] + self.N[1]],
                                                                                       delta_deck_lst[delta_index[1]], 
                                                                                       self.limit)
        multi_cos_h_int2, multi_mod_h2, multi_avg_h2, multi_phase_h2 = nstep.phase_cal(data_array[2 * self.N[0] + self.N[1]:2 * self.N[0] + 2 * self.N[1]],
                                                                                       delta_deck_lst[delta_index[1]], 
                                                                                       self.limit)
        multi_cos_v_int3, multi_mod_v3, multi_avg_v3, multi_phase_v3 = nstep.phase_cal(data_array[2 * self.N[0] + 2 * self.N[1]:2 * self.N[0] + 2 * self.N[1] + self.N[2]],
                                                                                       delta_deck_lst[delta_index[2]], 
                                                                                       self.limit)
        multi_cos_h_int3, multi_mod_h3, multi_avg_h3, multi_phase_h3 = nstep.phase_cal(data_array[2 * self.N[0] + 2 * self.N[1] + self.N[2]:2 * self.N[0] + 2 * self.N[1] + 2 * self.N[2]],
                                                                                       delta_deck_lst[delta_index[2]], 
                                                                                       self.limit)
        multi_cos_v_int4, multi_mod_v4, multi_avg_v4, multi_phase_v4 = nstep.phase_cal(data_array[2 * self.N[0] + 2 * self.N[1] + 2 * self.N[2]:2 * self.N[0] + 2 * self.N[1] + 2*self.N[2]+self.N[3]], 
                                                                                       delta_deck_lst[delta_index[3]], 
                                                                                       self.limit)
        multi_cos_h_int4, multi_mod_h4, multi_avg_h4, multi_phase_h4 = nstep.phase_cal(data_array[2 * self.N[0] + 2 * self.N[1] + 2*self.N[2] + self.N[3]: 2 * self.N[0] + 2 * self.N[1] + 2*self.N[2] + 2 * self.N[3]], 
                                                                                       delta_deck_lst[delta_index[3]],
                                                                                       self.limit)
        
        orig_img = multi_avg_h4 + multi_mod_h4

        multi_phase_v1[multi_phase_v1 < EPSILON] = multi_phase_v1[multi_phase_v1 < EPSILON] + 2 * np.pi
        multi_phase_h1[multi_phase_h1 < EPSILON] = multi_phase_h1[multi_phase_h1 < EPSILON] + 2 * np.pi
        
        phase_arr_v = [multi_phase_v1, multi_phase_v2, multi_phase_v3, multi_phase_v4]
        phase_arr_h = [multi_phase_h1, multi_phase_h2, multi_phase_h3, multi_phase_h4]
        
        multifreq_unwrap_v, k_arr_v = nstep.multifreq_unwrap(self.pitch, phase_arr_v, self.kernel_v, 'v')
        multifreq_unwrap_h, k_arr_h = nstep.multifreq_unwrap(self.pitch, phase_arr_h, self.kernel_h, 'h')                
       
        avg_arr = np.array([multi_avg_v1, multi_avg_v2, multi_avg_v3, multi_avg_v4, multi_avg_h1, multi_avg_h2, multi_avg_h3, multi_avg_h4])
        mod_arr = np.array([multi_mod_v1, multi_mod_v2, multi_mod_v3, multi_mod_v4, multi_mod_h1, multi_mod_h2, multi_mod_h3, multi_mod_h4])
        
        return multifreq_unwrap_v, multifreq_unwrap_h, phase_arr_v, phase_arr_h, orig_img, avg_arr, mod_arr
    
    def multifreq_analysis_cupy(self, data_array, delta_deck_lst, delta_index):
        """
        Helper function to compute unwrapped phase maps using multi frequency unwrapping on GPU.
        After computation all arrays are returned as numpy.
        Parameters
        ----------
        data_array: cp.ndarray:float64.
                    Cupy array of images used in 4 level phase unwrapping.
        delta_deck_lst: List of cp.ndarray
                       Delta images for each N.
        delta_index: list.
                    List indicating which delta_deck to use.
        Returns
        -------
        multifreq_unwrap_v: np.ndarray.
                            Unwrapped phase map for vertical fringes.
        multifreq_unwrap_h: np.ndarray.
                            Unwrapped phase map for horizontal fringes.
        phase_arr_v: list.
                     Wrapped phase maps for each pitch in vertical direction.
        phase_arr_h: list.
                     Wrapped phase maps for each pitch in horizontal direction.
        orig_img: np.ndarray.
                  True image without fringes.
        avg_arr: np.ndarray.
                 Average intensity image of each pitch.
        mod_arr: np.ndarray.
                 Modulation intensity image of each pitch.
        """
        multi_cos_v_int1, multi_mod_v1, multi_avg_v1, multi_phase_v1 = nstep_cp.phase_cal_cp(data_array[0:self.N[0]], 
                                                                                             delta_deck_lst[delta_index[0]],
                                                                                             self.limit)
        multi_cos_h_int1, multi_mod_h1, multi_avg_h1, multi_phase_h1 = nstep_cp.phase_cal_cp(data_array[self.N[0]:2 * self.N[0]], 
                                                                                             delta_deck_lst[delta_index[0]], 
                                                                                             self.limit)
        multi_cos_v_int2, multi_mod_v2, multi_avg_v2, multi_phase_v2 = nstep_cp.phase_cal_cp(data_array[2 * self.N[0]:2 * self.N[0] + self.N[1]],
                                                                                             delta_deck_lst[delta_index[1]],
                                                                                             self.limit)
        multi_cos_h_int2, multi_mod_h2, multi_avg_h2, multi_phase_h2 = nstep_cp.phase_cal_cp(data_array[2 * self.N[0] + self.N[1]:2 * self.N[0] + 2 * self.N[1]], 
                                                                                             delta_deck_lst[delta_index[1]],
                                                                                             self.limit)
        multi_cos_v_int3, multi_mod_v3, multi_avg_v3, multi_phase_v3 = nstep_cp.phase_cal_cp(data_array[2 * self.N[0] + 2 * self.N[1]:2 * self.N[0] + 2 * self.N[1] + self.N[2]],
                                                                                             delta_deck_lst[delta_index[2]],
                                                                                             self.limit)
        multi_cos_h_int3, multi_mod_h3, multi_avg_h3, multi_phase_h3 = nstep_cp.phase_cal_cp(data_array[2 * self.N[0] + 2 * self.N[1] + self.N[2]:2 * self.N[0] + 2 * self.N[1] + 2 * self.N[2]],
                                                                                             delta_deck_lst[delta_index[2]],
                                                                                             self.limit)
        multi_cos_v_int4, multi_mod_v4, multi_avg_v4, multi_phase_v4 = nstep_cp.phase_cal_cp(data_array[2 * self.N[0] + 2 * self.N[1] + 2 * self.N[2]:2 * self.N[0] + 2 * self.N[1] + 2 * self.N[2] + self.N[3]],
                                                                                             delta_deck_lst[delta_index[3]],
                                                                                             self.limit)
        multi_cos_h_int4, multi_mod_h4, multi_avg_h4, multi_phase_h4 = nstep_cp.phase_cal_cp(data_array[2 * self.N[0] + 2 * self.N[1] + 2*self.N[2] + self.N[3]: 2 * self.N[0] + 2 * self.N[1] + 2 * self.N[2] + 2 * self.N[3]],
                                                                                             delta_deck_lst[delta_index[3]],
                                                                                             self.limit)
        
        orig_img = multi_avg_h4 + multi_mod_h4

        multi_phase_v1[multi_phase_v1 < EPSILON] = multi_phase_v1[multi_phase_v1 < EPSILON] + 2 * np.pi
        multi_phase_h1[multi_phase_h1 < EPSILON] = multi_phase_h1[multi_phase_h1 < EPSILON] + 2 * np.pi
        
        phase_arr_v = [multi_phase_v1, multi_phase_v2, multi_phase_v3, multi_phase_v4]
        phase_arr_h = [multi_phase_h1, multi_phase_h2, multi_phase_h3, multi_phase_h4]
        
        multifreq_unwrap_v, k_arr_v = nstep_cp.multifreq_unwrap_cp(self.pitch, 
                                                                   phase_arr_v, 
                                                                   self.kernel_v, 
                                                                   'v')
        multifreq_unwrap_h, k_arr_h = nstep_cp.multifreq_unwrap_cp(self.pitch, 
                                                                   phase_arr_h, 
                                                                   self.kernel_h, 'h')
       
        avg_arr = np.array([cp.asnumpy(multi_avg_v1), cp.asnumpy(multi_avg_v2),
                            cp.asnumpy(multi_avg_v3), cp.asnumpy(multi_avg_v4), 
                            cp.asnumpy(multi_avg_h1), cp.asnumpy(multi_avg_h2),
                            cp.asnumpy(multi_avg_h3), cp.asnumpy(multi_avg_h4)])
        mod_arr = np.array([cp.asnumpy(multi_mod_v1), cp.asnumpy(multi_mod_v2),
                            cp.asnumpy(multi_mod_v3), cp.asnumpy(multi_mod_v4), 
                            cp.asnumpy(multi_mod_h1), cp.asnumpy(multi_mod_h2), 
                            cp.asnumpy(multi_mod_h3), cp.asnumpy(multi_mod_h4)])
        
        return cp.asnumpy(multifreq_unwrap_v), cp.asnumpy(multifreq_unwrap_h), cp.asnumpy(cp.asarray(phase_arr_v)), cp.asnumpy(cp.asarray(phase_arr_h)), cp.asnumpy(orig_img), avg_arr, mod_arr

    def projcam_calib_img_multifreq(self):
        """
        Function is used to generate absolute phase map and true (single channel gray) images 
        (object image without fringe patterns)from fringe image for camera and projector calibration from raw captured
        images using multi frequency temporal unwrapping method.
        Returns
        -------
        unwrapv_lst: list.
                     List of unwrapped phase maps obtained from horizontally varying intensity patterns.
        unwraph_lst: list.
                     List of unwrapped phase maps obtained from vertically varying intensity patterns.
        white_lst: list.
                   List of true images for each calibration pose.
        avg_lst: list.
                  List of average intensity images for each calibration pose.
        mod_lst: list.
                  List of modulation intensity images for each calibration pose.
        wrapped_phase_lst: dictionary.
                           List of vertical and horizontal phase maps

        """
        
        avg_lst = []
        mod_lst = []
        white_lst = []
        wrapv_lst = []
        wraph_lst = []
        unwrapv_lst = []
        unwraph_lst = []
        delta_deck_lst, delta_index = self.delta_deck_calculation()

        for x in tqdm(range(self.acquisition_index, (self.acquisition_index + self.no_pose)),
                      desc='generating unwrapped phases map for {} pose'.format(self.no_pose)):

            if self.data_type == 'jpeg':
                if os.path.exists(os.path.join(self.path, 'capt_%d_0.jpg' % x)):
                    img_path = sorted(glob.glob(os.path.join(self.path, 'capt_%d_*.jpg' % x)), key=os.path.getmtime)
                    images_arr = np.array([cv2.imread(file, 0) for file in img_path]).astype(np.float64)
                else:
                    print("ERROR: path is not exist!")
                    images_arr = None
            elif self.data_type == 'npy':
                if os.path.exists(os.path.join(self.path, 'capt_%d_0.npy' % x)):
                    images_arr = np.load(os.path.join(self.path, 'capt_%d_0.npy' % x)).astype(np.float64)
                else:
                    print("ERROR: path is not exist!")
                    images_arr = None
            else:
                print("ERROR: data type is not supported, must be '.jpeg' or '.npy'.")
                images_arr = None

            if (self.processing == 'cpu') and (images_arr is not None):
                unwrap_v, unwrap_h, phase_arr_v, phase_arr_h, orig_img, avg_arr, mod_arr = self.multifreq_analysis(images_arr,
                                                                                                                   delta_deck_lst,
                                                                                                                   delta_index)
            elif (self.processing == 'gpu') and (images_arr is not None):
                unwrap_v, unwrap_h, phase_arr_v, phase_arr_h, orig_img, avg_arr, mod_arr = self.multifreq_analysis_cupy(images_arr,
                                                                                                                        delta_deck_lst,
                                                                                                                        delta_index)
                cp.get_default_memory_pool().free_all_blocks()
            else:
                unwrap_v = None
                unwrap_h = None
                phase_arr_v = None
                phase_arr_h = None
                orig_img = None
                avg_arr = None
                mod_arr = None
                if self.processing in {'cpu', 'gpu'}:
                    pass
                else:
                    print("ERROR: processing type is not supported, must be 'cpu' or 'gpu'.")

            avg_lst.append(avg_arr)
            mod_lst.append(mod_arr)
            white_lst.append(orig_img)
            wrapv_lst.append(phase_arr_v)
            wraph_lst.append(phase_arr_h)
            unwrapv_lst.append(unwrap_v)
            unwraph_lst.append(unwrap_h)

        if None in (avg_lst + mod_lst + white_lst + wrapv_lst + wraph_lst + unwrapv_lst + unwraph_lst):
            print("WARNING: Some computational results are None")

        wrapped_phase_lst = {"wrapv": wrapv_lst,
                             "wraph": wraph_lst}
        return unwrapv_lst, unwraph_lst, white_lst, avg_lst, mod_lst, wrapped_phase_lst

    def projcam_calib_img_multiwave(self):
        """
        Function is used to generate absolute phase map and true (single channel gray) images (object image without 
        fringe patterns) from fringe image for camera and projector calibration from raw captured images using 
        multiwave temporal unwrapping method.
        Returns
        -------
        unwrapv_lst: list.
                     List of unwrapped phase maps obtained from horizontally varying intensity patterns.
        unwraph_lst: list.
                     List of unwrapped phase maps obtained from vertically varying intensity patterns.
        white_lst: list.
                   List of true images for each calibration pose.
        avg_lst: list.
                  List of average intensity images for each calibration pose.
        mod_lst: list.
                  List of modulation intensity images for each calibration pose.
        wrapped_phase_lst: dictionary.
                            List of vertical and horizontal phase maps
        """
        
        avg_lst = []
        mod_lst = []
        white_lst = []
        kv_lst = []
        kh_lst = []
        wrapv_lst = []
        wraph_lst = []
        unwrapv_lst = []
        unwraph_lst = []
        pitch_arr = self.pitch  # there is manipulation of pitch
        N_arr = self.N
        eq_wav12 = (pitch_arr[-1] * pitch_arr[1]) / (pitch_arr[1]-pitch_arr[-1])
        eq_wav123 = pitch_arr[0] * eq_wav12 / (pitch_arr[0] - eq_wav12)
        
        pitch_arr = np.insert(pitch_arr, 0, eq_wav123)
        pitch_arr = np.insert(pitch_arr, 2, eq_wav12)
        delta_deck_lst, delta_index = self.delta_deck_calculation()
        for x in tqdm(range(self.acquisition_index, (self.acquisition_index + self.no_pose)), desc='generating unwrapped phases map for {} images'.format(self.no_pose)):
            if os.path.exists(os.path.join(self.path, 'capt_%d_0.jpg' % x)):
                img_path = sorted(glob.glob(os.path.join(self.path, 'capt_%d_*.jpg' % x)), key=os.path.getmtime)
                images_arr = np.array([cv2.imread(file, 0) for file in img_path]).astype(np.float64)
                multi_cos_v_int3, multi_mod_v3, multi_avg_v3, multi_phase_v3 = nstep.phase_cal(images_arr[0: N_arr[0]],
                                                                                               delta_deck_lst[delta_index[0]],
                                                                                               self.limit)
                multi_cos_h_int3, multi_mod_h3, multi_avg_h3, multi_phase_h3 = nstep.phase_cal(images_arr[N_arr[0]:2 * N_arr[0]],
                                                                                               delta_deck_lst[delta_index[0]],
                                                                                               self.limit)
                multi_cos_v_int2, multi_mod_v2, multi_avg_v2, multi_phase_v2 = nstep.phase_cal(images_arr[2 * N_arr[0]:2 * N_arr[0] + N_arr[1]],
                                                                                               delta_deck_lst[delta_index[1]],
                                                                                               self.limit)
                multi_cos_h_int2, multi_mod_h2, multi_avg_h2, multi_phase_h2 = nstep.phase_cal(images_arr[2 * N_arr[0] + N_arr[1]:2 * N_arr[0] + 2 * N_arr[1]],
                                                                                               delta_deck_lst[delta_index[1]],
                                                                                               self.limit)
                multi_cos_v_int1, multi_mod_v1, multi_avg_v1, multi_phase_v1 = nstep.phase_cal(images_arr[2 * N_arr[0] + 2 * N_arr[1]:2 * N_arr[0] + 2 * N_arr[1] + N_arr[2]],
                                                                                               delta_deck_lst[delta_index[2]],
                                                                                               self.limit)
                multi_cos_h_int1, multi_mod_h1, multi_avg_h1, multi_phase_h1 = nstep.phase_cal(images_arr[2 * N_arr[0] + 2 * N_arr[1] + N_arr[2]:2 * N_arr[0] + 2 * N_arr[1] + 2 * N_arr[2]],
                                                                                               delta_deck_lst[delta_index[2]],
                                                                                               self.limit)
               
                orig_img = multi_avg_h1 + multi_mod_h1
                
                multi_phase_v12 = np.mod(multi_phase_v1 - multi_phase_v2, 2 * np.pi)
                multi_phase_h12 = np.mod(multi_phase_h1 - multi_phase_h2, 2 * np.pi)
                multi_phase_v123 = np.mod(multi_phase_v12 - multi_phase_v3, 2 * np.pi)
                multi_phase_h123 = np.mod(multi_phase_h12 - multi_phase_h3, 2 * np.pi)
                
                multi_phase_v123[multi_phase_v123 > TAU] = multi_phase_v123[multi_phase_v123 > TAU] - 2 * np.pi
                multi_phase_h123[multi_phase_h123 > TAU] = multi_phase_h123[multi_phase_h123 > TAU] - 2 * np.pi                
                
                phase_arr_v = [multi_phase_v123, multi_phase_v3, multi_phase_v12, multi_phase_v2, multi_phase_v1]
                phase_arr_h = [multi_phase_h123, multi_phase_h3, multi_phase_h12, multi_phase_h2, multi_phase_h1]
                
                multiwav_unwrap_v, k_arr_v = nstep.multiwave_unwrap(pitch_arr, phase_arr_v, self.kernel_v, 'v')
                multiwav_unwrap_h, k_arr_h = nstep.multiwave_unwrap(pitch_arr, phase_arr_h, self.kernel_h, 'h')
                
                avg_lst.append(np.array([multi_avg_v3, multi_avg_v2, multi_avg_v1, multi_avg_h3, multi_avg_h2, multi_avg_h1]))
                mod_lst.append(np.array([multi_mod_v3, multi_mod_v2, multi_mod_v1, multi_mod_h3, multi_mod_h2, multi_mod_h1]))
                white_lst.append(orig_img)
                wrapv_lst.append(phase_arr_v)
                wraph_lst.append(phase_arr_h)
                kv_lst.append(k_arr_v)
                kh_lst.append(k_arr_h)
                unwrapv_lst.append(multiwav_unwrap_v)
                unwraph_lst.append(multiwav_unwrap_h)
        wrapped_phase_lst = {"wrapv": wrapv_lst,
                             "wraph": wraph_lst}
        return unwrapv_lst, unwraph_lst, white_lst, avg_lst, mod_lst, wrapped_phase_lst
    
    @staticmethod
    def _image_resize(image_lst, fx, fy):
        resize_img_lst = []
        for i in image_lst:
            resize_img_lst.append(cv2.resize(i, None, fx=fx, fy=fy))
        return resize_img_lst

    def projector_img(self, unwrap_v_lst, unwrap_h_lst, white_lst, fx, fy):
        """
        Function to generate projector image using absolute phase maps from horizontally and vertically varying patterns.
        Parameters
        ----------
        unwrap_v_lst: list.
                      List of unwrapped absolute phase map from horizontally varying pattern.
        unwrap_h_lst: list.
                      List of unwrapped absolute phase map from vertically varying pattern.
        white_lst: list.
                   List of true object image (without patterns).
        fx: float.
            Scale factor along the horizontal axis.
        fy: float.
            Scale factor along the vertical axis
        Returns
        -------
        proj_img: list.
                  Projector image list.
    
        """
        unwrap_v_lst = Calibration._image_resize(unwrap_v_lst, fx, fy)
        unwrap_h_lst = Calibration._image_resize(unwrap_h_lst, fx, fy)
        white_lst = Calibration._image_resize(white_lst, fx, fy)
        proj_img = []
        for i in tqdm(range(0, len(unwrap_v_lst)), desc='projector images'):
            # Convert phase map to coordinates
            unwrap_proj_u = (unwrap_v_lst[i] - self.phase_st) * self.pitch[-1] / (2 * np.pi)
            unwrap_proj_v = (unwrap_h_lst[i] - self.phase_st) * self.pitch[-1] / (2 * np.pi)
            unwrap_proj_u = unwrap_proj_u.astype(int)
            unwrap_proj_v = unwrap_proj_v.astype(int)
            
            orig_u = unwrap_proj_u.ravel()
            orig_v = unwrap_proj_v.ravel()
            orig_int = white_lst[i].ravel()
            orig_data = np.column_stack((orig_u, orig_v, orig_int))
            orig_df = pd.DataFrame(orig_data, columns=['u', 'v', 'int'])
            orig_new = orig_df.groupby(['u', 'v'])['int'].mean().reset_index()
            
            proj_y = np.arange(0, self.proj_height)
            proj_x = np.arange(0, self.proj_width)
            proj_u, proj_v = np.meshgrid(proj_x, proj_y)
            proj_data = np.column_stack((proj_u.ravel(), proj_v.ravel()))
            proj_df = pd.DataFrame(proj_data, columns=['u', 'v'])
    
            proj_df_merge = pd.merge(proj_df, orig_new, how='left', on=['u', 'v'])
            proj_df_merge['int'] = proj_df_merge['int'].fillna(0)
    
            proj_mean_img = proj_df_merge['int'].to_numpy()
            proj_mean_img = proj_mean_img.reshape(self.proj_height, self.proj_width)
            proj_mean_img = cv2.normalize(proj_mean_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            proj_img.append(proj_mean_img)
            
        return proj_img
    
    def camera_calib(self, objp, white_lst, display=True):
        """
        Function to calibrate camera using asymmetric circle pattern. 
        OpenCV bob detector is used to detect circle centers which is used for calibration.
    
        Parameters
        ----------
        objp: list.
              World object coordinate.
        white_lst: list.
                   List of calibration poses used for calibrations.
        display: bool.
                 If set each calibration drawings are displayed.
        Returns
        -------
        r_error: float.
                 Average re projection error.
        objpoints: list.
                   List of image object points for each pose.
        cam_imgpoints: list.
                       List of circle center grid coordinates for each pose.
        cam_mtx: np.array.
                 Camera matrix from calibration.
        cam_dist: np.array.
                  Camera distortion array from calibration.
        cam_rvecs: np.array.
                   Array of rotation vectors for each calibration pose.
        cam_tvecs: np.array.
                   Array of translational vectors for each calibration pose.
    
        """
        # Set bob detector properties
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        blobParams = cv2.SimpleBlobDetector_Params()
        
        # color
        blobParams.filterByColor = True
        blobParams.blobColor = 255
    
        # Filter by Area.
        blobParams.filterByArea = True
        blobParams.minArea = self.bobdetect_areamin  # 2000
        
        # Convexity
        blobParams.filterByConvexity = True
        blobParams.minConvexity = self.bobdetect_convexity
        
        blobDetector = cv2.SimpleBlobDetector_create(blobParams)
        objpoints = []  # 3d point in real world space
        cam_imgpoints = []  # 2d points in image plane.
        found = 0
        
        cv2.startWindowThread()
        count_lst = []
        ret_lst = []
        
        for i, white in enumerate(white_lst):
            # Convert float image to uint8 type image.
            white = cv2.normalize(white, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) 
            white_color = cv2.cvtColor(white, cv2.COLOR_GRAY2RGB)  # only for drawing purpose
            keypoints = blobDetector.detect(white)  # Detect blobs.
          
            # Draw detected blobs as green circles. This helps cv2.findCirclesGrid() .
            im_with_keypoints = cv2.drawKeypoints(white_color, keypoints, np.array([]), (0, 255, 0),
                                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                                  )
            im_with_keypoints_gray = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)
            
            ret, corners = cv2.findCirclesGrid(im_with_keypoints_gray, (self.board_gridrows, self.board_gridcolumns), None, 
                                               flags=cv2.CALIB_CB_ASYMMETRIC_GRID+cv2.CALIB_CB_CLUSTERING,
                                               blobDetector=blobDetector)  # Find the circle grid
            ret_lst.append(ret)
            
            if ret:
        
                objpoints.append(objp)  # Certainly, every loop objp is the same, in 3D.
                
                cam_imgpoints.append(corners)
                count_lst.append(found)
                found += 1
                if display:
                    # Draw and display the centers.
                    im_with_keypoints = cv2.drawChessboardCorners(white_color,
                                                                  (self.board_gridrows, self.board_gridcolumns),
                                                                  corners,
                                                                  ret)  # circles
                    cv2.imshow("Camera calibration", im_with_keypoints)  # display
                    cv2.waitKey(100)
    
        cv2.destroyAllWindows()
        if not all(ret_lst):
            print('Warning: Centers are not detected for some poses. Modify bobdetect_areamin and bobdetect_areamin parameter')
        # set flags to have tangential distortion = 0, k4 = 0, k5 = 0, k6 = 0
        flags = cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6
        # camera calibration
        cam_ret, cam_mtx, cam_dist, cam_rvecs, cam_tvecs = cv2.calibrateCamera(objpoints,
                                                                               cam_imgpoints,
                                                                               white_lst[0].shape[::-1],
                                                                               None,
                                                                               None,
                                                                               flags=flags,
                                                                               criteria=criteria)
       
        # Average re projection error
        tot_error = 0
        for i in range(len(objpoints)):
            cam_img2, _ = cv2.projectPoints(objpoints[i], cam_rvecs[i], cam_tvecs[i], cam_mtx, cam_dist)
            error = cv2.norm(cam_imgpoints[i], cam_img2, cv2.NORM_L2) / len(cam_img2)
            tot_error += error
        r_error = tot_error/len(objpoints)
        print("Re projection error:", r_error)
        return r_error, objpoints, cam_imgpoints, cam_mtx, cam_dist, cam_rvecs, cam_tvecs
    
    def proj_calib(self, 
                   cam_objpts, 
                   cam_imgpts, 
                   unwrap_v_lst, 
                   unwrap_h_lst, 
                   proj_img_lst=None):
        """
        Function to calibrate projector by using absolute phase maps. 
        Circle centers detected using OpenCV is mapped to the absolute phase maps and the corresponding projector image coordinate for the centers are calculated.
    
        Parameters
        ----------
        cam_objpts: list.
                    List of world coordinates used for camera calibration for each pose.
        cam_imgpts: list.
                    List of circle centers grid for each calibration pose.
        unwrap_v_lst: list.
                      List of absolute phase maps for horizontally varying patterns for
                      each calibration pose.
        unwrap_h_lst: list.
                      List of absolute phase maps for vertically varying patterns for
                      each calibration pose.
        proj_img_lst: list/None.
                      List of computed projector image for each calibration pose.
                      If it is None calibration drawing is not diaplayed.
        Returns
        -------
        r_error: float.
                 Average re projection error.
        proj_imgpts: list.
                     List of circle center grid coordinates for each pose of projector calibration.
        proj_mtx: np.array.
                  Projector matrix from calibration.
        proj_dist: np.array.
                   Projector distortion array from calibration.
        proj_rvecs: np.array.
                    Array of rotation vectors for each calibration pose.
        proj_tvecs: np.array.
                    Array of translational vectors for each calibration pose.
    
        """
        centers = [i.reshape(cam_objpts[0].shape[0], 2) for i in cam_imgpts]
        proj_imgpts = []
        for x, c in enumerate(centers):
            # Phase to coordinate conversion for each pose
            u = (nstep.bilinear_interpolate(unwrap_v_lst[x], c) - self.phase_st) * self.pitch[-1] / (2*np.pi)
            v = (nstep.bilinear_interpolate(unwrap_h_lst[x], c) - self.phase_st) * self.pitch[-1] / (2*np.pi)
            coordi = np.column_stack((u, v)).reshape(cam_objpts[0].shape[0], 1, 2).astype(np.float32)
            proj_imgpts.append(coordi)
            if proj_img_lst is not None:
                proj_color = cv2.cvtColor(proj_img_lst[x], cv2.COLOR_GRAY2RGB)  # only for drawing
                proj_keypoints = cv2.drawChessboardCorners(proj_color, (self.board_gridrows, self.board_gridcolumns), coordi, True)
                cv2.imshow("Projector calibration", proj_keypoints)  # display
                cv2.waitKey(100)
        cv2.destroyAllWindows()
        # Set all distortion = 0. linear model assumption
        flags = cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6
       
        # Projector calibration
        proj_ret, proj_mtx, proj_dist, proj_rvecs, proj_tvecs = cv2.calibrateCamera(cam_objpts,
                                                                                    proj_imgpts,
                                                                                    (self.proj_width, self.proj_height),
                                                                                    None,
                                                                                    None,
                                                                                    flags=flags,
                                                                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 2e-16))
        tot_error = 0
        # Average re projection error
        for i in range(len(cam_objpts)):
            proj_img2, _ = cv2.projectPoints(cam_objpts[i], proj_rvecs[i], proj_tvecs[i], proj_mtx, proj_dist)
            error = cv2.norm(proj_imgpts[i], proj_img2, cv2.NORM_L2) / len(proj_img2)
            tot_error += error
        r_error = tot_error / len(cam_objpts)
        print("Re projection error:", r_error)
        
        return r_error, proj_imgpts, proj_mtx, proj_dist, proj_rvecs, proj_tvecs
    
    def image_analysis(self, unwrap, title, aspect_ratio=1):
        """
        Function to plot list of images for calibration diagnostic purpose. Eg: To plot list of
        unwrapped phase maps of all calibration poses.
        Parameters
        ----------
        unwrap: list.
                List of images
        title: str.
               Title to be displayed on plots.
        aspect_ratio: float.
                      Aspect ratio for plotting.
        """
        for i in range(0, len(unwrap)):
            plt.figure()
            plt.imshow(unwrap[i], aspect=aspect_ratio)
            plt.title('{}'.format(title), fontsize=20)
        return

    def wrap_profile_analysis(self, wrapped_phase_lst, direc):
        """
        Function to plot cross-section of calculated wrapped phase map of cosine and stair patterns in 
        phase coded temporal unwrapping method for verification.
        Parameters
        ----------
        wrapped_phase_lst: dictionary of wrapped phase maps of cosine varying intensity pattern
        for all calibration poses.
        direc: str.
               vertical (v) or horizontal(h) patterns.
        """
        
        if self.type_unwrap == 'phase':        
            for i in range(0, len(wrapped_phase_lst['wrapv'])):
                if direc == 'v':
                    plt.figure()
                    plt.plot(wrapped_phase_lst['wrapv'][i][600])
                    plt.plot(wrapped_phase_lst['stepwrapv'][i][600])
                elif direc == 'h':
                    plt.figure()
                    plt.plot(wrapped_phase_lst['wraph'][i][:, 960])
                    plt.plot(wrapped_phase_lst['stepwraph'][i][:, 960])
                plt.title('Wrap phase map %s' % direc, fontsize=20)
                plt.xlabel('Dimension', fontsize=20)
                plt.ylabel('Phase', fontsize=20)
        elif self.type_unwrap in {'multifreq', 'multiwave'}:
            if direc == 'v':
                for phase_arr_v in wrapped_phase_lst['wrapv']:
                    plt.figure()
                    n_subplot = len(phase_arr_v)
                    for i, wrapped_phase in enumerate(phase_arr_v):
                        plt.subplot(n_subplot, 1, i+1)
                        plt.plot(wrapped_phase[600])
                    plt.title('Wrap phase map %s' % direc, fontsize=20)
                    plt.xlabel('Dimension', fontsize=20)
                    plt.ylabel('Phase', fontsize=20)
            if direc == 'h':
                for phase_arr_h in wrapped_phase_lst['wraph']:
                    plt.figure()
                    n_subplot = len(phase_arr_h)
                    for i, wrapped_phase in enumerate(phase_arr_h):
                        plt.subplot(n_subplot, 1, i+1)
                        plt.plot(wrapped_phase[:, 960])
                    plt.title('Wrap phase map %s' % direc, fontsize=20)
                    plt.xlabel('Dimension', fontsize=20)
                    plt.ylabel('Phase', fontsize=20)
        else:
            print('Unwrap method is not supported.')

    def intrinsic_error_analysis(self, objpts, imgpts, mtx, dist, rvecs, tvecs):
        """
        Function to calculate mean error per calibration pose,re projection errors and absolute
        re projection error in the x and y directions.
        Parameters
        ----------
        objpts: list.
                List of circle center grid coordinates for each pose of calibration.
        imgpts: list.
                List of circle centers grid for each calibration pose.
        mtx: np.array.
             Device matrix from calibration.
        dist: np.array.
              Device distortion matrix from calibration..
        rvecs: np.array.
               Array of rotation vectors for each calibration pose.
        tvecs: np.array.
               Array of translational vectors for each calibration pose.
        Returns
        -------
        mean_error: list.
                    List of mean error per calibration pose.
        delta_lst: list.
                   List of re projection error.
        abs_df: pandas dataframe.
                Dataframe of absolute error in x and y of all poses
        """
            
        delta_lst = []
        mean_error = []
        abs_error = []
        for i in range(len(objpts)):
            img2, _ = cv2.projectPoints(objpts[i], rvecs[i], tvecs[i], mtx, dist)
            delta = imgpts[i]-img2
            delta_lst.append(delta.reshape(objpts[i].shape[0], 2))
            error = cv2.norm(imgpts[i], img2, cv2.NORM_L2) / len(img2)
            mean_error.append(error)
            abs_error.append(abs(delta).reshape(objpts[i].shape[0], 2))
        
        abs_error = np.array(abs_error)
        df_a, df_b, df_c = abs_error.shape
        abs_df = pd.DataFrame(abs_error.reshape(df_a * df_b, df_c),
                              index=np.repeat(np.arange(df_a), df_b),
                              columns=['absdelta_x', 'absdelta_y'])
        abs_df = abs_df.reset_index().rename(columns={'index': 'image'})
        return mean_error, np.array(delta_lst), abs_df

    def intrinsic_errors_plts(self, mean_error, delta, df, dev):
        """
        Function to plot mean error per calibration pose, re projection error and absolute
        re projection errors in x and y directions.
        Parameters
        ----------
        mean_error: list.
                    List of mean error per calibration pose of a device.
        delta: list.
                List of re projection error in x and y directions .
        df: pandas dataframe.
            Dataframe of absolute re projection error for each calibration pose.
        dev: str.
             Device name: Camera or projector
        """
        xaxis = np.arange(0, len(mean_error), dtype=int)
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.bar(xaxis, mean_error)
        ax.set_title("{} mean error per pose ".format(dev), fontsize=30)
        ax.set_xlabel('Pose', fontsize=20)
        ax.set_ylabel('Pixel)', fontsize=20)
        ax.set_xticks(xaxis)
        plt.xticks(fontsize=15, rotation=45)
        plt.yticks(fontsize=20)
        plt.figure()
        plt.scatter(delta[:, :, 0].ravel(), delta[:, :, 1].ravel())
        plt.xlabel('x(pixel)', fontsize=30)
        plt.ylabel('y(pixel)', fontsize=30)
        # axes = plt.gca()
        # axes.set_aspect(1) #to set aspect equal
        plt.title('Re projection error for {} '.format(dev), fontsize=30)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.figure()
        sns.histplot(data=df, x='absdelta_x', hue='image', multiple='stack', palette='Paired', legend=False)
        plt.xlabel('Abs($\delta x$)', fontsize=30)
        plt.ylabel('Count', fontsize=30)
        plt.title('{} re projection error x direction '.format(dev), fontsize=30)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.figure()
        sns.histplot(data=df, x='absdelta_y', hue='image', multiple='stack', palette='Paired', legend=False)
        plt.xlabel('Abs($\delta y$)', fontsize=30)
        plt.ylabel('Count', fontsize=30)
        plt.title('{} re projection error y direction '.format(dev), fontsize=30)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        return

    # center reconstruction
    def center_xyz(self, center_pts, unwrap_phase, c_mtx, c_dist, p_mtx, cp_rot_mtx, cp_trans_mtx,):
        """
        Function to obtain 3d coordinates of detected circle centers.

        Parameters
        ----------
        center_pts: list.
                    List of circle centers of each calibration pose.
        unwrap_phase: list.
                      Unwrapped phase maps of each calibration pose.
        c_mtx: np.array.
               Camera matrix.
        c_dist: np.array.
                Camera distortion matrix.
        p_mtx: np.array.
               Projector matrix.
        cp_rot_mtx: np.array.
                    Camera projector rotation matrix.
        cp_trans_mtx: np.array.
                      Camera projector translational matrix.
        Returns
        -------
        center_cordi_lst: list.
                          Array of x,y,z coordinates of detected circle centers in each calibration pose.

        """
        center_cordi_lst = []
        for i in tqdm(range(0, len(center_pts)), desc='building camera centers 3d coordinates'):
            # undistort points
            x, y, z = rc.reconstruction_pts(center_pts[i], 
                                            unwrap_phase[i], 
                                            c_mtx, c_dist, 
                                            p_mtx, 
                                            cp_rot_mtx, cp_trans_mtx, 
                                            self.phase_st, self.pitch[-1])
            cordi = np.hstack((x, y, z))
            center_cordi_lst.append(cordi)
        return np.array(center_cordi_lst)

    # Projective coordinates based on camera - projector extrinsics
    def project_validation(self, rvec, tvec, true_coordinates):
        """
        Function to generate world projective coordinates in camera coordinate system for each 
        calibration pose using pose extrinsic.

        Parameters
        ----------
        rvec: list.
              List of rotation vectors for each calibration pose.
        tvec: list.
              List of translational vectors for each calibration pose.
        true_coordinates: list.
                         Defined world coordinates of circle centers.
        Returns
        -------
        proj_xyz_lst: list.
                      Array of projective coordinates each calibration poses.
        """
        t = np.ones((true_coordinates.shape[0], 1))
        homo_true_cordi = np.hstack((true_coordinates, t))
        homo_true_cordi = homo_true_cordi.reshape((homo_true_cordi.shape[0], homo_true_cordi.shape[1], 1))
        proj_xyz_lst = []
        for i in tqdm(range(0, len(tvec)), desc='projective centers'):
            rvec_mtx = cv2.Rodrigues(rvec[i])[0]
            h_vecs = np.hstack((rvec_mtx, tvec[i]))
            updated_hvecs = np.repeat(h_vecs[np.newaxis, :, :], true_coordinates.shape[0], axis=0)
            proj_xyz = updated_hvecs @ homo_true_cordi
            proj_xyz = proj_xyz.reshape((proj_xyz.shape[0], proj_xyz.shape[1]))
            proj_xyz_lst.append(proj_xyz)
        return np.array(proj_xyz_lst)

    # Calculate error = center reconstruction - projector coordinates
    def center_err_analysis(self, cordi_arr, proj_xyz_arr):
        """
        Function to compute error of 3d coordinates of detected circle centers from projective coordinates 
        in camera coordinate.

        Parameters
        ----------
        cordi_arr: np.array.
                  Array of x,y,z coordinates of detected circle centers in each calibration pose.
        proj_xyz_arr: np.array.
                      Array of projective coordinates for each calibration poses.
        Returns
        -------
        delta_df: pandas dataframe.
                  Data frame of error for each pose all circle centers.
        abs_delta_df: pandas dataframe.
                      Data frame of absolute error for each pose all circle centers.
        """
        delta = cordi_arr-proj_xyz_arr
        abs_delta = abs(cordi_arr-proj_xyz_arr)

        delta_df = pd.DataFrame([list(l) for l in delta]).stack().apply(pd.Series).reset_index(1, drop=True)
        delta_df.index.name = 'Poses'
        delta_df.columns = ['delta_x', 'delta_y', 'delta_z']

        abs_delta_df = pd.DataFrame([list(l) for l in abs_delta]).stack().apply(pd.Series).reset_index(1, drop=True)
        abs_delta_df.index.name = 'Poses'
        abs_delta_df.columns = ['abs$(\Delta x)$', 'abs$(\Delta y)$', 'abs$(\Delta z)$']

        g_delta = delta_df.groupby(delta_df.index)
        delta_group = [g_delta.get_group(x) for x in g_delta.groups]

        n_colors = len(delta_group)  # no. of poses
        cm = plt.get_cmap('gist_rainbow')
        fig = plt.figure(figsize=(16, 15))
        fig.suptitle(' Error in reconstructed coordinates compared to true coordinates ', fontsize=20)
        ax = plt.axes(projection='3d')
        ax.set_prop_cycle(color=[cm(1.*i/n_colors) for i in range(n_colors)])
        # cm = plt.get_cmap('gist_rainbow')
        for i in range(0, len(delta_group)):
            x1 = delta_group[i]['delta_x']
            y1 = delta_group[i]['delta_y']
            z1 = delta_group[i]['delta_z']
            ax.scatter(x1, y1, z1, label='Pose %d' % i)
        ax.set_xlabel('$\Delta x$ (mm)', fontsize=20, labelpad=10)
        ax.set_ylabel('$\Delta y$ (mm)', fontsize=20, labelpad=10)
        ax.set_zlabel('$\Delta z$ (mm)', fontsize=20, labelpad=10)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.tick_params(axis='z', labelsize=15)
        # ax.legend(loc="upper left",bbox_to_anchor=(1.2, 1),fontsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(self.path, 'error.png'))
        fig, ax = plt.subplots()
        fig.suptitle('Abs error histogram of all poses compared to true coordinates', fontsize=20)
        abs_plot = sns.histplot(abs_delta_df, multiple="layer")
        labels = ['$\Delta x$', '$\Delta y$', '$\Delta z$']
        mean_deltas = abs_delta_df.mean()
        std_deltas = abs_delta_df.std()
        ax.text(0.7, 0.8, 'Mean', fontsize=20, horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
        ax.text(0.85, 0.8, 'Std', fontsize=20, horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
        ax.text(0.7, 0.75, '$\Delta x $:{0:.3f}'.format(mean_deltas[0]), fontsize=20, horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
        ax.text(0.85, 0.75, '{0:.3f}'.format(std_deltas[0]), fontsize=20, horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
        ax.text(0.7, 0.7, '$\Delta y $:{0:.3f}'.format(mean_deltas[1]), fontsize=20, horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
        ax.text(0.85, 0.7, '{0:.3f}'.format(std_deltas[1]), fontsize=20, horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
        ax.text(0.7, 0.65, '$\Delta z $:{0:.3f}'.format(mean_deltas[2]), fontsize=20, horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
        ax.text(0.85, 0.65, '{0:.3f}'.format(std_deltas[2]), fontsize=20, horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
        plt.xlabel('abs(error) mm', fontsize=20)
        plt.ylabel('Count', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        for t, l in zip(abs_plot.legend_.texts, labels):
            t.set_text(l)
       
        plt.setp(abs_plot.get_legend().get_texts(), fontsize='20') 
        plt.savefig(os.path.join(self.path, 'abs_err.png'), bbox_inches="tight")
        
        return delta_df, abs_delta_df
    
    def recon_xyz(self,
                  unwrap_phase,  
                  white_imgs, 
                  mask_cond, 
                  modulation=None,
                  int_limit=None,
                  resid_outlier_limit=None):
        """
        Function to reconstruct 3d coordinates of calibration board and save as point cloud for each calibration pose.
        Parameters
        ----------
        unwrap_phase: list.
                      Unwrapped phase maps of each calibration pose.
        white_imgs: list.
                    True intensity image for texture mapping.
        mask_cond: str.
                   Mask condition for reconstruction based on 'intensity' or 'modulation'.
                   Intensity based mask is applied for reconstructing selected regions based on surface texture.
                   For example: if appropriate int_limit is set and mask_condition =  'intensity' the white
                   region of the calibration board can be reconstructed.
        modulation: list.
                    Modulation image for each calibration pose for applying mask to build the calibration board.
                    Default value is None and used if 'intensity' is used as 'mask_cond'.
        int_limit: float.
                   Minimum intensity value to extract white region.
        resid_outlier_limit: float.
                            This parameter is used to eliminate outlier points (points too far).

        Returns
        -------
        cordi_lst: list.
                    List of 3d coordinates of board points.
        color_lst: list.
                    List of color for each 3d point.

        """
        calibration = np.load(os.path.join(self.path, '{}_calibration_param.npz'.format(self.type_unwrap)))
        c_mtx = calibration["arr_0"]
        c_dist = calibration["arr_1"]
        p_mtx = calibration["arr_2"]
        cp_rot_mtx = calibration["arr_3"]
        cp_trans_mtx = calibration["arr_4"]
        
        cordi_lst = []
        color_lst = []
        for i, (u, w) in tqdm(enumerate(zip(unwrap_phase, white_imgs)), desc='building board 3d coordinates'):
            u_copy = deepcopy(u)
            w_copy = deepcopy(w)
            roi_mask = np.full(u_copy.shape, False)
            if mask_cond == 'modulation': 
                if len(modulation) > 0:
                    roi_mask[modulation[i][-1] > self.limit] = True
                    point_cloud_dir = os.path.join(self.path, 'modulation_mask')
                    if not os.path.exists(point_cloud_dir):
                        os.makedirs(point_cloud_dir)  
                else:
                    print('Please provide modulation images for mask.')
            elif mask_cond == 'intensity':
                if w.size != 0:
                    roi_mask[w > int_limit] = True
                    point_cloud_dir = os.path.join(self.path, 'intensity_mask')
                    if not os.path.exists(point_cloud_dir):
                        os.makedirs(point_cloud_dir)
                else:
                    print('Please provide intensity (texture) image.')
            else:
                roi_mask = True  # all pixels are selected.
                point_cloud_dir = os.path.join(self.path, 'no_mask')
                if not os.path.exists(point_cloud_dir):
                    os.makedirs(point_cloud_dir)
                print("The input mask_cond is not supported, no mask is applied.")
            u_copy[~roi_mask] = np.nan
            x, y, z = rc.reconstruction_obj(u_copy, c_mtx, c_dist, p_mtx, cp_rot_mtx, cp_trans_mtx, self.phase_st, self.pitch[-1])
            
            w_copy[~roi_mask] = False
            x[~roi_mask] = np.nan
            y[~roi_mask] = np.nan
            z[~roi_mask] = np.nan
            cordi = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
            nan_mask = np.isnan(cordi)
            up_cordi = cordi[~nan_mask.all(axis=1)]
            xyz = list(map(tuple, up_cordi)) 
            inte_img = w_copy / np.nanmax(w_copy)
            inte_rgb = np.stack((inte_img, inte_img, inte_img), axis=-1)
            rgb_intensity_vect = np.vstack((inte_rgb[:, :, 0].ravel(), inte_rgb[:, :, 1].ravel(), inte_rgb[:, :, 2].ravel())).T
            up_rgb_intensity_vect = rgb_intensity_vect[~nan_mask.all(axis=1)]
            color = list(map(tuple, up_rgb_intensity_vect))
            cordi_lst.append(up_cordi)
            color_lst.append(up_rgb_intensity_vect)
            PlyData(
                [
                    PlyElement.describe(np.array(xyz, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'points'),
                    PlyElement.describe(np.array(color, dtype=[('r', 'f4'), ('g', 'f4'), ('b', 'f4')]), 'color'),
                ]).write(os.path.join(point_cloud_dir, 'obj_%d.ply' % i))
            
        if mask_cond == 'intensity':
            residual_lst, outlier_lst = self.white_center_planefit(cordi_lst, resid_outlier_limit)
            return cordi_lst, color_lst, residual_lst, outlier_lst
        else:
            return cordi_lst, color_lst
    
    def white_center_planefit(self, cordi_lst, resid_outlier_limit):
        """
        Function to fit plane to extracted white region of calibration board. The function computes the plane and 
        calculates distance of points to the plane (residue) for each calibration pose.
        The function then plots the histogram residue of all calibration poses.

        Parameters
        ----------
        cordi_lst: list.
                   List of 3d coordinates of white points.
        resid_outlier_limit: float.
                             This parameter is used to eliminate outlier points (points too far).

        Returns
        -------
        residual_lst: list.
                      List of residue from each calibration pose.
        outlier_lst: list.
                     List of outlier points from each calibration pose.

        """
        residual_lst = []
        outlier_lst = []
        for i in tqdm(cordi_lst, desc='residual calculation for each pose'):
            xcord = i[:, 0]
            xcord = xcord[~np.isnan(xcord)]
            ycord = i[:, 1]
            ycord = ycord[~np.isnan(ycord)]
            zcord = i[:, 2]
            zcord = zcord[~np.isnan(zcord)]
            fit, residual = plane_fit(xcord, ycord, zcord)
            outliers = residual[(residual < -resid_outlier_limit) | (residual > resid_outlier_limit)]
            updated_resid = residual[(residual > -resid_outlier_limit) & (residual < resid_outlier_limit)]
            residual_lst.append(updated_resid)
            outlier_lst.append(outliers)
        plane_resid_plot(residual_lst)
        return residual_lst, outlier_lst
    
    def pp_distance_analysis(self, center_cordi_lst, val_label):
        """
        Function to compute given point to point distance on the calibration board over all calibration poses and 
        plot error plot.

        Parameters
        ----------
        center_cordi_lst: list.
                          List of x,y,z coordinates of detected circle centers in each calibration pose.
        val_label: float.
                   Any distance between two circle centers on the calibration board used.
        Returns
        -------
        distances: pandas dataframe.
                   Dataframe of distance calculated for each calibration pose.
        """
        dist = []
        true_coordinates = self.world_points()
        true_dist = distance.pdist(true_coordinates, 'euclidean')
        true_dist = np.sort(true_dist)
        dist_diff = np.diff(true_dist)
        dist_pos = np.where(dist_diff > abs(0.4))[0] + 1
        dist_split = np.split(true_dist, dist_pos)
        true_val, true_count = np.unique(true_dist, return_counts=True)
        true_val = np.around(true_val, decimals=3)
        true_df = pd.DataFrame(dist_split).T
        true_df.columns = true_val
        if val_label != true_val.all():
            distances = dist_l(center_cordi_lst, true_val)
            dist.append(distances)
            fig, ax = plt.subplots()
            x = distances[val_label]-val_label
            sns.histplot(x)
            mean = x.mean()
            std = x.std()
            # ax.set_xlim(mean-0.2,mean+0.5)
            ax.text(0.75, 0.75, 'Mean:{0:.3f}'.format(mean), fontsize=20, horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes)
            ax.text(0.75, 0.65, 'Std:{0:.3f}'.format(std), fontsize=20, horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            ax.set_xlabel('Measured error for true value {}mm'.format(val_label), fontsize=20)
            ax.set_ylabel('Count', fontsize=20)
        else:
            print('Invalid point to point distance. For given calibration board distance values are {}'.format(true_val))
            distances = None
        return distances
    
    def copy_tofolder(self, sample_index_list, source_folder, dest_folder):
        """
        Function for copying samples into a sub folder for bootstrapping. 
        Parameters
        ----------
        sample_index_list: list.
                           Index of poses to be transfered.
        source_folder: str.
                       Folder containing data of all poses
        dest_folder: str.
                     Folder into which the selected poses data will be copied for calculations.
                       
        """
        # empty destination folder
        for f in os.listdir(dest_folder):
            os.remove(os.path.join(dest_folder, f))
        # copy contents
        if self.data_type == 'jpeg':    
            to_be_moved = [glob.glob(os.path.join(source_folder, 'capt_%d_*.jpg' % x)) for x in sample_index_list]
        elif self.data_type == 'npy':
            to_be_moved = [glob.glob(os.path.join(source_folder, 'capt_%d_0.npy' % x)) for x in sample_index_list]
        flat_list = [item for sublist in to_be_moved for item in sublist]
        for t in flat_list:
            shutil.copy(t, dest_folder)
        return

    @staticmethod
    def sample_statistics(sample):
        mean = np.mean(sample, axis=0)
        std = np.std(sample, axis=0)
        return mean, std
    # processing all poses, eg:100 poses, together will lead to memory issue.

    def bootstrap_intrinsics_extrinsics(self, 
                                        delta_pose, 
                                        sub_sample_size, 
                                        no_sample_parameters):
        """
        Function to apply bootstrapping and system intrinsics and extrinsics.
        Parameters
        ----------
        delta_pose: int.
                    Number of images in each 4 directions, alteast 4 directions 
                    must be used to avoid any bias.
        sub_sample_size: int.
                         Sample size to sample from 4 subsets 
        no_sample_parameters:int.
                             Total number of samples of intrinsics and extrinsics parameters. 
        """
        left_direction = np.arange(0, delta_pose)
        right_direction = np.arange(delta_pose, 2*delta_pose)
        down_direction = np.arange(2*delta_pose, 3*delta_pose)
        up_direction = np.arange(3*delta_pose, 4*delta_pose)
        source_folder = self.path
        dest_folder = os.path.join(self.path, 'sub_calib')
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        self.path = dest_folder
        cam_mtx_sample = []
        cam_dist_sample = []
        proj_mtx_sample = []
        proj_dist_sample = []
        st_rmat_sample = []
        st_tvec_sample = []
        proj_h_mtx_sample = []
        cam_h_mtx_sample = []
        for i in range(no_sample_parameters):
            sample_index_l = np.random.choice(left_direction, size=sub_sample_size, replace=False)
            sample_index_r = np.random.choice(right_direction, size=sub_sample_size, replace=False)
            sample_index_d = np.random.choice(down_direction, size=sub_sample_size, replace=False)
            sample_index_u = np.random.choice(up_direction, size=sub_sample_size, replace=False)
            sample_index = np.sort(np.concatenate((sample_index_l, sample_index_r, sample_index_d, sample_index_u)))
            self.copy_tofolder(sample_index, source_folder, dest_folder)
            objp = self.world_points()
            unwrapv_lst, unwraph_lst, white_lst, avg_lst, mod_lst, wrapped_phase_lst = self.projcam_calib_img_multifreq()
            # Camera calibration
            camr_error, cam_objpts, cam_imgpts, cam_mtx, cam_dist, cam_rvecs, cam_tvecs = self.camera_calib(objp, white_lst, display=False)
            # Projector calibration
            proj_ret, proj_imgpts, proj_mtx, proj_dist, proj_rvecs, proj_tvecs = self.proj_calib(cam_objpts,
                                                                                                 cam_imgpts,
                                                                                                 unwrapv_lst,
                                                                                                 unwraph_lst)
            # Stereo calibration
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.0001)
            stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K3+cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5+cv2.CALIB_FIX_K6
            
            st_retu, st_cam_mtx, st_cam_dist, st_proj_mtx, st_proj_dist, st_cam_proj_rmat, st_cam_proj_tvec, E, F = cv2.stereoCalibrate(cam_objpts,
                                                                                                                                        cam_imgpts,
                                                                                                                                        proj_imgpts,
                                                                                                                                        cam_mtx,
                                                                                                                                        cam_dist,
                                                                                                                                        proj_mtx,
                                                                                                                                        proj_dist,
                                                                                                                                        white_lst[0].shape[::-1],
                                                                                                                                        flags=stereocalibration_flags,
                                                                                                                                        criteria=criteria)
            proj_h_mtx = np.dot(proj_mtx, np.hstack((st_cam_proj_rmat, st_cam_proj_tvec)))
            cam_h_mtx = np.dot(cam_mtx, np.hstack((np.identity(3), np.zeros((3, 1)))))
            cam_mtx_sample.append(cam_mtx)
            cam_dist_sample.append(cam_dist)
            proj_mtx_sample.append(proj_mtx)
            proj_dist_sample.append(proj_dist)
            st_rmat_sample.append(st_cam_proj_rmat)
            st_tvec_sample.append(st_cam_proj_tvec)
            proj_h_mtx_sample.append(proj_h_mtx)
            cam_h_mtx_sample.append(cam_h_mtx)
        cam_mtx_mean, cam_mtx_std = Calibration.sample_statistics(cam_mtx_sample)
        cam_dist_mean, cam_dist_std = Calibration.sample_statistics(cam_dist_sample)
        proj_mtx_mean, proj_mtx_std = Calibration.sample_statistics(proj_mtx_sample)
        proj_dist_mean, proj_dist_std = Calibration.sample_statistics(proj_dist_sample)
        st_rmat_mean, st_rmat_std = Calibration.sample_statistics(st_rmat_sample)
        st_tvec_mean, st_tvec_std = Calibration.sample_statistics(st_tvec_sample)
        proj_h_mtx_mean, proj_h_mtx_std = Calibration.sample_statistics(proj_h_mtx_sample)
        cam_h_mtx_mean, cam_h_mtx_std = Calibration.sample_statistics(cam_h_mtx_sample)
        np.savez(os.path.join(self.path, 'sample_calibration_param.npz'), cam_mtx_sample, cam_dist_sample, proj_mtx_sample, proj_dist_sample, st_rmat_sample, st_tvec_sample, proj_h_mtx_sample, cam_h_mtx_sample)
        np.savez(os.path.join(self.path, 'mean_calibration_param.npz'), cam_mtx_mean, cam_mtx_std, cam_dist_mean, cam_dist_std, proj_mtx_mean, proj_mtx_std, proj_dist_mean, proj_dist_std, st_rmat_mean, st_rmat_std, st_tvec_mean, st_tvec_std)
        np.savez(os.path.join(self.path, 'h_matrix_param.npz'), cam_h_mtx_mean, cam_h_mtx_std, proj_h_mtx_mean, proj_h_mtx_std)
        return cam_mtx_mean, cam_mtx_std, cam_dist_mean, cam_dist_std, proj_mtx_mean, proj_mtx_std, proj_dist_mean, proj_dist_std, st_rmat_mean, st_rmat_std, st_tvec_mean, st_tvec_std, proj_h_mtx_mean, proj_h_mtx_std, cam_h_mtx_mean, cam_h_mtx_std
      
def dist_l(center_cordi_lst, true_val):
    """
    Function to build a pandas dataframe of calculated point to point distance between each circle centers.

    Parameters
    ----------
    center_cordi_lst: list.
                      Array of x,y,z coordinates of detected circle centers in each calibration pose.
    true_val: float.
               Any distance between two circle centers.

    Returns
    -------
    dist_df : pandas dataframe
    """
    dist_df = pd.DataFrame()
    for i in range(len(center_cordi_lst)):
        dist = distance.pdist(center_cordi_lst[i], 'euclidean')
        # group into different distances
        dist = np.sort(dist)
        dist_diff = np.diff(dist)
        dist_pos = np.where(dist_diff > abs(0.4))[0] + 1  # to find each group starting(+1 since difference, [0] np where returns tuple)
        dist_split = np.split(dist, dist_pos)
        temp_df = pd.DataFrame(dist_split).T
        dist_df = dist_df.append(temp_df, ignore_index=True)
    dist_df = dist_df.iloc[:, 0:len(true_val)]
    dist_df.columns = true_val
    return dist_df

def obj(X, p):
    """
    Objective function for plane fitting. Used to calculate distance of a point from a plane.
    Parameters
    ----------
    X: list.
       3d coordinates array.
    p: list.
       List of plane parameters.
    Returns
    -------
    distances: pandas dataframe.
               Dataframe of distance calculated for each calibration pose.
    """
    plane_xyz = p[0:3]
    distance_pp = (plane_xyz * X.T).sum(axis=1) + p[3]
    return distance_pp / np.linalg.norm(plane_xyz)

def residuals(p, X):
    """
    Function to compute residuals for optimization.
    Parameters
    ----------
    p: np.array.
       Plane parameters.
    X: np.array.
       3d coordinates.
    Returns
    -------
    Distance of point from plane.
    """
    return obj(X, p)

def plane_fit(xcord, ycord, zcord):
    """
    Function to get optimized plane solution and calculate residue of each point with respect to the fitted plane.
    Parameters
    ----------
    xcord: list.
           List of X coordinates of each point.
    ycord: list.
           List of Y coordinates of each point.
    zcord: list.
           List of Z coordinates of each point.
    Returns
    -------
    coeff: np.array.
           Fitted plane coefficients.
    resid: np.array.
           Residue of each point from plane.

    """
    p0 = np.array([1, 1, 1, 1])
    xyz = np.vstack((xcord, ycord, zcord))
    coeff = leastsq(residuals, p0, args=xyz)[0]
    resid = obj(xyz, coeff)
    return coeff, resid  

def plane_resid_plot(residual_lst):
    """
    Function to generate diagnostic plot of residuals from white region plane fitting.
    Parameters
    ----------
    residual_lst: list.
                  List of residuals for each calibration pose.
    """
    residual = np.concatenate(residual_lst, axis=0)
    rms = np.sqrt(np.mean(residual**2))
    fig, ax = plt.subplots()
    plt.title('Residual from fitted planes', fontsize=20)
    sns.histplot(residual, legend=False)
    mean = np.mean(residual)
    std = np.std(residual)
    ax.text(0.75, 0.75, 'Mean:{0:.3f}'.format(mean), fontsize=20, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
    ax.text(0.75, 0.65, 'Std:{0:.3f}'.format(std), fontsize=20, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
    ax.text(0.75, 0.55, 'RMS:{0:.3f}'.format(rms), fontsize=20, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
    ax.set_xlabel('Residual', fontsize=20)
    ax.set_ylabel('Count', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.set_xlim(-1, 1)
    return

def main():
    
    # Initial parameters for calibration and testing results
    # proj properties
    proj_width = 912
    proj_height = 1140  # 800 1280 912 1140
    cam_width = 1920
    cam_hieght = 1200
    # type of unwrapping
    type_unwrap = 'multifreq'

    # circle detection parameters
    bobdetect_areamin = 100
    bobdetect_convexity = 0.75

    # calibration board properties
    dist_betw_circle = 25  # Distance between centers
    board_gridrows = 5
    board_gridcolumns = 15  # calibration board parameters

    # Define the path from which data is to be read. The calibration parameters will be saved in the same path.
    # reconstruction point clouds will also be saved in the same path
    root_dir = r'C:\Users\kl001\Documents\pyfringe_test'
    path = os.path.join(root_dir, '%s_calib_images' % type_unwrap)
    data_type = 'npy'
    processing = 'gpu'
    # multifrequency unwrapping parameters
    if type_unwrap == 'multifreq':
        pitch_list = [1375, 275, 55, 11]
        N_list = [3, 3, 3, 9]
        kernel_v = 7
        kernel_h = 7

    # multi wavelength unwrapping parameters
    if type_unwrap == 'multiwave':
        pitch_list = [139, 21, 18]
        N_list = [5, 5, 9]
        kernel_v = 9
        kernel_h = 9

    # phase coding unwrapping parameters
    if type_unwrap == 'phase':
        pitch_list = [20]
        N_list = [9]
        kernel_v = 25
        kernel_h = 25

    # no_pose = int(len(glob.glob(os.path.join(path,'capt*.jpg'))) / np.sum(np.array(N_list)) / 2)
    no_pose = 2
    acquisition_index = 0

    sigma_path = r'C:\Users\kl001\Documents\pyfringe_test\mean_pixel_std\mean_std_pixel.npy'
    quantile_limit = 1
    limit = nstep.B_cutoff_limit(sigma_path, quantile_limit, N_list, pitch_list)
    # Instantiate calibration class

    calib_inst = Calibration(proj_width=proj_width,
                             proj_height=proj_height,
                             cam_width=cam_width,
                             cam_height=cam_hieght,
                             no_pose=no_pose,
                             acquisition_index=acquisition_index,
                             mask_limit=limit,
                             type_unwrap=type_unwrap,
                             N_list=N_list,
                             pitch_list=pitch_list,
                             board_gridrows=board_gridrows,
                             board_gridcolumns=board_gridcolumns,
                             dist_betw_circle=dist_betw_circle,
                             bobdetect_areamin=bobdetect_areamin,
                             bobdetect_convexity=bobdetect_convexity,
                             kernel_v=kernel_v,
                             kernel_h=kernel_h,
                             path=path,
                             data_type=data_type,
                             processing=processing)
    unwrapv_lst, unwraph_lst, white_lst, mod_lst, proj_img_lst, cam_objpts, cam_imgpts, proj_imgpts, euler_angles, cam_mean_error, cam_delta, cam_df1, proj_mean_error, proj_delta, proj_df1 = calib_inst.calib(fx=1, fy=2)
    # Plot for re projection error analysis
    calib_inst.intrinsic_errors_plts(cam_mean_error, cam_delta, cam_df1, 'Camera')
    calib_inst.intrinsic_errors_plts(proj_mean_error, proj_delta, proj_df1, 'Projector')
    return


if __name__ == '__main__':
    main()
        