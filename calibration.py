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
import shutil
EPSILON = -0.5
TAU = 5.5

#TODO: Fix calibration reconstruction
class Calibration:
    """
    Calibration class is used to calibrate camera and projector setting. User can choose between phase coded , multi frequency and multi wavelength temporal unwrapping.
    After calibration the camera and projector parameters are saved as npz file at the given calibration image path.
    """
    def __init__(self,
                 proj_width,
                 proj_height,
                 cam_width,
                 cam_height,
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
        type_unwrap: string.
                     Type of temporal unwrapping to be applied.
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
                  Calibration image data can be either .tiff or .npy.
        processing:str.
                   Type of data processing. Use 'cpu' for desktop computation and 'gpu' for gpu.

        """
        self.proj_width = proj_width
        self.proj_height = proj_height
        self.cam_width = cam_width
        self.cam_height = cam_height
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
        if (self.type_unwrap == 'multifreq') or (self.type_unwrap == 'multiwave'):
            self.phase_st = 0
        else:
            print('ERROR: Invalid type_unwrap')
            return
        if not os.path.exists(self.path):
            print('ERROR: %s does not exist' % self.path)
        if (data_type != 'tiff') and (data_type != 'npy'):
            print('ERROR: Invalid data type. Data type should be \'tiff\' or \'npy\'')
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
        if self.type_unwrap == 'multiwave':
            unwrapv_lst, unwraph_lst, white_lst, mod_lst, wrapped_phase_lst, mask_lst = self.projcam_calib_img_multiwave()
        else:
            if self.type_unwrap != 'multifreq':
                print("phase unwrapping type is not recognized, use 'multifreq'")
            unwrapv_lst, unwraph_lst, white_lst, mod_lst, wrapped_phase_lst, mask_lst = self.projcam_calib_img_multifreq()
        
            
        unwrapv_lst = [nstep.recover_image(u, mask_lst[i], self.cam_height, self.cam_width) for i,u in enumerate(unwrapv_lst)]
        unwraph_lst = [nstep.recover_image(u, mask_lst[i], self.cam_height, self.cam_width) for i,u in enumerate(unwraph_lst)]
            
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
        cam_mean_error, cam_delta = self.intrinsic_error_analysis(cam_objpts, 
                                                                           cam_imgpts, 
                                                                           cam_mtx, 
                                                                           cam_dist, 
                                                                           cam_rvecs, 
                                                                           cam_tvecs)
        # Projector calibration error analysis
        proj_mean_error, proj_delta = self.intrinsic_error_analysis(cam_objpts,
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
        proj_h_mtx = np.dot(st_proj_mtx, np.hstack((st_cam_proj_rmat, st_cam_proj_tvec)))
        cam_h_mtx = np.dot(st_cam_mtx, np.hstack((np.identity(3), np.zeros((3, 1)))))
        np.savez(os.path.join(self.path, '{}_mean_calibration_param.npz'.format(self.type_unwrap)), 
                  cam_mtx_mean=st_cam_mtx, 
                  cam_dist_mean=st_cam_dist, 
                  proj_mtx_mean=st_proj_mtx, 
                  proj_dist_mean=st_proj_dist,
                  st_rmat_mean=st_cam_proj_rmat, 
                  st_tvec_mean=st_cam_proj_tvec,
                  cam_h_mtx_mean=cam_h_mtx,
                  proj_h_mtx_mean=proj_h_mtx)
        np.savez(os.path.join(self.path, '{}_cam_rot_tvecs.npz'.format(self.type_unwrap)), cam_rvecs, cam_tvecs)
        return unwrapv_lst, unwraph_lst, white_lst, mask_lst, mod_lst, proj_img_lst, cam_objpts, cam_imgpts, proj_imgpts, euler_angles, cam_mean_error, cam_delta, proj_mean_error, proj_delta
    
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
        cam_mean_error, cam_delta = self.intrinsic_error_analysis(cam_objpts,
                                                                  cam_imgpts,
                                                                  cam_mtx,
                                                                  cam_dist,
                                                                  cam_rvecs,
                                                                  cam_tvecs)
        # Projector calibration error analysis
        proj_mean_error, proj_delta = self.intrinsic_error_analysis(cam_objpts,
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
        proj_h_mtx = np.dot(st_proj_mtx, np.hstack((st_cam_proj_rmat, st_cam_proj_tvec)))
        cam_h_mtx = np.dot(st_cam_mtx, np.hstack((np.identity(3), np.zeros((3, 1)))))
        np.savez(os.path.join(self.path, '{}_calibration_param.npz'.format(self.type_unwrap)), 
                 cam_mtx_mean=st_cam_mtx, 
                 cam_dist_mean=st_cam_dist, 
                 proj_mtx_mean=st_proj_mtx, 
                 proj_dist_mean=st_proj_dist,
                 st_rmat_mean=st_cam_proj_rmat, 
                 st_tvec_std=st_cam_proj_tvec,
                 cam_h_mtx_mean=cam_h_mtx,
                 proj_h_mtx_mean=proj_h_mtx)
        np.savez(os.path.join(self.path, '{}_cam_rot_tvecs.npz'.format(self.type_unwrap)), cam_rvecs, cam_tvecs)
        return up_unwrapv_lst, up_unwraph_lst, up_white_lst, up_mod_lst, up_proj_img_lst, cam_objpts, cam_imgpts, proj_imgpts, euler_angles, cam_mean_error, cam_delta, proj_mean_error, proj_delta 

    def calib_center_reconstruction(self, cam_imgpts, unwrap_phase, mask_lst, sigma_path):
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
        vectors = np.load(os.path.join(self.path, '{}_cam_rot_tvecs.npz'.format(self.type_unwrap)))
        rvec = vectors["arr_0"]
        tvec = vectors["arr_1"]
        # Function call to get all circle center x,y,z coordinates
        center_cordi_lst = self.center_xyz(cam_imgpts, 
                                           unwrap_phase,
                                           mask_lst, 
                                           sigma_path)
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
    
    def multifreq_analysis(self, data_array):
        """
        Helper function to compute unwrapped phase maps using multi frequency unwrapping on CPU.
        Parameters
        ----------
        data_array: np.ndarray:float64.
                    Array of images used in 4 level phase unwrapping.
        Returns
        -------
        unwrap_v: np.ndarray.
                  Unwrapped phase map for vertical fringes.
        unwrap_h: np.ndarray.
                  Unwrapped phase map for horizontal fringes.
        phase_v: np.ndarray.
                 Wrapped phase maps for each pitch in vertical direction.
        phase_h: np.ndarray.
                 Wrapped phase maps for each pitch in horizontal direction.
        orig_img: np.ndarray.
                  True image without fringes.
        modulation: np.ndarray.
                    Modulation intensity image of each pitch.
        flag: np.ndarray.
              Flag to recover image from vector 
        """
        modulation, orig_img, phase_map, mask = nstep.phase_cal(data_array, self.limit, self.N, True )
        phase_v = phase_map[::2]
        phase_h = phase_map[1::2]
        phase_v[0][phase_v[0] < EPSILON] = phase_v[0][phase_v[0] < EPSILON] + 2 * np.pi
        phase_h[0][phase_h[0] < EPSILON] = phase_h[0][phase_h[0] < EPSILON] + 2 * np.pi
        unwrap_v, k_arr_v = nstep.multifreq_unwrap(self.pitch, 
                                                   phase_v, 
                                                   self.kernel_v, 
                                                   'v', 
                                                   mask, 
                                                   self.cam_width, 
                                                   self.cam_height)
        unwrap_h, k_arr_h = nstep.multifreq_unwrap(self.pitch, 
                                                   phase_h, 
                                                   self.kernel_h, 
                                                   'h', 
                                                   mask,
                                                   self.cam_width, 
                                                   self.cam_height)
        
        return unwrap_v, unwrap_h, phase_v, phase_h, orig_img[-1], modulation, mask
    
    def multifreq_analysis_cupy(self, data_array):
        """
        Helper function to compute unwrapped phase maps using multi frequency unwrapping on GPU.
        After computation all arrays are returned as numpy.
        Parameters
        ----------
        data_array: cp.ndarray:float64.
                    Cupy array of images used in 4 level phase unwrapping.
        Returns
        -------
        unwrap_v: cp.ndarray.
                  Unwrapped phase map for vertical fringes.
        unwrap_h: cp.ndarray.
                  Unwrapped phase map for horizontal fringes.
        phase_v: cp.ndarray.
                 Wrapped phase maps for each pitch in vertical direction.
        phase_h: cp.ndarray.
                 Wrapped phase maps for each pitch in horizontal direction.
        orig_img: cp.ndarray.
                  True image without fringes.
        modulation: cp.ndarray.
                    Modulation intensity image of each pitch.
        flag: cp.ndarray.
              Flag to recover image from vector 
        """
        modulation, orig_img, phase_map, mask = nstep_cp.phase_cal_cp(data_array, self.limit, self.N, True )
        phase_v = phase_map[::2]
        phase_h = phase_map[1::2]
        phase_v[0][phase_v[0] < EPSILON] = phase_v[0][phase_v[0] < EPSILON] + 2 * np.pi
        phase_h[0][phase_h[0] < EPSILON] = phase_h[0][phase_h[0] < EPSILON] + 2 * np.pi
        unwrap_v, k_arr_v = nstep_cp.multifreq_unwrap_cp(self.pitch, 
                                                         phase_v, 
                                                         self.kernel_v, 
                                                         'v',
                                                         mask, 
                                                         self.cam_width, 
                                                         self.cam_height)
        unwrap_h, k_arr_h = nstep_cp.multifreq_unwrap_cp(self.pitch, 
                                                         phase_h, 
                                                         self.kernel_h, 
                                                         'h',
                                                         mask, 
                                                         self.cam_width, 
                                                         self.cam_height)
        return cp.asnumpy(unwrap_v), cp.asnumpy(unwrap_h), cp.asnumpy(phase_v), cp.asnumpy(phase_h), cp.asnumpy(orig_img[-1]), cp.asnumpy(modulation), cp.asnumpy(mask)

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
        mod_lst: list.
                  List of modulation intensity images for each calibration pose.
        wrapped_phase_lst: dictionary.
                           List of vertical and horizontal phase maps

        """
        mask_lst = []
        mod_lst = []
        white_lst = []
        wrapv_lst = []
        wraph_lst = []
        unwrapv_lst = []
        unwraph_lst = []
        all_img_paths = sorted(glob.glob(os.path.join(self.path, 'capt_*')), key=os.path.getmtime)
        acquisition_index_list = [int(i[-14:-11]) for i in all_img_paths]
        for x in tqdm(acquisition_index_list,
                      desc='generating unwrapped phases map for {} poses'.format(len(acquisition_index_list))):

            if self.data_type == 'tiff':
                if os.path.exists(os.path.join(self.path, 'capt_%03d_000000.tiff' % x)):
                    img_path = sorted(glob.glob(os.path.join(self.path, 'capt_%03d*.tiff' % x)), key=os.path.getmtime)
                    images_arr = np.array([cv2.imread(file, 0) for file in img_path]).astype(np.float64)
                else:
                    print("ERROR: path is not exist! None item appended to the result")
                    images_arr = None
            elif self.data_type == 'npy':
                if os.path.exists(os.path.join(self.path, 'capt_%03d_000000.npy' % x)):
                    images_arr = np.load(os.path.join(self.path, 'capt_%03d_000000.npy' % x)).astype(np.float64)
                else:
                    print("ERROR: path is not exist! None item appended to the result")
                    images_arr = None
            else:
                print("ERROR: data type is not supported, must be '.tiff' or '.npy'.")
                images_arr = None

            if images_arr is not None:
                if self.processing == 'cpu':
                   unwrap_v, unwrap_h, phase_v, phase_h, orig_img, modulation, mask = self.multifreq_analysis(images_arr)
                else:
                    if self.processing != 'gpu':
                        print("WARNING: processing type is not recognized, use 'gpu'")
                    images_arr = cp.asarray(images_arr)
                    unwrap_v, unwrap_h, phase_v, phase_h, orig_img, modulation, mask = self.multifreq_analysis_cupy(images_arr)
                    cp._default_memory_pool.free_all_blocks()
                    
            else:
                unwrap_v = None
                unwrap_h = None
                phase_v = None
                phase_h = None
                orig_img = None
                modulation = None
                mask = None
            mask_lst.append(mask)
            mod_lst.append(modulation)
            white_lst.append(orig_img)
            wrapv_lst.append(phase_v)
            wraph_lst.append(phase_h)
            unwrapv_lst.append(unwrap_v)
            unwraph_lst.append(unwrap_h)

        wrapped_phase_lst = {"wrapv": wrapv_lst,
                             "wraph": wraph_lst}
        return unwrapv_lst, unwraph_lst, white_lst, mod_lst, wrapped_phase_lst, mask_lst

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
        flag_lst = []
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
        all_img_paths = sorted(glob.glob(os.path.join(self.path, 'capt_*')), key=os.path.getmtime)
        acquisition_index_list = [int(i[-14:-11]) for i in all_img_paths]
        for x in tqdm(acquisition_index_list, desc='generating unwrapped phases map for {} images'.format(len(acquisition_index_list))):
            if os.path.exists(os.path.join(self.path, 'capt_%03d_000000.tiff' % x)):
                img_path = sorted(glob.glob(os.path.join(self.path, 'capt_%3d*.tiff' % x)), key=os.path.getmtime)
                images_arr = np.array([cv2.imread(file, 0) for file in img_path]).astype(np.float64)
                multi_mod_v3, multi_white_v3, multi_phase_v3, flagv3 = nstep.phase_cal(images_arr[0: N_arr[0]],
                                                                             self.limit)
                multi_mod_h3, multi_white_h3, multi_phase_h3, flagh3 = nstep.phase_cal(images_arr[N_arr[0]:2 * N_arr[0]],
                                                                             self.limit)
                multi_mod_v2, multi_white_v2, multi_phase_v2, flagv2 = nstep.phase_cal(images_arr[2 * N_arr[0]:2 * N_arr[0] + N_arr[1]],
                                                                             self.limit)
                multi_mod_h2, multi_white_h2, multi_phase_h2, flagh2 = nstep.phase_cal(images_arr[2 * N_arr[0] + N_arr[1]:2 * N_arr[0] + 2 * N_arr[1]],
                                                                             self.limit)
                multi_mod_v1, multi_white_v1, multi_phase_v1, flagv1 = nstep.phase_cal(images_arr[2 * N_arr[0] + 2 * N_arr[1]:2 * N_arr[0] + 2 * N_arr[1] + N_arr[2]],
                                                                             self.limit)
                multi_mod_h1, multi_white_h1, multi_phase_h1, flagh1 = nstep.phase_cal(images_arr[2 * N_arr[0] + 2 * N_arr[1] + N_arr[2]:2 * N_arr[0] + 2 * N_arr[1] + 2 * N_arr[2]],
                                                                             self.limit)
                flagv = flagv1 and flagv2 and flagv3 
                multi_phase_v3 = multi_phase_v3[flagv]
                multi_phase_h3 = multi_phase_h3[flagv]
                multi_phase_v2 = multi_phase_v2[flagv]
                multi_phase_h2 = multi_phase_h2[flagv]
                multi_phase_v1 = multi_phase_v1[flagv]
                multi_phase_h1 = multi_phase_h1[flagv]
                
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
                
                mod_lst.append(np.array([multi_mod_v3, multi_mod_v2, multi_mod_v1, multi_mod_h3, multi_mod_h2, multi_mod_h1]))
                flag_lst.append(flagv)
                white_lst.append(multi_white_h1)
                wrapv_lst.append(phase_arr_v)
                wraph_lst.append(phase_arr_h)
                kv_lst.append(k_arr_v)
                kh_lst.append(k_arr_h)
                unwrapv_lst.append(multiwav_unwrap_v)
                unwraph_lst.append(multiwav_unwrap_h)
        wrapped_phase_lst = {"wrapv": wrapv_lst,
                             "wraph": wraph_lst}
        return unwrapv_lst, unwraph_lst, white_lst, mod_lst, wrapped_phase_lst, flag_lst
    
    @staticmethod
    def _image_resize(image_lst, fx, fy):
        resize_img_lst = []
        for i in image_lst:
            if i is not None:
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
            if white_lst[i] is not None:
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
            else:
                proj_img.append(None)        
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
        
        for white in white_lst:
            # Convert float image to uint8 type image.
            if white is not None:
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
                        cv2.waitKey(200)
    
        cv2.destroyAllWindows()
        if not all(ret_lst):
            print('Warning: Centers are not detected for some poses. Modify bobdetect_areamin and bobdetect_areamin parameter')
        # set flags to have tangential distortion = 0, k4 = 0, k5 = 0, k6 = 0
        flags = cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6
        # camera calibration
        cam_ret, cam_mtx, cam_dist, cam_rvecs, cam_tvecs = cv2.calibrateCamera(objpoints,
                                                                               cam_imgpoints,
                                                                               (self.cam_width, self.cam_height),
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
        if display:
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
            u = (nstep.bilinear_interpolate(unwrap_v_lst[x], c[:,0], c[:,1]) - self.phase_st) * self.pitch[-1] / (2*np.pi)
            v = (nstep.bilinear_interpolate(unwrap_h_lst[x], c[:,0], c[:,1]) - self.phase_st) * self.pitch[-1] / (2*np.pi)
            coordi = np.column_stack((u, v)).reshape(cam_objpts[0].shape[0], 1, 2).astype(np.float32)
            proj_imgpts.append(coordi)
            if proj_img_lst is not None:
                proj_color = cv2.cvtColor(proj_img_lst[x], cv2.COLOR_GRAY2RGB)  # only for drawing
                proj_keypoints = cv2.drawChessboardCorners(proj_color, (self.board_gridrows, self.board_gridcolumns), coordi, True)
                cv2.imshow("Projector calibration", proj_keypoints)  # display
                cv2.waitKey(200)
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
        if proj_img_lst is not None:
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
            plt.show()
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
        for i in range(len(objpts)):
            img2, _ = cv2.projectPoints(objpts[i], rvecs[i], tvecs[i], mtx, dist)
            delta = imgpts[i]-img2
            delta_lst.append(delta.reshape(objpts[i].shape[0], 2))
            error = cv2.norm(imgpts[i], img2, cv2.NORM_L2) / len(img2)
            mean_error.append(error)
        
        return mean_error, np.array(delta_lst)

    def intrinsic_errors_plts(self, mean_error, delta, dev, pixel_size):
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
        pixel_size: list.
                    x,y direction dimension of a pixel.
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
        plt.show()
        plt.figure()
        plt.scatter((delta[:, :, 0]*pixel_size[0]).ravel(), (delta[:, :, 1]*pixel_size[1]).ravel())
        plt.xlabel('x [pixel]', fontsize=30)
        plt.ylabel('y [pixel]', fontsize=30)
        plt.xlim(-0.5, 0.5)
        plt.ylim(-0.5, 0.5)
        axes = plt.gca()
        axes.set_aspect(1)  # to set aspect equal
        #axes.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
        axes.yaxis.offsetText.set_fontsize(15)
        axes.xaxis.offsetText.set_fontsize(15)
        plt.title('Re projection error for {}\n '.format(dev), fontsize=30)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.show()
        return

    # center reconstruction
    def center_xyz(self, center_pts, unwrap_phase, mask_lst, sigma_path):
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
        reconst_instance = rc.Reconstruction(proj_width=self.proj_width,
                                             proj_height=self.proj_height,
                                             cam_width=self.cam_width,
                                             cam_height=self.cam_height,
                                             type_unwrap=self.type_unwrap,
                                             limit=self.limit,
                                             N_list=self.N,
                                             pitch_list=self.pitch,
                                             fringe_direc='v',
                                             kernel=self.kernel_v,
                                             data_type=self.data_type,
                                             processing=self.processing,
                                             calib_path=self.path,
                                             sigma_path=sigma_path,
                                             object_path=self.path,
                                             temp=False,
                                             probability=False)
        for i in tqdm(range(0, len(center_pts)), desc='building camera centers 3d coordinates'):
            # undistort points
            cordi = reconst_instance.reconstruction_pts(center_pts[i], unwrap_phase[i], mask_lst[i])
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
                  mask_lst, 
                  sigma_path,
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
        print(sigma_path)
        cordi_lst = []
        color_lst = []
        reconstruct = rc.Reconstruction(self.proj_width,
                                        self.proj_height,
                                        self.cam_width,
                                        self.cam_height,
                                        self.type_unwrap,
                                        self.limit,
                                        self.N,
                                        self.pitch,
                                        'v',
                                        self.kernel_v,
                                        self.data_type,
                                        self.processing,
                                        self.path,
                                        sigma_path,
                                        self.path,
                                        False,
                                        True,
                                        False)
        for i, (u, w, m) in tqdm(enumerate(zip(unwrap_phase, white_imgs, mask_lst)), desc='building board 3d coordinates'):
            
            cordi = reconstruct.reconstruction_obj(u[m])
            if int_limit:
                roi_mask = np.full(u.shape, False)
                roi_mask[w > int_limit] = True
                w[~roi_mask] = np.nan
                point_cloud_dir = os.path.join(self.path, 'intensity_mask')
            else:
                w = w[m]
                point_cloud_dir = os.path.join(self.path, 'modulation_mask') 
            xyz = list(map(tuple, cordi)) 
            inte_img = (w/ np.nanmax(w)).ravel()
            inte_rgb = np.stack((inte_img, inte_img, inte_img), axis=-1)
            color = list(map(tuple, inte_rgb))
            cordi_lst.append(cordi)
            color_lst.append(inte_rgb)
            if not os.path.exists(point_cloud_dir):
                os.makedirs(point_cloud_dir)
                
            PlyData(
                [
                    PlyElement.describe(np.array(xyz, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'points'),
                    PlyElement.describe(np.array(color, dtype=[('r', 'f4'), ('g', 'f4'), ('b', 'f4')]), 'color'),
                ]).write(os.path.join(point_cloud_dir, 'obj_%d.ply' % i))
        
            
        if resid_outlier_limit:
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
    
    
    @staticmethod
    def sample_indices(delta_pose, pool_size_list, no_sample_sets):
        sample_indices_list = []
        for p in pool_size_list:
            for i in range (no_sample_sets):
                sub_sample_size = int(p/4) # no of poses from each delta_pose
                left = np.arange(0, delta_pose)
                right = np.arange(delta_pose, 2*delta_pose)
                down = np.arange(2*delta_pose, 3*delta_pose)
                up = np.arange(3*delta_pose, 4*delta_pose)
                sample_index = np.sort(np.concatenate((np.random.choice(left, size = sub_sample_size, replace = False),
                                                      np.random.choice(right, size = sub_sample_size, replace = False),
                                                      np.random.choice(down, size = sub_sample_size, replace = False),
                                                      np.random.choice(up, size = sub_sample_size, replace = False))))
                if len(sample_index) < p:
                    total = np.arange(0,4*delta_pose)
                    extras = np.random.choice(total, size = p - len(sample_index), replace = False)
                    sample_index = np.sort(np.append(sample_index, extras))
                sample_indices_list.append(sample_index)
        return sample_indices_list
    
    @staticmethod
    def sample_statistics(sample, len_pool_list):
        sample = sample.reshape(len_pool_list, -1, sample.shape[-2], sample.shape[-1])
        mean = np.mean(sample, axis=1)
        std = np.std(sample, axis=1)
        return mean, std, sample
    def sub_phase_map_gen(self, sample_index):
        mask_lst = []
        mod_lst = []
        white_lst = []
        wrapv_lst = []
        wraph_lst = []
        unwrapv_lst = []
        unwraph_lst = []
        for x in tqdm(sample_index,
                      desc='generating unwrapped phases map for {} poses'.format(len(sample_index))):
            if self.data_type == 'tiff':
                if os.path.exists(os.path.join(self.path, 'capt_%03d_000000.tiff' % x)):
                    img_path = sorted(glob.glob(os.path.join(self.path, 'capt_%03d*.tiff' % x)), key=os.path.getmtime)
                    images_arr = np.array([cv2.imread(file, 0) for file in img_path]).astype(np.float64)
                else:
                    print("ERROR: path is not exist! None item appended to the result")
                    images_arr = None
            elif self.data_type == 'npy':
                if os.path.exists(os.path.join(self.path, 'capt_%03d_000000.npy' % x)):
                    images_arr = np.load(os.path.join(self.path, 'capt_%03d_000000.npy' % x)).astype(np.float64)
                else:
                    print("ERROR: path is not exist! None item appended to the result")
                    images_arr = None
            else:
                print("ERROR: data type is not supported, must be '.tiff' or '.npy'.")
                images_arr = None

            if images_arr is not None:
                if self.processing == 'cpu':
                   unwrap_v, unwrap_h, phase_v, phase_h, orig_img, modulation, mask = self.multifreq_analysis(images_arr)
                else:
                    if self.processing != 'gpu':
                        print("WARNING: processing type is not recognized, use 'gpu'")
                    images_arr = cp.asarray(images_arr)
                    unwrap_v, unwrap_h, phase_v, phase_h, orig_img, modulation, mask = self.multifreq_analysis_cupy(images_arr)
                    cp._default_memory_pool.free_all_blocks()
            else:
                unwrap_v = None
                unwrap_h = None
                phase_v = None
                phase_h = None
                orig_img = None
                modulation = None
                mask = None
            mask_lst.append(mask)
            mod_lst.append(modulation)
            white_lst.append(orig_img)
            wrapv_lst.append(phase_v)
            wraph_lst.append(phase_h)
            unwrapv_lst.append(unwrap_v)
            unwraph_lst.append(unwrap_h)

        wrapped_phase_lst = {"wrapv": wrapv_lst,
                             "wraph": wraph_lst}
        return unwrapv_lst, unwraph_lst, white_lst, mod_lst, wrapped_phase_lst, mask_lst
            
   
    def sub_calibration(self,sample_index):
        
        unwrapv_lst, unwraph_lst, white_lst, mod_lst, wrapped_phase_lst, mask_lst = self.sub_phase_map_gen(sample_index)
        unwrapv_lst = [nstep.recover_image(u, mask_lst[i], self.cam_height, self.cam_width) for i,u in enumerate(unwrapv_lst)]
        unwraph_lst = [nstep.recover_image(u, mask_lst[i], self.cam_height, self.cam_width) for i,u in enumerate(unwraph_lst)]
        objp = self.world_points()
        camr_error, cam_objpts, cam_imgpts, cam_mtx, cam_dist, cam_rvecs, cam_tvecs = self.camera_calib(objp, white_lst, display=False)
        # Projector calibration
        proj_ret, proj_imgpts, proj_mtx, proj_dist, proj_rvecs, proj_tvecs = self.proj_calib(cam_objpts,
                                                                                             cam_imgpts,
                                                                                             unwrapv_lst,
                                                                                             unwraph_lst,
                                                                                             proj_img_lst=None)
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
        proj_h_mtx = np.dot(proj_mtx, np.hstack((st_cam_proj_rmat, st_cam_proj_tvec)))
        cam_h_mtx = np.dot(cam_mtx, np.hstack((np.identity(3), np.zeros((3, 1)))))
        return st_cam_mtx, st_cam_dist, st_proj_mtx, st_proj_dist, st_cam_proj_rmat, st_cam_proj_tvec, cam_h_mtx, proj_h_mtx

    def bootstrap_intrinsics_extrinsics(self, 
                                        delta_pose, 
                                        pool_size_list, 
                                        no_sample_sets):
        """
        Function to apply bootstrapping and system intrinsics and extrinsics.
        Parameters
        ----------
        delta_pose: int.
                    Number of images in each 4 directions, alteast 4 directions 
                    must be used to avoid any bias.
        pool_size_list: list.
                        list of no. of poses.
        no_sample_parameters:int.
                             Total number of samples of intrinsics and extrinsics parameters.(Iterations per pool size) 
        """
        sample_indices_list = Calibration.sample_indices(delta_pose, pool_size_list, no_sample_sets) 
        cam_mtx_sample = []
        cam_dist_sample = []
        proj_mtx_sample = []
        proj_dist_sample = []
        st_rmat_sample = []
        st_tvec_sample = []
        proj_h_mtx_sample = []
        cam_h_mtx_sample = []
        for s in sample_indices_list:
            cam_mtx, cam_dist, proj_mtx, proj_dist, st_cam_proj_rmat, st_cam_proj_tvec, cam_h_mtx, proj_h_mtx = self.sub_calibration(s)
            cam_mtx_sample.append(cam_mtx)
            cam_dist_sample.append(cam_dist)
            proj_mtx_sample.append(proj_mtx)
            proj_dist_sample.append(proj_dist)
            st_rmat_sample.append(st_cam_proj_rmat)
            st_tvec_sample.append(st_cam_proj_tvec)
            cam_h_mtx_sample.append(cam_h_mtx)
            proj_h_mtx_sample.append(proj_h_mtx)
        
        cam_mtx_mean, cam_mtx_std, cam_mtx_sample  = Calibration.sample_statistics(np.array(cam_mtx_sample), len(pool_size_list))
        cam_dist_mean, cam_dist_std, cam_dist_sample = Calibration.sample_statistics(np.array(cam_dist_sample), len(pool_size_list))
        proj_mtx_mean, proj_mtx_std, proj_mtx_sample = Calibration.sample_statistics(np.array(proj_mtx_sample), len(pool_size_list))
        proj_dist_mean, proj_dist_std, proj_dist_sample = Calibration.sample_statistics(np.array(proj_dist_sample), len(pool_size_list))
        st_rmat_mean, st_rmat_std, st_rmat_sample = Calibration.sample_statistics(np.array(st_rmat_sample), len(pool_size_list))
        st_tvec_mean, st_tvec_std, st_tvec_sample = Calibration.sample_statistics(np.array(st_tvec_sample), len(pool_size_list))
        proj_h_mtx_mean, proj_h_mtx_std, proj_h_mtx_sample = Calibration.sample_statistics(np.array(proj_h_mtx_sample), len(pool_size_list))
        cam_h_mtx_mean, cam_h_mtx_std, cam_h_mtx_sample = Calibration.sample_statistics(np.array(cam_h_mtx_sample), len(pool_size_list))
        np.savez(os.path.join(self.path, '{}_sample_calibration_param.npz'.format(self.type_unwrap)), 
                 cam_mtx_sample=cam_mtx_sample, 
                 cam_dist_sample=cam_dist_sample, 
                 proj_mtx_sample=proj_mtx_sample, 
                 proj_dist_sample=proj_dist_sample, 
                 st_rmat_sample=st_rmat_sample, 
                 st_tvec_sample=st_tvec_sample, 
                 proj_h_mtx_sample=proj_h_mtx_sample, 
                 cam_h_mtx_sample=cam_h_mtx_sample)
        np.savez(os.path.join(self.path, '{}_mean_calibration_param.npz'.format(self.type_unwrap)), 
                 cam_mtx_mean=cam_mtx_mean, 
                 cam_dist_mean=cam_dist_mean, 
                 proj_mtx_mean=proj_mtx_mean, 
                 proj_dist_mean=proj_dist_mean, 
                 st_rmat_mean=st_rmat_mean, 
                 st_tvec_mean=st_tvec_mean, 
                 cam_h_mtx_mean=cam_h_mtx_mean,
                 proj_h_mtx_mean=proj_h_mtx_mean)
        np.savez(os.path.join(self.path, '{}_std_calibration_param.npz'.format(self.type_unwrap)),
                 cam_mtx_std=cam_mtx_std, 
                 cam_dist_std=cam_dist_std, 
                 proj_mtx_std=proj_mtx_std, 
                 proj_dist_std=proj_mtx_std, 
                 st_rmat_std=st_rmat_std, 
                 st_tvec_std=st_tvec_std, 
                 cam_h_mtx_std=cam_h_mtx_std, 
                 proj_h_mtx_std=proj_h_mtx_std)
        return cam_mtx_sample, cam_dist_sample, proj_mtx_sample, proj_dist_sample, st_rmat_sample, st_tvec_sample, cam_h_mtx_sample, proj_h_mtx_sample
      
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
    cam_height = 1200
    fx=1 
    fy=2
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

    # multi wavelength unwrapping parameters
    if type_unwrap == 'multiwave':
        pitch_list = [139, 21, 18]
        N_list = [5, 5, 9]
        kernel_v = 9
        kernel_h = 9

    # multifrequency unwrapping parameters
    else:
        if type_unwrap != 'multifreq':
            print("Unwrap type is not recognized, use 'multifreq'.")
        pitch_list = [1375, 275, 55, 11]
        N_list = [3, 3, 3, 9]
        kernel_v = 7
        kernel_h = 7

    sigma_path = r'C:\Users\kl001\Documents\pyfringe_test\mean_pixel_std\mean_std_pixel.npy'
    quantile_limit = 4.5
    limit = nstep.B_cutoff_limit(sigma_path, quantile_limit, N_list, pitch_list)
    # Instantiate calibration class

    calib_inst = Calibration(proj_width=proj_width, 
                             proj_height=proj_height,
                             cam_width=cam_width,
                             cam_height=cam_height,
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
    unwrapv_lst, unwraph_lst, white_lst, mask_lst, mod_lst, proj_img_lst, cam_objpts, cam_imgpts, proj_imgpts, euler_angles, cam_mean_error, cam_delta, proj_mean_error, proj_delta = calib_inst.calib(fx, fy)
    # Plot for re projection error analysis
    calib_inst.intrinsic_errors_plts( cam_mean_error, cam_delta, 'Camera', pixel_size = [1,1])
    calib_inst.intrinsic_errors_plts( proj_mean_error, proj_delta, 'Projector', pixel_size =[1,0.5]) 
    option = input("\nDo you want to see constructed projector images?(y/n):")
    if option == "y":
        calib_inst.image_analysis(proj_img_lst, title='Projector image', aspect_ratio=0.5)
    print_option = input("Do you want to display calibration parameters?(y/n):")
    if print_option == "y":
        calibration = np.load(os.path.join(path,'{}_mean_calibration_param.npz'.format(type_unwrap)))
        print("\nCamera matrix:\n{}".format(calibration["cam_mtx_mean"]))
        print("\nCamera distortion:\n{}".format(calibration["cam_dist_mean"]))
        print("Projector matrix:\n{}".format(calibration["proj_mtx_mean"]))
        print("Camera projector rotation matrix:\n{}".format(calibration["st_rmat_mean"]))
        print("Camera projector translation vector:\n{}".format(calibration["st_tvec_mean"]))
    return


if __name__ == '__main__':
    main()
        