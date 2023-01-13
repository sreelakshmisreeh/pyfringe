#!/usr/bin/env python
# coding: utf-8



import numpy as np
import nstep_fringe as nstep
import cv2
import os
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

EPSILON = -0.5
TAU = 5.5
#TODO: modify to compute directly from list of images 
class calibration:
    '''
    Calibration class is used to calibrate camera and projector setting. User can choose between phase coded , multifrequency and multiwavelength temporal unwrapping.
    After calibration the camera and projector parameters are saved as npz file at the given calibration image path.
    '''
    def __init__(self,proj_width, 
                 proj_height, 
                 mask_limit, 
                 type_unwrap, 
                 N_list, 
                 pitch_list, 
                 board_gridrows, 
                 board_gridcolumns, 
                 dist_betw_circle, 
                 path):
        '''
        Parameters
        ----------
        proj_width = type: float. Width of projector.
        proj_height = type: float. Height of projector.
        mask_limit = type: float. Modulation limit for applying mask to captured images.
        type_unwrap = type: string. Type of temporal unwrapping to be applied. 
                      'phase' = phase coded unwrapping method, 
                      'multifreq' = multifrequency unwrapping method
                      'multiwave' = multiwavelength unwrapping method.
        N_list = type: float array. The number of steps in phase shifting algorithm. 
                                    If phase coded unwrapping method is used this is a single element array. 
                                    For other methods corresponding to each pitch one element in the list.
        pitch_list = type: float array. Array of number of pixels per fringe period.
        board_gridrows = type: int. Number of rows in the assymetric circle pattern.
        board_gridcolumns = type: int. Number of columns in the asymmetric circle pattern.
        dist_betw_circle = type: float. Distance between circle centers.
        path = type: string. Path to read captured calibration images and save calibration results.

        '''
        self.width = proj_width
        self.height = proj_height
        self.limit = mask_limit
        self.type_unwrap = type_unwrap
        self.N = N_list
        self.pitch = pitch_list
        self.path = path
        self.board_gridrows = board_gridrows
        self.board_gridcolumns = board_gridcolumns
        self.dist_betw_circle = dist_betw_circle
        
    def calib(self, no_pose,  bobdetect_areamin, bobdetect_convexity,  kernel_v = 1, kernel_h=1):
        '''
        Function to calibrate camera and projector and save npz file of calibration parameter based on user choice 
        of temporal phase unwrapping. 
        
        Parameters
        ----------
        no_pose = type:int. Number of calibration poses
        bobdetect_areamin = type: float. Minimum area of area for bob detector. 
        bobdetect_convexity = type: float. Circle convexity for bob detector.
        
        Returns
        -------
        unwrapv_lst = type:list of float. List of unwrapped phase maps obtained from horizontally varying intensity patterns.
        unwraph_lst type:list of float. List of unwrapped phase maps obtained from vertically varying intensity patterns..
        white_lst = type:list of float. List of true images for each calibration pose..
        mod_lst = type: list of float. List of modulation intensity images for each calibration pose for intensity 
                                        varying both horizontally and vertically.
        cam_objpts = type: list of float. List of world cordintaes used for camera calibration for each pose.
        cam_imgpts = type: list of float. List of circle centers grid for each calibration pose.
        proj_imgpts = type: float. List of circle center grid coordinates for each pose of projector calibration.
        euler_angles = type:float. Array of roll,pitch and yaw angles between camera and projector in degrees.
        cam_mean_error = type: list of floats. List of camera mean error per calibration pose.
        cam_delta = type: list of float. List of camera reprojection error.
        cam_df1 = type: pandas dataframe of floats. Dataframe of camera absolute error in x and y directions of all poses.
        proj_mean_error := type: list of floats. List of projector mean error per calibration pose.
        proj_delta = type: list of float. List of projector reprojection error.
        proj_df1 = type: pandas dataframe of floats. Dataframe of projector absolute error in x and y directions of all poses.

        '''
        objp = self.world_points(self.dist_betw_circle, self.board_gridrows, self.board_gridcolumns)
        if self.type_unwrap == 'phase':
            phase_st = -np.pi
            unwrapv_lst, unwraph_lst, white_lst, avg_lst, mod_lst, gamma_lst, wrapped_phase_lst = self.projcam_calib_img_phase(no_pose,self.limit, 
                                                                                                                               self.N[0], self.pitch[-1], 
                                                                                                                               self.width, self.height,
                                                                                                                               kernel_v, kernel_h, self.path)
        elif self.type_unwrap == 'multifreq':
            phase_st = 0
            unwrapv_lst, unwraph_lst, white_lst, avg_lst, mod_lst, gamma_lst, wrapped_phase_lst = self.projcam_calib_img_multifreq(no_pose, self.limit, 
                                                                                                                                   self.N, self.pitch, 
                                                                                                                                   self.width, self.height, 
                                                                                                                                   kernel_v, kernel_h, self.path)
        elif self.type_unwrap == 'multiwave':
            phase_st = 0
            unwrapv_lst, unwraph_lst, white_lst, avg_lst, mod_lst, gamma_lst, wrapped_phase_lst = self.projcam_calib_img_multiwave(no_pose,self.limit, 
                                                                                                                                   self.N, self.pitch, 
                                                                                                                                   self.width, self.height, 
                                                                                                                                   kernel_v, kernel_h, self.path)
            
        #Projector images
        proj_img_lst=self.projector_img(unwrapv_lst, unwraph_lst, white_lst, self.width, self.height, self.pitch[-1], phase_st)
        #Camera calibration
        camr_error, cam_objpts, cam_imgpts, cam_mtx, cam_dist, cam_rvecs, cam_tvecs = self.camera_calib(objp,
                                                                                             white_lst,
                                                                                             bobdetect_areamin,
                                                                                             bobdetect_convexity, 
                                                                                             self.board_gridrows, self.board_gridcolumns)
        
        #Projector calibration
        proj_ret, proj_imgpts,proj_mtx, proj_dist, proj_rvecs, proj_tvecs = self.proj_calib(cam_objpts, 
                                                                                            cam_imgpts, 
                                                                                            unwrapv_lst, 
                                                                                            unwraph_lst, 
                                                                                            proj_img_lst, 
                                                                                            self.pitch[-1], 
                                                                                            phase_st, 
                                                                                            self.board_gridrows, 
                                                                                            self.board_gridcolumns) 
        # Camera calibration error analysis
        cam_mean_error, cam_delta, cam_df1 = self.intrinsic_error_analysis(cam_objpts, 
                                                                           cam_imgpts, 
                                                                           cam_mtx, 
                                                                           cam_dist, 
                                                                           cam_rvecs, 
                                                                           cam_tvecs)
        #Projector calibration error analysis
        proj_mean_error, proj_delta, proj_df1 = self.intrinsic_error_analysis(cam_objpts,
                                                                              proj_imgpts,
                                                                              proj_mtx,
                                                                              proj_dist,
                                                                              proj_rvecs,
                                                                              proj_tvecs)
        #Stereo calibration
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.0001)
        stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC+cv2.CALIB_ZERO_TANGENT_DIST+cv2.CALIB_FIX_K3+cv2.CALIB_FIX_K4+cv2.CALIB_FIX_K5+cv2.CALIB_FIX_K6

        st_retu,st_cam_mtx,st_cam_dist,st_proj_mtx,st_proj_dist,st_cam_proj_rmat,st_cam_proj_tvec,E,F=cv2.stereoCalibrate(cam_objpts,cam_imgpts,proj_imgpts,
                                                                                                        cam_mtx,cam_dist,proj_mtx,proj_dist,
                                                                                                        white_lst[0].shape[::-1],
                                                                                                        flags=stereocalibration_flags,
                                                                                                        criteria=criteria)
        project_mat=np.hstack((st_cam_proj_rmat,st_cam_proj_tvec))
        _,_,_,_,_,_,euler_angles=cv2.decomposeProjectionMatrix(project_mat)
        
        np.savez(os.path.join(self.path,'{}_calibration_param.npz'.format(self.type_unwrap)), st_cam_mtx,st_cam_dist, st_proj_mtx, st_cam_proj_rmat, st_cam_proj_tvec) 
        np.savez(os.path.join(self.path,'{}_cam_rot_tvecs.npz'.format(self.type_unwrap)),cam_rvecs,cam_tvecs)
        
        return unwrapv_lst, unwraph_lst, white_lst, mod_lst, proj_img_lst, cam_objpts, cam_imgpts, proj_imgpts, euler_angles, cam_mean_error, cam_delta, cam_df1, proj_mean_error, proj_delta, proj_df1
    
    def update_list_calib(self, proj_df1, unwrapv_lst, unwraph_lst, white_lst, mod_lst,proj_img_lst, bobdetect_areamin, bobdetect_convexity, reproj_criteria):
        '''
        Function to remove outlier calibration poses.

        Parameters
        ----------
        proj_df1 : type: pandas dataframe of floats. Dataframe of projector absolute error in x and y directions of all poses.
        unwrapv_lst = type:list of float. List of unwrapped phase maps obtained from horizontally varying intensity patterns.
        unwraph_lst type:list of float. List of unwrapped phase maps obtained from vertically varying intensity patterns.
        white_lst = type:list of float. List of true images for each calibration pose.
        mod_lst = type: list of float. List of modulation intensity images for each calibration pose for intensity 
                                    varying both horizontally and vertically.
        proj_img_lst = type: float. List of circle center grid coordinates for each pose of projector calibration.
        bobdetect_areamin = type: float. Minimum area of area for bob detector. 
        bobdetect_convexity = type: float. Circle convexity for bob detector.
        reproj_criteria = type: float. Criteria to remove outlier poses.

        Returns
        -------
        up_unwrapv_lst = type:list of float. Updated list of unwrapped phase maps obtained from horizontally 
                                                varying intensity patterns.
        up_unwraph_lst type:list of float. Updated list of unwrapped phase maps obtained from vertically 
                                            varying intensity patterns.
        up_white_lst = type:list of float. Updated list of true images for each calibration pose.
        up_mod_lst = type:list of float. Updated list of modulation intensity images for each 
                                        calibration pose for intensity varying both horizontally and vertically.
        up_proj_img_lst = type:list of float. Updated list of modulation intensity images for each 
                                            calibration pose for intensity varying both horizontally and vertically.
        cam_objpts = type:list of float. Updated list of world cordintaes used for camera calibration for each pose.
        cam_imgpts = type: list of float. Updated list of circle centers grid for each calibration pose.
        proj_imgpts = type: float. Updated list of circle center grid coordinates for each pose of projector calibration.
        euler_angles = type:float. Array of roll,pitch and yaw angles between camera and projector in degrees
        cam_mean_error = type: list of floats. List of camera mean error per calibration pose.
        cam_delta = type: list of float. List of camera reprojection error.
        cam_df1 = type: pandas dataframe of floats. Dataframe of camera absolute error in x and y directions 
                                                    of updated list of poses.
        proj_mean_error := type: list of floats. List of projector mean error per calibration pose.
        proj_delta = type: list of float. List of projector reprojection error.
        proj_df1 = type: pandas dataframe of floats. Dataframe of projector absolute error in x and y directions
                                                    of updated list of poses.

        '''
        up_lst = list(set(proj_df1[proj_df1['absdelta_x']>(reproj_criteria)]['image'].to_list() + proj_df1[proj_df1['absdelta_y']>(reproj_criteria)]['image'].to_list()))
        up_white_lst =[]; up_unwrapv_lst=[];up_unwraph_lst=[];up_mod_lst=[];up_proj_img_lst=[]
        for index, element in enumerate(white_lst):
            if index not in up_lst:
                up_white_lst.append(element)
                up_unwrapv_lst.append(unwrapv_lst[index])
                up_unwraph_lst.append(unwraph_lst[index])
                up_mod_lst.append(mod_lst[index])
                up_proj_img_lst.append(proj_img_lst[index])
        objp = self.world_points(self.dist_betw_circle, self.board_gridrows, self.board_gridcolumns)
        if self.type_unwrap == 'phase':
            phase_st = -np.pi
        elif (self.type_unwrap == 'multifreq') or (self.type_unwrap == 'multiwave'):
            phase_st = 0
        camr_error, cam_objpts, cam_imgpts, cam_mtx, cam_dist, cam_rvecs, cam_tvecs = self.camera_calib(objp,
                                                                                             up_white_lst,
                                                                                             bobdetect_areamin,
                                                                                             bobdetect_convexity, 
                                                                                             self.board_gridrows, 
                                                                                             self.board_gridcolumns)
        
        #Projector calibration
        proj_ret, proj_imgpts,proj_mtx, proj_dist, proj_rvecs, proj_tvecs = self.proj_calib(cam_objpts, cam_imgpts, up_unwrapv_lst, up_unwraph_lst, up_proj_img_lst, self.pitch[-1], phase_st, self.board_gridrows, self.board_gridcolumns) 
        # Camera calibration error analysis
        cam_mean_error, cam_delta, cam_df1 = self.intrinsic_error_analysis(cam_objpts, cam_imgpts, cam_mtx, cam_dist, cam_rvecs, cam_tvecs)
        #Projector calibration error analysis
        proj_mean_error, proj_delta, proj_df1 = self.intrinsic_error_analysis(cam_objpts,proj_imgpts,proj_mtx,proj_dist,proj_rvecs,proj_tvecs)
        #Stereo calibration
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.0001)
        stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC+cv2.CALIB_ZERO_TANGENT_DIST+cv2.CALIB_FIX_K3+cv2.CALIB_FIX_K4+cv2.CALIB_FIX_K5+cv2.CALIB_FIX_K6

        st_retu,st_cam_mtx,st_cam_dist,st_proj_mtx,st_proj_dist,st_cam_proj_rmat,st_cam_proj_tvec,E,F=cv2.stereoCalibrate(cam_objpts,
                                                                                                                          cam_imgpts,
                                                                                                                          proj_imgpts,
                                                                                                                          cam_mtx,cam_dist,
                                                                                                                          proj_mtx,proj_dist,
                                                                                                                          white_lst[0].shape[::-1],
                                                                                                                          flags=stereocalibration_flags,
                                                                                                                          criteria=criteria)
        project_mat=np.hstack((st_cam_proj_rmat,st_cam_proj_tvec))
        _,_,_,_,_,_,euler_angles=cv2.decomposeProjectionMatrix(project_mat)
        
        np.savez(os.path.join(self.path,'{}_calibration_param.npz'.format(self.type_unwrap)), st_cam_mtx,st_cam_dist, st_proj_mtx, st_cam_proj_rmat, st_cam_proj_tvec) 
        np.savez(os.path.join(self.path,'{}_cam_rot_tvecs.npz'.format(self.type_unwrap)),cam_rvecs,cam_tvecs)      
        return up_unwrapv_lst, up_unwraph_lst, up_white_lst, up_mod_lst, up_proj_img_lst, cam_objpts, cam_imgpts, proj_imgpts, euler_angles, cam_mean_error, cam_delta, cam_df1, proj_mean_error, proj_delta, proj_df1 
        
    
    def calib_center_reconstruction(self, cam_imgpts, unwrap_phase):
        '''
        This function is a wrapper function to reconstruct circle centers for each camera pose and compute error 
        with computed world projective coordinates in camera coordinate system.

        Parameters
        ----------
        cam_imgpts = type:float. List of detected circle centers in camera images. 
        unwrap_phase = type: float. Unwrapped phase map

        Returns
        -------
        delta_df = type: pandas dataframe. Data frame of error for each pose all circle centers.
        abs_delta_df = type: pandas dataframe. Data frame of absolute error for each pose all circle centers.
        center_cordi_lst = = type: float array. Array of x,y,z coordinates of detected circle centers in each calibration pose.

        '''
        calibration = np.load(os.path.join(self.path,'{}_calibration_param.npz'.format(self.type_unwrap)))
        c_mtx = calibration["arr_0"]
        c_dist = calibration["arr_1"]
        p_mtx = calibration["arr_2"]
        cp_rot_mtx = calibration["arr_3"]
        cp_trans_mtx = calibration["arr_4"]
        vectors = np.load(os.path.join(self.path,'{}_cam_rot_tvecs.npz'.format(self.type_unwrap)))
        rvec = vectors["arr_0"]
        tvec = vectors["arr_1"]
        if self.type_unwrap == 'phase':
            phase_st = -np.pi
        elif (self.type_unwrap == 'multifreq') or (self.type_unwrap == 'multiwave'):
            phase_st = 0
        #Function call to get all circle center x,y,z coordinates
        center_cordi_lst = self.center_xyz(cam_imgpts, unwrap_phase, 
                                              c_mtx, c_dist, p_mtx, cp_rot_mtx, cp_trans_mtx, 
                                              phase_st, self.pitch[-1])
        true_cordinates = self.world_points(self.dist_betw_circle, self.board_gridrows, self.board_gridcolumns)
        
        #Function call to get projective xyz for each pose
        proj_xyz_arr = self.project_validation(rvec, tvec, true_cordinates)
        
        #Function call to get projective xyz for each pose
        proj_xyz_arr = self.project_validation(rvec, tvec, true_cordinates)
        
        # Error dataframes
        delta_df, abs_delta_df = self.center_err_analysis(center_cordi_lst, proj_xyz_arr)
        
        return delta_df, abs_delta_df, center_cordi_lst
    
    

    def world_points(self,dist_betw_circle, board_gridrows, board_gridcolumns):
        '''
        Function to generate world coordinate for asymmetric circle center calibration of camera and projector.
    
        Parameters
        ----------
        dist_betw_circle = type: float. Distance between circle centers.
        board_gridrows = type: int. Number of rows in the assymetric circle pattern.
        board_gridcolumns = type: int. Number of columns in the asymmetric circle pattern.
    
        Returns
        -------
        coord = type: float. Array of world coordinates.
    
        '''
        col1 = np.append(np.tile([0,0.5], int((board_gridcolumns-1) / 2)),0).reshape(board_gridcolumns, 1)
        col2 = np.ones((board_gridcolumns, board_gridrows)) * np.arange(0, board_gridrows)
        col_mat = col1 + col2
    
        row_mat = (0.5 * np.arange(0, board_gridcolumns).reshape(board_gridcolumns, 1))@np.ones((1, board_gridrows))
        zer = np.zeros((board_gridrows * board_gridcolumns))
        coord = np.column_stack((row_mat.ravel(), col_mat.ravel(), zer)) * dist_betw_circle
        return coord.astype('float32')

    def projcam_calib_img_phase(self,no_pose, limit, N, pitch, width, height, kernel_v, kernel_h, path):
        '''
        Function is used to generate absolute phase maps and true (single channel gray) images 
        (object image without fringe patterns) from fringe image for camera and projector calibration 
        from raw captured images using phase coded temporal unwrapping method.
        'no_pose' is the total number of poses used for calibration. 
        Hence the function generates 'no_pose' number of absolute phase maps and true images. 
    
        Parameters
        ----------
        no_pose = type: int. Total number of calibration poses. 
        limit = type: float. Data modulation limit. Regions with data modulation lower than limit will be masked out.
        N = type: int. The number of steps in phase shifting algorithm.
        pitch = type: float. Number of pixels per fringe period.
        width  = type: float. Width of pattern image.
        height = type: float. Height of the pattern image.
        kernel_v = type: int. Kernel size for median filter to be applied in the horizontal direction.
        kernel_h = type: int. Kernel size for median filter to be applied in the vertical direction.
        path = type: string. Path to read captured calibration images.
    
        Returns
        -------
        unwrap_v_lst = type:list of float. List of unwrapped phase maps obtained from horizontally varying intensity patterns.
        unwrap_h_lst = type:list of float. List of unwrapped phase maps obtained from vertically varying intensity patterns
        white_lst = type:list of float. List of true images for each calibration pose.
        avg_lst = type: list of float. List of average intensity images for each calibration pose.
        mod_lst = type: list of float. List of modulation intensity images for each calibration pose.
        gamma_lst = type: list of float. List of data modulation (relative modulation) images for each calibration pose.
        kv_lst = type: list of int. List of fringe order for horizontally varying intensity for each calibration pose.
        kh_lst = type: list of int. List of fringe order for vertically varying intensity for each calibration pose.
        coswrapv_lst = type: list of float. List of wrapped phase map for cosine variation of intensity in the 
                                            horizontal direction.
        coswraph_lst = type: list of float. List of wrapped phase map for cosine variation of intensity in the 
                                            vertical direction.
        stepwrapv_lst = type: list of float. List of wrapped phase map for stair  phase coded pattern 
                                            varying in the horizontal direction.
        stepwraph_lst = type: list of float. List of wrapped phase map for stair  phase coded pattern 
                                            varying in the vertical direction.
    
        '''
        unwrap_v_lst=[]
        unwrap_h_lst=[]
        white_lst=[]
        avg_lst=[]
        mod_lst=[]
        gamma_lst=[]
        kv_lst=[]
        kh_lst=[]
        coswrapv_lst=[]
        coswraph_lst=[]
        stepwrapv_lst=[]
        stepwraph_lst=[]
        prefix=path
        for x in tqdm(range (0,no_pose),desc='generating unwrapped phases map for {} images'.format(no_pose)):  
            if os.path.exists(os.path.join(prefix,'capt%d_0.jpg'%x)):
                #Read and apply mask to each captured images for cosine and stair patterns
                cos_v_int8,  mod1, avg1, gamma1, capt_delta_deck1 = nstep.mask_img(np.array([cv2.imread(os.path.join(path,'capt%d_%d.jpg'%(x, i)),0) for i in range(0,N)]), limit)
                cos_h_int8,  mod2, avg2, gamma2, capt_delta_deck2 = nstep.mask_img(np.array([cv2.imread(os.path.join(path,'capt%d_%d.jpg'%(x, i)),0) for i in range(N,2*N)]), limit)
                step_v_int8, mod3, avg3, gamma3, capt_delta_deck3 = nstep.mask_img(np.array([cv2.imread(os.path.join(path,'capt%d_%d.jpg'%(x, i)),0) for i in range(2*N,3*N)]), limit)
                step_h_int8, mod4, avg4, gamma4, capt_delta_deck4 = nstep.mask_img(np.array([cv2.imread(os.path.join(path,'capt%d_%d.jpg'%(x, i)),0) for i in range(3*N,4*N)]), limit)
                
                
                unwrap_v, unwrap_h, k0_v, k0_h, cos_wrap_v, cos_wrap_h, step_wrap_v, step_wrap_h = nstep.ph_temp_unwrap(cos_v_int8, cos_h_int8, 
                                                                                                              step_v_int8, step_h_int8,
                                                                                                              pitch, height, width, capt_delta_deck1,
                                                                                                              kernel_v, kernel_h)
                # True image for a given pose  
                orig_img = (avg2 ) + (mod2 )
                unwrap_v_lst.append(unwrap_v)
                unwrap_h_lst.append(unwrap_h)
                white_lst.append(orig_img)
                avg_lst.append(np.array([avg1, avg2, avg3, avg4]))
                mod_lst.append(np.array([mod1, mod2, mod3, mod4]))
                gamma_lst.append(np.array([gamma1, gamma2, gamma3, gamma4]))
                kv_lst.append(k0_v)
                kh_lst.append(k0_h)
                coswrapv_lst.append(cos_wrap_v)
                coswraph_lst.append(cos_wrap_h)
                stepwrapv_lst.append(step_wrap_v)
                stepwraph_lst.append(step_wrap_h)
                wrapped_phase_lst = {
                    "wrapv":coswrapv_lst,
                    "wraph":coswraph_lst,
                    "stepwrapv":stepwrapv_lst,
                    "stepwraph":stepwraph_lst
                    }
                
        return unwrap_v_lst, unwrap_h_lst, white_lst, avg_lst, mod_lst, gamma_lst, wrapped_phase_lst

    def projcam_calib_img_multifreq(self, no_pose, limit, N_list, pitch_list, width, height, kernel_v, kernel_h, path):
        '''
        Function is used to generate absolute phase map and true (single channel gray) images 
        (object image without fringe patterns)from fringe image for camera and projector calibration from raw captured images 
        using multifrequency temporal unwrapping method.
    
        Parameters
        ----------
        no_pose = type: int. Total number of calibration poses.
        limit = type: float. Data modulation limit. Regions with data modulation lower than limit will be masked out.
        N_list = type: list of float. List of number of steps for each wavelength.
        pitch_list = type: list of wavelengths. List of number of fringes per pixel or wavengths.
        width = type: float. Width of image.
        height = type: float. Height of image.
        path = type: string. Path to read raw captured images of fringe patterns on object.
    
        Returns
        -------
        unwrapv_lst = type:list of float. List of unwrapped phase maps obtained from horizontally varying intensity patterns.
        unwraph_lst = type:list of float. List of unwrapped phase maps obtained from vertically varying intensity patterns.
        white_lst = type:list of float. List of true images for each calibration pose. 
        avg_vlst = type: list of float. List of average intensity images for each calibration pose for horizontally 
                                        varying intensity.
        avg_hlst = type: list of float. List of average intensity images for each calibration pose for vertically 
                                        varying intensity.
        mod_vlst = type: list of float. List of modulation intensity images for each calibration pose for horizontally 
                                        varying intensity.
        mod_hlst = type: list of float. List of modulation intensity images for each calibration pose for vertically 
                                        varying intensity.
        gamma_vlst = type: list of float. List of data modulation (relative modulation) images for each calibration pose 
                                        for horizontally varying intensity.
        gamma_hlst = type: list of float. List of data modulation (relative modulation) images for each calibration pose 
                                        for vertically varying intensity.
        wrapv_lst = type: list of float. List of wrapped phase maps for each calibration pose for horizontally varying intensity.
        wraph_lst = type: list of float. List of wrapped phase maps for each calibration pose for vertically varying intensity.
        kv_lst = type: list of int. List of fringe order for each calibration pose for horizontally varying intensity. 
        kh_lst = type: list of int. List of fringe order for each calibration pose for vertically varying intensity. 
    
        '''
        
        avg_lst = []
        mod_lst = []
        gamma_lst = []
        white_lst = []
        kv_lst = []
        kh_lst = []
        wrapv_lst = []
        wraph_lst = []
        unwrapv_lst = []
        unwraph_lst = []
        
        prefix=path
        for x in tqdm(range (0,no_pose),desc='generating unwrapped phases map for {} images'.format(no_pose)):  
            if os.path.exists(os.path.join(prefix,'capt%d_0.jpg'%x)):
                multi_cos_v_int1, multi_mod_v1, multi_avg_v1, multi_gamma_v1 , multi_delta_deck_v1 = nstep.mask_img(np.array([cv2.imread(os.path.join(path,'capt%d_%d.jpg'%(x,i)),0) for i in range(0, N_list[0])]), limit)
                multi_cos_h_int1, multi_mod_h1, multi_avg_h1, multi_gamma_h1, multi_delta_deck_h1 = nstep.mask_img(np.array([cv2.imread(os.path.join(path,'capt%d_%d.jpg'%(x,i)),0) for i in range(N_list[0],2 * N_list[0])]), limit)
                multi_cos_v_int2, multi_mod_v2, multi_avg_v2, multi_gamma_v2, multi_delta_deck_v2 = nstep.mask_img(np.array([cv2.imread(os.path.join(path,'capt%d_%d.jpg'%(x,i)),0) for i in range(2 * N_list[0],2 * N_list[0] + N_list[1])]), limit)
                multi_cos_h_int2, multi_mod_h2, multi_avg_h2, multi_gamma_h2, multi_delta_deck_h2 = nstep.mask_img(np.array([cv2.imread(os.path.join(path,'capt%d_%d.jpg'%(x,i)),0) for i in range(2 * N_list[0] + N_list[1],2 * N_list[0] + 2 * N_list[1])]), limit)
                multi_cos_v_int3, multi_mod_v3, multi_avg_v3, multi_gamma_v3, multi_delta_deck_v3 = nstep.mask_img(np.array([cv2.imread(os.path.join(path,'capt%d_%d.jpg'%(x,i)),0) for i in range(2 * N_list[0] + 2 * N_list[1] ,2 * N_list[0] + 2 * N_list[1] + N_list[2])]), limit)
                multi_cos_h_int3, multi_mod_h3, multi_avg_h3, multi_gamma_h3, multi_delta_deck_h3 = nstep.mask_img(np.array([cv2.imread(os.path.join(path,'capt%d_%d.jpg'%(x,i)),0) for i in range(2 * N_list[0] + 2 * N_list[1] + N_list[2],2 * N_list[0] + 2 * N_list[1] + 2 * N_list[2])]), limit)
                multi_cos_v_int4, multi_mod_v4, multi_avg_v4, multi_gamma_v4, multi_delta_deck_v4 = nstep.mask_img(np.array([cv2.imread(os.path.join(path,'capt%d_%d.jpg'%(x,i)),0) for i in range(2 * N_list[0] + 2 * N_list[1] + 2 * N_list[2],2 * N_list[0] + 2 * N_list[1] +2 *  N_list[2]+ N_list[3])]), limit)
                multi_cos_h_int4, multi_mod_h4, multi_avg_h4, multi_gamma_h4, multi_delta_deck_h4 = nstep.mask_img(np.array([cv2.imread(os.path.join(path,'capt%d_%d.jpg'%(x,i)),0) for i in range(2 * N_list[0] + 2 * N_list[1] +2 *  N_list[2]+ N_list[3],2 * N_list[0] + 2 * N_list[1] +2 *  N_list[2]+ 2 * N_list[3])]), limit)
                
                orig_img = (multi_avg_h4 ) + (multi_mod_h4 )
                
                multi_phase_v1 = nstep.phase_cal(multi_cos_v_int1, N_list[0], multi_delta_deck_v1 )
                multi_phase_h1 = nstep.phase_cal(multi_cos_h_int1, N_list[0], multi_delta_deck_h1 )
                multi_phase_v2 = nstep.phase_cal(multi_cos_v_int2, N_list[1], multi_delta_deck_v2 )
                multi_phase_h2 = nstep.phase_cal(multi_cos_h_int2, N_list[1], multi_delta_deck_h2 )
                multi_phase_v3 = nstep.phase_cal(multi_cos_v_int3, N_list[2], multi_delta_deck_v3 )
                multi_phase_h3 = nstep.phase_cal(multi_cos_h_int3, N_list[2], multi_delta_deck_h3 )
                multi_phase_v4 = nstep.phase_cal(multi_cos_v_int4, N_list[3], multi_delta_deck_v4 )
                multi_phase_h4 = nstep.phase_cal(multi_cos_h_int4, N_list[3], multi_delta_deck_h4 )                    

                #index_v = np.argmax(abs(np.diff(multi_phase_v1, axis = 0)))
                #index_h = np.argmax(abs(np.diff(multi_phase_h1, axis = 1)))
                #multi_phase_v1[:,index_v:] = multi_phase_v1[:,index_v:] + 2 * np.pi
                #multi_phase_h1[index_h:] = multi_phase_h1[index_h:] + 2 * np.pi
                multi_phase_v1[multi_phase_v1< EPSILON] = multi_phase_v1[multi_phase_v1 < EPSILON ] + 2 * np.pi
                multi_phase_h1[multi_phase_h1< EPSILON] = multi_phase_h1[multi_phase_h1 < EPSILON ] + 2 * np.pi 
                
                phase_arr_v = [multi_phase_v1, multi_phase_v2, multi_phase_v3, multi_phase_v4]
                phase_arr_h = [multi_phase_h1, multi_phase_h2, multi_phase_h3, multi_phase_h4]
                
                multifreq_unwrap_v, k_arr_v = nstep.multifreq_unwrap(pitch_list, phase_arr_v, kernel_v, 'v')
                multifreq_unwrap_h, k_arr_h = nstep.multifreq_unwrap(pitch_list, phase_arr_h, kernel_h, 'h')                
               
                avg_lst.append(np.array([multi_avg_v1, multi_avg_v2, multi_avg_v3, multi_avg_v4,multi_avg_h1, multi_avg_h2, multi_avg_h3, multi_avg_h4]))
                mod_lst.append(np.array([multi_mod_v1, multi_mod_v2, multi_mod_v3, multi_mod_v4, multi_mod_h1, multi_mod_h2, multi_mod_h3, multi_mod_h4]))
                gamma_lst.append(np.array([multi_gamma_v1, multi_gamma_v2, multi_gamma_v3, multi_gamma_v4, multi_gamma_h1, multi_gamma_h2, multi_gamma_h3, multi_gamma_h4]))
                
                white_lst.append(orig_img)
                
                wrapv_lst.append(phase_arr_v)
                wraph_lst.append(phase_arr_h)
                
                wrapped_phase_lst = {"wrapv":wrapv_lst,
                                    "wraph":wraph_lst
                                    }
                
                kv_lst.append(k_arr_v)
                kh_lst.append(k_arr_h)
                
                unwrapv_lst.append(multifreq_unwrap_v)
                unwraph_lst.append(multifreq_unwrap_h)
                
        return unwrapv_lst, unwraph_lst, white_lst, avg_lst, mod_lst,  gamma_lst, wrapped_phase_lst

    def projcam_calib_img_multiwave(self,no_pose, limit, N_arr, pitch_arr, width, height, kernel_v, kernel_h, path):
        '''
        Function is used to generate absolute phase map and true (single channel gray) images (object image without 
        fringe patterns) from fringe image for camera and projector calibration from raw captured images using 
        multiwave temporal unwrapping method.
    
        Parameters
        ----------
        no_pose = type: int. Total number of calibration poses.
        limit = type: float. Data modulation limit. Regions with data modulation lower than limit will be masked out.
        N_arr = type: array of float. Array of number of steps for each wavelength.
        pitch_arr = type: array of wavelengths. Array of number of fringes per pixel or wavengths.
        width = type: float. Width of image.
        height = type: float. Height of image.
        kernel_v = type: int. Kernel size for median filter to be applied in the horizontal direction.
        kernel_h = type: int. Kernel size for median filter to be applied in the vertical direction.
        path = type: string. Path to read raw captured images of fringe patterns on object.
    
        Returns
        -------
        unwrapv_lst = type:list of float. List of unwrapped phase maps obtained from horizontally varying intensity patterns.
        unwraph_lst = type:list of float. List of unwrapped phase maps obtained from vertically varying intensity patterns.
        white_lst = type:list of float. List of true images for each calibration pose. 
        avg_vlst = type: list of float. List of average intensity images for each calibration pose for horizontally 
                                        varying intensity.
        avg_hlst = type: list of float. List of average intensity images for each calibration pose for vertically 
                                        varying intensity.
        mod_vlst = type: list of float. List of modulation intensity images for each calibration pose for horizontally 
                                        varying intensity.
        mod_hlst = type: list of float. List of modulation intensity images for each calibration pose for vertically 
                                        varying intensity.
        gamma_vlst = type: list of float. List of data modulation (relative modulation) images for each calibration pose 
                                        for horizontally varying intensity.
        gamma_hlst = type: list of float. List of data modulation (relative modulation) images for each calibration pose 
                                        for vertically varying intensity.
        wrapv_lst = type: list of float. List of wrapped phase maps for each calibration pose for horizontally 
                                        varying intensity.
        wraph_lst = type: list of float. List of wrapped phase maps for each calibration pose for vertically varying intensity.
        kv_lst = type: list of int. List of fringe order for each calibration pose for horizontally varying intensity. 
        kh_lst = type: list of int. List of fringe order for each calibration pose for vertically varying intensity. 
        
    
        '''
        
        avg_lst = []
        mod_lst = []
        gamma_lst = []
        white_lst = []
        kv_lst = []
        kh_lst = []
        wrapv_lst = []
        wraph_lst = []
        unwrapv_lst = []
        unwraph_lst = []
        
        eq_wav12 = (pitch_arr[-1] * pitch_arr[1]) / (pitch_arr[1]-pitch_arr[-1])
        eq_wav123 = pitch_arr[0] *eq_wav12 / (pitch_arr[0] - eq_wav12)
        
        pitch_arr=np.insert(pitch_arr,0,eq_wav123)
        pitch_arr=np.insert(pitch_arr,2,eq_wav12)
        
        prefix=path
        for x in tqdm(range (0,no_pose),desc='generating unwrapped phases map for {} images'.format(no_pose)):  
            if os.path.exists(os.path.join(prefix,'capt%d_0.jpg'%x)):
                multi_cos_v_int3, multi_mod_v3, multi_avg_v3, multi_gamma_v3, multi_delta_deck_v3 = nstep.mask_img(np.array([cv2.imread(os.path.join(path,'capt%d_%d.jpg'%(x,i)),0) for i in range(0, N_arr[0])]), limit)
                multi_cos_h_int3, multi_mod_h3, multi_avg_h3, multi_gamma_h3, multi_delta_deck_h3 = nstep.mask_img(np.array([cv2.imread(os.path.join(path,'capt%d_%d.jpg'%(x,i)),0) for i in range(N_arr[0],2 * N_arr[0])]), limit)
                multi_cos_v_int2, multi_mod_v2, multi_avg_v2, multi_gamma_v2, multi_delta_deck_v2 = nstep.mask_img(np.array([cv2.imread(os.path.join(path,'capt%d_%d.jpg'%(x,i)),0) for i in range(2 * N_arr[0],2 * N_arr[0] + N_arr[1])]), limit)
                multi_cos_h_int2, multi_mod_h2, multi_avg_h2, multi_gamma_h2, multi_delta_deck_h2 = nstep.mask_img(np.array([cv2.imread(os.path.join(path,'capt%d_%d.jpg'%(x,i)),0) for i in range(2 * N_arr[0] + N_arr[1],2 * N_arr[0] + 2 * N_arr[1])]), limit)
                multi_cos_v_int1, multi_mod_v1, multi_avg_v1, multi_gamma_v1, multi_delta_deck_v1 = nstep.mask_img(np.array([cv2.imread(os.path.join(path,'capt%d_%d.jpg'%(x,i)),0) for i in range(2 * N_arr[0] + 2 * N_arr[1],2 * N_arr[0] + 2 * N_arr[1] + N_arr[2])]), limit)
                multi_cos_h_int1, multi_mod_h1, multi_avg_h1, multi_gamma_h1, multi_delta_deck_h1 = nstep.mask_img(np.array([cv2.imread(os.path.join(path,'capt%d_%d.jpg'%(x,i)),0) for i in range(2 * N_arr[0] + 2 * N_arr[1]+ N_arr[2],2 * N_arr[0] + 2 * N_arr[1] + 2 * N_arr[2])]), limit)
               
                
                orig_img = (multi_avg_h1 ) + (multi_mod_h1)
                
                multi_phase_v3 = nstep.phase_cal(multi_cos_v_int3, N_arr[0], multi_delta_deck_v3 )
                multi_phase_h3 = nstep.phase_cal(multi_cos_h_int3, N_arr[0], multi_delta_deck_h3 )
                multi_phase_v2 = nstep.phase_cal(multi_cos_v_int2, N_arr[1], multi_delta_deck_v2 )
                multi_phase_h2 = nstep.phase_cal(multi_cos_h_int2, N_arr[1], multi_delta_deck_h2 )
                multi_phase_v1 = nstep.phase_cal(multi_cos_v_int1, N_arr[2], multi_delta_deck_v1 )
                multi_phase_h1 = nstep.phase_cal(multi_cos_h_int1, N_arr[2], multi_delta_deck_h1 )
                
                multi_phase_v12 = np.mod(multi_phase_v1 - multi_phase_v2, 2 * np.pi)
                multi_phase_h12 = np.mod(multi_phase_h1 - multi_phase_h2, 2 * np.pi)
                multi_phase_v123 = np.mod(multi_phase_v12 - multi_phase_v3, 2 * np.pi)
                multi_phase_h123 = np.mod(multi_phase_h12 - multi_phase_h3, 2 * np.pi)
                
                #multi_phase_v123 = nstep.edge_rectification(multi_phase_v123, 'v')
                #multi_phase_h123 = nstep.edge_rectification(multi_phase_h123, 'h')
                
                multi_phase_v123[multi_phase_v123 > TAU] = multi_phase_v123[multi_phase_v123 > TAU] - 2 * np.pi
                multi_phase_h123[multi_phase_h123 > TAU] = multi_phase_h123[multi_phase_h123 > TAU] - 2 * np.pi                
                
                phase_arr_v = [multi_phase_v123, multi_phase_v3, multi_phase_v12, multi_phase_v2, multi_phase_v1]
                phase_arr_h = [multi_phase_h123, multi_phase_h3, multi_phase_h12, multi_phase_h2, multi_phase_h1]
                
                multiwav_unwrap_v, k_arr_v = nstep.multiwave_unwrap(pitch_arr, phase_arr_v, kernel_v, 'v')
                multiwav_unwrap_h, k_arr_h = nstep.multiwave_unwrap(pitch_arr, phase_arr_h, kernel_h, 'h')
                
                
                avg_lst.append(np.array([multi_avg_v3, multi_avg_v2, multi_avg_v1,multi_avg_h3, multi_avg_h2, multi_avg_h1]))
                mod_lst.append(np.array([multi_mod_v3, multi_mod_v2, multi_mod_v1, multi_mod_h3, multi_mod_h2, multi_mod_h1]))
                gamma_lst.append(np.array([multi_gamma_v3, multi_gamma_v2, multi_gamma_v1, multi_gamma_h3, multi_gamma_h2, multi_gamma_h1]))
              
                white_lst.append(orig_img)
                
                wrapv_lst.append(phase_arr_v)
                wraph_lst.append(phase_arr_h)
                
                wrapped_phase_lst = {
                    "wrapv":wrapv_lst,
                    "wraph":wraph_lst
                    }
                
                kv_lst.append(k_arr_v)
                kh_lst.append(k_arr_h)
                
                unwrapv_lst.append(multiwav_unwrap_v)
                unwraph_lst.append(multiwav_unwrap_h)
                
        return unwrapv_lst, unwraph_lst, white_lst, avg_lst, mod_lst, gamma_lst, wrapped_phase_lst
    


    def projector_img(self,unwrap_v_lst, unwrap_h_lst, white_lst, width, height, pitch, phase_st):
        '''
        Function to generate projector image using absolute phase maps from horizontally and vertically varying patterns. 
    
        Parameters
        ----------
        unwrap_v_lst = type:list of float. List of unwrapped absolute phase map from horizontally varying pattern. 
        unwrap_h_lst = type:list of float. List of unwrapped absolute phase map from vertically varying pattern.
        white_lst = type:list of float. List of true object image (without patterns).
        width = type: float. Width of projector image.
        height = type: float. Height of projector image.
        pitch = type: float. Number of pixels per fringe period. 
        phase_st = type: float. Starting phase. To apply multifrequency and multiwavelength temporal unwraping 
                                starting phase should be zero. Whereas for phase coding trmporal 
                                nwrapping starting phase should be -π.
    
        Returns
        -------
        proj_img = type: float. Projector image.
    
        '''
        proj_img=[];
        for i in tqdm(range(0,len(unwrap_v_lst)),desc='projector images'):
            # Convert phase map to coordinates
            unwrap_proj_u = (unwrap_v_lst[i] - phase_st) * pitch / (2 * np.pi)
            unwrap_proj_v = (unwrap_h_lst[i] - phase_st) * pitch / (2 * np.pi)
            unwrap_proj_u = unwrap_proj_u.astype(int)
            unwrap_proj_v = unwrap_proj_v.astype(int)
            
            orig_u = unwrap_proj_u.ravel()
            orig_v = unwrap_proj_v.ravel()
            orig_int = white_lst[i].ravel()
            orig_data = np.column_stack((orig_u, orig_v, orig_int))
            orig_df = pd.DataFrame(orig_data, columns = ['u','v','int'])
            orig_new = orig_df.groupby(['u', 'v'])['int'].mean().reset_index()
            
            proj_y = np.arange(0, height)
            proj_x = np.arange(0, width)
            proj_u,proj_v = np.meshgrid(proj_x, proj_y)
            proj_data = np.column_stack((proj_u.ravel(), proj_v.ravel()))
            proj_df = pd.DataFrame(proj_data,columns = ['u','v'])
    
            proj_df_merge = pd.merge(proj_df, orig_new, how = 'left', on = ['u','v'])
            proj_df_merge['int'] = proj_df_merge['int'].fillna(0)
    
            proj_mean_img = proj_df_merge['int'].to_numpy()
            proj_mean_img = proj_mean_img.reshape(height,width)
            proj_mean_img = cv2.normalize(proj_mean_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            proj_img.append(proj_mean_img)
            
        return proj_img
    
            
    
    def camera_calib(self, objp, white_lst, bobdetect_areamin, bobdetect_convexity, grid_row, grid_column):
        '''
        Function to calibrate camera using asymmetric circle pattern. 
        OpenCV bob detector is used to detect circle centers which is used for calibration.
    
        Parameters
        ----------
        objp = type: array of float. World object coordinate. 
        white_lst = type: list of float. List of calibration poses used for calibrations.
        bobdetect_areamin = type: float. Minimum area of area for bob detector. 
        bobdetect_convexity = type: float. Circle convexity for bob detector.
        grid_row = type: int. Number of rows in the asymmetric circle pattern.
        grid_column = type: int. Number of columns in the asymmetric circle pattern.
    
        Returns
        -------
        r_error = type:float. Average reprojection error.
        objpoints = type: float. List of image objectpoints for each pose.
        cam_imgpoints = type: float. List of circle center grid coordinates for each pose.
        cam_mtx = type: float. Camera matrix from calibration.
        cam_dist = type: float. Camera distortion array from calibartion.
        cam_rvecs = type: float. Array of rotation vectors for each calibration pose.
        cam_tvecs = type: float. Array of translational vectors for each calibration pose.
    
        '''
        #Set bob detector properties
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        blobParams = cv2.SimpleBlobDetector_Params()
        
        #color
        blobParams.filterByColor=True
        blobParams.blobColor=255
    
        # Filter by Area.
        blobParams.filterByArea = True
        blobParams.minArea = bobdetect_areamin#2000
        
        #Convexity
        blobParams.filterByConvexity = True
        blobParams.minConvexity = bobdetect_convexity
        
        blobDetector = cv2.SimpleBlobDetector_create(blobParams)
        objpoints = [] # 3d point in real world space
        cam_imgpoints = [] # 2d points in image plane.
        found = 0
        
        cv2.startWindowThread()
        count_lst = []
        ret_lst = []
        
        for i,white in enumerate(white_lst):
            # Convert float image to uint8 type image.
            white = cv2.normalize(white, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) 
            white_color = cv2.cvtColor(white,cv2.COLOR_GRAY2RGB) #only for drawing purpose
            keypoints = blobDetector.detect(white) # Detect blobs.
          
            # Draw detected blobs as green circles. This helps cv2.findCirclesGrid() .
            im_with_keypoints = cv2.drawKeypoints(white_color, keypoints, np.array([]), (0,255,0), 
                                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                                 )
            im_with_keypoints_gray = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)
            
            ret, corners = cv2.findCirclesGrid(im_with_keypoints_gray, (grid_row, grid_column), None, 
                                               flags = cv2.CALIB_CB_ASYMMETRIC_GRID+cv2.CALIB_CB_CLUSTERING,
                                              blobDetector = blobDetector)# Find the circle grid
            ret_lst.append(ret)
            
            if ret == True:
        
                objpoints.append(objp)  # Certainly, every loop objp is the same, in 3D.
                
                cam_imgpoints.append(corners)
                # Draw and display the centers.
                im_with_keypoints = cv2.drawChessboardCorners(white_color, (grid_row, grid_column), corners, ret)# circles
                count_lst.append(found)
                found += 1
                
            cv2.imshow("Camera calibration", im_with_keypoints) # display
            cv2.waitKey(500)
    
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        if not all(ret_lst) == True:
            print('Warning: Centers are not detected for some poses. Modify bobdetect_areamin and bobdetect_areamin parameter')
        #set flags to have tangential distortion = 0, k4 = 0, k5 = 0, k6 = 0 
        flags = cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6
        #camera calibration
        cam_ret, cam_mtx, cam_dist, cam_rvecs, cam_tvecs = cv2.calibrateCamera(objpoints, 
                                                                             cam_imgpoints, white_lst[0].shape[::-1], 
                                                                               None, None, flags=flags,criteria = criteria)
       
        #Average reprojection error
        tot_error=0
        for i in range(len(objpoints)):
            cam_img2,_= cv2.projectPoints(objpoints[i], cam_rvecs[i], cam_tvecs[i], cam_mtx,cam_dist )
            error = cv2.norm(cam_imgpoints[i], cam_img2, cv2.NORM_L2) / len(cam_img2)
            tot_error += error
        r_error = tot_error/len(objpoints)
        print("Reprojection error:",r_error)
       
        return r_error, objpoints, cam_imgpoints,cam_mtx, cam_dist, cam_rvecs, cam_tvecs
    
    
    
    def proj_calib(self, cam_objpts, cam_imgpts, 
                   unwrap_v_lst, unwrap_h_lst, 
                   proj_img_lst, pitch, phase_st, 
                   grid_row, grid_column):
        '''
        Function to calibrate projector by using absolute phase maps. 
        Circle centers detected using OpenCV is mapped to the absolute phase maps and the corresponding projector image coordinate for the centers are calculated.
    
        Parameters
        ----------
        cam_objpts = type: list of float. List of world cordintaes used for camera calibration for each pose.
        cam_imgpts = type: list of float. List of circle centers grid for each calibration pose.
        unwrap_v_lst = type: list of float. List of absolute phase maps for horizontally varying patterns for 
                                            each calibration pose.
        unwrap_h_lst = type: list of float. List of absolute phase maps for vertically varying patterns for 
                                            each calibration pose.
        proj_img_lst = type: list of float. List of computed projector image for each calibration pose.
        pitch = type: float. Number of pixels per fringe period. 
        phase_st = type:float. Initial phase to be subtracted for phase to coordinate conversion.
        grid_row = type: int. Number of rows in the asymmetric circle pattern.
        grid_column = type: int. Number of columns in the asymmetric circle pattern.
    
        Returns
        -------
        r_error : = type:float. Average reprojection error.
        proj_imgpts = type: float. List of circle center grid coordinates for each pose of projector calibration.
        proj_mtx type: float. Projector matrix from calibration.
        proj_dist = type: float. Projector distortion array from calibartion.
        proj_rvecs = type: float. Array of rotation vectors for each calibration pose.
        proj_tvecs = type: float. Array of translational vectors for each calibration pose.
    
        '''
        centers = [i.reshape(cam_objpts[0].shape[0],2) for i in cam_imgpts]
        proj_imgpts = []
        for x,c in enumerate(centers):
            #Phase to coordinate conversion for each pose
            u = [(nstep.bilinear_interpolate(unwrap_v_lst[x],i) - phase_st) * (pitch / (2*np.pi)) for i in c]
            v = [(nstep.bilinear_interpolate(unwrap_h_lst[x],i) - phase_st) * (pitch / (2*np.pi)) for i in c]
            coordi = np.column_stack((u, v)).reshape(cam_objpts[0].shape[0],1,2).astype(np.float32)
            proj_imgpts.append(coordi)
            proj_color = cv2.cvtColor(proj_img_lst[x],cv2.COLOR_GRAY2RGB) #only for drawing
            proj_keypoints = cv2.drawChessboardCorners(proj_color, (grid_row, grid_column), coordi, True)
            cv2.imshow("Projector calibration", proj_keypoints) # display
            cv2.waitKey(500)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        #Set all distortion =0. linear model assumption
        flags=cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6
       
        #Projector calibration
        proj_ret, proj_mtx, proj_dist, proj_rvecs, proj_tvecs = cv2.calibrateCamera(cam_objpts, 
                                                                              proj_imgpts,proj_img_lst[x].shape[::-1], 
                                                                              None, None, flags=flags,
                                                                              criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 2e-16))
        tot_error=0
        #Average reprojection error
        for i in range(len(cam_objpts)):
            proj_img2,_ = cv2.projectPoints(cam_objpts[i], proj_rvecs[i], proj_tvecs[i], proj_mtx,proj_dist )
            error = cv2.norm(proj_imgpts[i], proj_img2, cv2.NORM_L2) / len(proj_img2)
            tot_error += error
        r_error = tot_error / len(cam_objpts)
        print("Reprojection error:",r_error)
        
        return r_error, proj_imgpts, proj_mtx, proj_dist, proj_rvecs, proj_tvecs
    

    def image_analysis(self, unwrap):
        '''
        Function to plot list of images for calibration diagonastic purpose. Eg: To plot list of 
        unwrapped phase maps of all calibration poses.
    
        Parameters
        ----------
        unwrap = type: list of float. List of images.
        
        Returns
        -------
        None.
    
        '''
        for i in range(0,len(unwrap)):
            plt.figure()
            plt.imshow(unwrap[i])
            plt.title('Unwrap phase map',fontsize=20)
        return

    def wrap_profile_analysis(self, wrapped_phase_lst, direc):
        '''
        Function to plot cross-section of calculated wrapped phase map of cosine and stair patterns in 
        phase coded temporal unwrapping method for verification.
    
        Parameters
        ----------
        wrapped_phase_lst = type: dictionary of wrapped phase maps of cosine varying intensity pattern 
        for all calibration poses.
        direc = type: string. vertical (v) or horizontal(h) patterns.
    
        Returns
        -------
        None.
    
        '''
        
        if self.type_unwrap == 'phase':        
            for i in range(0,len(wrapped_phase_lst['wrapv'])):
                if direc == 'v':
                    plt.figure()
                    plt.plot(wrapped_phase_lst['wrapv'][i][600])
                    plt.plot(wrapped_phase_lst['stepwrapv'][i][600])
                elif direc == 'h':
                     plt.figure()
                     plt.plot(wrapped_phase_lst['wraph'][i][:,960])
                     plt.plot(wrapped_phase_lst['stepwraph'][i][:,960])
                plt.title('Wrap phase map %s'%direc,fontsize=20)
                plt.xlabel('Dimension',fontsize=20)
                plt.ylabel('Phase',fontsize=20)
        elif self.type_unwrap in {'multifreq', 'multiwave'}:
            if direc == 'v':
                for phase_arr_v in wrapped_phase_lst['wrapv']:
                    plt.figure()
                    n_subplot = len(phase_arr_v)
                    for i,wrapped_phase in enumerate(phase_arr_v):
                        plt.subplot(n_subplot,1,i+1)
                        plt.plot(wrapped_phase[600])
                    plt.title('Wrap phase map %s'%direc,fontsize=20)
                    plt.xlabel('Dimension',fontsize=20)
                    plt.ylabel('Phase',fontsize=20)                    
            if direc == 'h':
                for phase_arr_h in wrapped_phase_lst['wraph']:
                    plt.figure()
                    n_subplot = len(phase_arr_h)
                    for i,wrapped_phase in enumerate(phase_arr_h):
                        plt.subplot(n_subplot,1,i+1)
                        plt.plot(wrapped_phase[:,960])
                    plt.title('Wrap phase map %s'%direc,fontsize=20)
                    plt.xlabel('Dimension',fontsize=20)
                    plt.ylabel('Phase',fontsize=20)
        else:
            print('Unwrap method is not supported.')

    def intrinsic_error_analysis(self, objpts, imgpts, mtx , dist , rvecs , tvecs ):
        '''
        Function to calculate mean error per calibration pose,reprojection errors and absolute 
        reprojection error in the x and y directions.
        
    
        Parameters
        ----------
        objpts = type: float. List of circle center grid coordinates for each pose of calibration.
        imgpts = type: list of float. List of circle centers grid for each calibration pose.
        mtx = type: float. Device matrix from calibration.
        dist = type: float. Device distortion matrix from calibration..
        rvecs = type: float. Array of rotation vectors for each calibration pose.
        tvecs = type: float. Array of translational vectors for each calibration pose.
    
        Returns
        -------
        mean_error = type: list of floats. List of mean error per calibration pose.
        delta_lst = type: list of float. List of reprojection error.
        abs_df = type: pandas dataframe of floats. Dataframe of absolute error in x and y of all poses
    
        '''
            
        delta_lst = []
        mean_error = []
        abs_error = []
        for i in range(len(objpts)):
            img2,_ = cv2.projectPoints(objpts[i], rvecs[i], tvecs[i], mtx, dist )
            delta = imgpts[i]-img2
            delta_lst.append(delta.reshape(objpts[i].shape[0], 2))
            error = cv2.norm(imgpts[i],img2,cv2.NORM_L2) / len(img2)
            mean_error.append(error)
            abs_error.append(abs(delta).reshape(objpts[i].shape[0],2))
        
        abs_error = np.array(abs_error)
        df_a, df_b, df_c = abs_error.shape
        abs_df = pd.DataFrame(abs_error.reshape(df_a * df_b, df_c), index = np.repeat(np.arange(df_a), df_b),
                            columns = ['absdelta_x', 'absdelta_y'])
        abs_df = abs_df.reset_index().rename(columns = {'index':'image'})
        return mean_error, np.array(delta_lst), abs_df

    def intrinsic_errors_plts(self, mean_error, delta, df, dev):
        '''
        Function to plot mean error per calibration pose, reprojection error and absolute 
        reprojection errors in x and y directions.
    
        Parameters
        ----------
        mean_error = type: list of float. List of mean error per calibration pose of a device.
        delta = type: list of float. List of reprojection error in x and y directions .
        df = type: pandas dataframe. Dataframe of absolute reprojection error for each calibration pose.
        pitch = type: float. Number of pixels per fringe period.
        dev = type: string. Device name: Camera or projector
    
        Returns
        -------
        None.
    
        '''
        xaxis = np.arange(0, len(mean_error), dtype=int)
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.bar(xaxis, mean_error)
        ax.set_title("{} mean error per pose ".format(dev),fontsize=30)
        ax.set_xlabel('Pose',fontsize=20)
        ax.set_ylabel('Pixel)',fontsize=20)
        ax.set_xticks(xaxis)
        plt.xticks(fontsize=15,rotation = 45)
        plt.yticks(fontsize=20)
        plt.figure()
        plt.scatter(delta[:,:,0].ravel(), delta[:,:,1].ravel())
        plt.xlabel('x(pixel)',fontsize=30)
        plt.ylabel('y(pixel)',fontsize=30)
        axes = plt.gca()
        axes.set_aspect(1)
        plt.title('Reprojection error for {} '.format(dev),fontsize=30)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.figure()
        sns.histplot(data = df,x = 'absdelta_x',hue = 'image',multiple = 'stack',palette = 'Paired',legend = False)
        plt.xlabel('Abs($\delta x$)',fontsize = 30)
        plt.ylabel('Count',fontsize = 30)
        plt.title('{} reprojection error x direction '.format(dev),fontsize=30)
        plt.xticks(fontsize = 30)
        plt.yticks(fontsize = 30)
        plt.figure()
        sns.histplot(data = df,x = 'absdelta_y',hue = 'image',multiple = 'stack',palette = 'Paired', legend = False)
        plt.xlabel('Abs($\delta y$)',fontsize = 30)
        plt.ylabel('Count',fontsize = 30)
        plt.title('{} reprojection error y direction '.format(dev), fontsize = 30)
        plt.xticks(fontsize = 30)
        plt.yticks(fontsize = 30)
        return
    # center reconstruction
    def center_xyz(self,center_pts, unwrap_phase, c_mtx, c_dist, p_mtx, cp_rot_mtx, cp_trans_mtx, phase_st, pitch):
        '''
        Function to obtain 3d coordinates of detected circle centers.

        Parameters
        ----------
        center_pts = type: float. List of circle centers of each calibration pose.
        unwrap_phase = type: float. Unwrapped phase maps of each calibration pose.
        c_mtx = type: float array. Camera matrix.
        c_dist = type: float array. Camera distortion matrix.
        p_mtx = type: float array. Projector matrix.
        cp_rot_mtx = type: float array. Camera projector rotation matrix.
        cp_trans_mtx = type: float array. Camera projector translational matrix.
        phase_st = type: float. Starting phase. To apply multifrequency and multiwavelength temporal 
                                unwraping starting phase should be zero. Whereas for phase coding 
                                temporal unwrapping starting phase should be -π.
        pitch = type:float. Number of pixels per fringe period.
        Returns
        -------
        center_cordi_lst = type: float array. Array of x,y,z coordinates of detected circle centers in each calibration pose.

        '''
        center_cordi_lst = []
        for i in tqdm(range (0,len(center_pts)),desc='building camera centers 3d coordinates'):  
            # undistort points
            x, y, z = rc.reconstruction_pts(center_pts[i], 
                                            unwrap_phase[i], 
                                            c_mtx, c_dist, 
                                            p_mtx, 
                                            cp_rot_mtx, cp_trans_mtx, 
                                            phase_st, pitch)
            cordi = np.hstack((x,y,z))
            center_cordi_lst.append(cordi)
        return np.array(center_cordi_lst)

    # Projective coordinates based on camera - projector extrinsics
    def project_validation(self,rvec, tvec, true_cordinates):
        '''
        Function to generate world projective coordinates in camera coordinate system for each 
        calibration pose using pose extrinsics. 

        Parameters
        ----------
        rvec = type: float.  List of rotation vectors for each calibration pose.
        tvec = type: float.  List of translational vectors for each calibration pose.
        true_cordinates = type:float. Defined world coordinates of circle centers.

        Returns
        -------
        proj_xyz_lst = type: float. Array of projective coordinates each calibration poses.

        '''
        t=np.ones((true_cordinates.shape[0],1))
        homo_true_cordi = np.hstack((true_cordinates,t))
        homo_true_cordi = homo_true_cordi.reshape((homo_true_cordi.shape[0],homo_true_cordi.shape[1],1))
        proj_xyz_lst = []
        for i in tqdm(range (0,len(tvec)),desc='projective centers'): 
            rvec_mtx = cv2.Rodrigues(rvec[i])[0]
            h_vecs = np.hstack((rvec_mtx,tvec[i]))
            updated_hvecs = np.repeat(h_vecs[np.newaxis,:,:],true_cordinates.shape[0],axis=0)
            proj_xyz = updated_hvecs @ homo_true_cordi
            proj_xyz = proj_xyz.reshape((proj_xyz.shape[0],proj_xyz.shape[1]))
            proj_xyz_lst.append(proj_xyz)
        return np.array(proj_xyz_lst)


    # Calculate error = center reconstruction - projectivr coordinates
    def center_err_analysis(self,cordi_arr, proj_xyz_arr):
        '''
        Function to compute error of 3d coordinates of detected circle centers from projective coordinates 
        in camera coordinate.

        Parameters
        ----------
        cordi_arr = type: float. Array of x,y,z coordinates of detected circle centers in each calibration pose.
        proj_xyz_arr = type: float. Array of projective coordinates for each calibration poses.

        Returns
        -------
        delta_df =  type: pandas dataframe. Data frame of error for each pose all circle centers.
        abs_delta_df = type: pandas dataframe. Data frame of absolute error for each pose all circle centers.

        '''
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

        n_colors = len(delta_group)  #no. of poses
        cm = plt.get_cmap('gist_rainbow')
        fig=plt.figure(figsize=(16,15))
        fig.suptitle(' Error in reconstructed coordinates compared to true coordinates ',fontsize = 20)
        ax = plt.axes(projection='3d')
        ax.set_prop_cycle(color=[cm(1.*i/n_colors) for i in range(n_colors)])
        cm = plt.get_cmap('gist_rainbow')
        for i in range (0,len(delta_group)):
            x1 = delta_group[i]['delta_x']
            y1 = delta_group[i]['delta_y']
            z1 = delta_group[i]['delta_z']
            ax.scatter(x1,y1,z1, label='Pose %d'%i)
        ax.set_xlabel('$\Delta x$ (mm)',fontsize = 20, labelpad = 10)
        ax.set_ylabel('$\Delta y$ (mm)',fontsize = 20, labelpad = 10)
        ax.set_zlabel('$\Delta z$ (mm)',fontsize = 20, labelpad = 10)
        ax.tick_params(axis = 'x', labelsize = 15)
        ax.tick_params(axis = 'y', labelsize = 15)
        ax.tick_params(axis = 'z', labelsize = 15)
        #ax.legend(loc="upper left",bbox_to_anchor=(1.2, 1),fontsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(self.path,'error.png'))
        fig, ax = plt.subplots()
        fig.suptitle('Abs error histogram of all poses compared to true coordinates', fontsize = 20)
        abs_plot = sns.histplot(abs_delta_df,multiple="layer")
        labels = ['$\Delta x$','$\Delta y$','$\Delta z$']
        mean_deltas = abs_delta_df.mean()
        std_deltas = abs_delta_df.std()
        ax.text(0.7,0.8,'Mean',fontsize=20,horizontalalignment='center',
                 verticalalignment='center',transform = ax.transAxes)
        ax.text(0.85,0.8,'Std',fontsize=20,horizontalalignment='center',
                 verticalalignment='center',transform = ax.transAxes)
        ax.text(0.7,0.75,'$\Delta x $:{0:.3f}'.format(mean_deltas[0]),fontsize=20,horizontalalignment='center',
                 verticalalignment='center',transform = ax.transAxes)
        ax.text(0.85,0.75,'{0:.3f}'.format(std_deltas[0]),fontsize=20,horizontalalignment='center',
                 verticalalignment='center',transform = ax.transAxes)
        ax.text(0.7,0.7,'$\Delta y $:{0:.3f}'.format(mean_deltas[1]),fontsize=20,horizontalalignment='center',
                 verticalalignment='center',transform = ax.transAxes)
        ax.text(0.85,0.7,'{0:.3f}'.format(std_deltas[1]),fontsize=20,horizontalalignment='center',
                 verticalalignment='center',transform = ax.transAxes)
        ax.text(0.7,0.65,'$\Delta z $:{0:.3f}'.format(mean_deltas[2]),fontsize=20,horizontalalignment='center',
                 verticalalignment='center',transform = ax.transAxes)
        ax.text(0.85,0.65,'{0:.3f}'.format(std_deltas[2]),fontsize=20,horizontalalignment='center',
                 verticalalignment='center',transform = ax.transAxes)
        plt.xlabel('abs(error) mm',fontsize = 20)
        plt.ylabel('Count',fontsize = 20)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        for t, l in zip(abs_plot.legend_.texts, labels):
            t.set_text(l)
       
        plt.setp(abs_plot.get_legend().get_texts(), fontsize='20') 
        plt.savefig(os.path.join(self.path,'abs_err.png'),bbox_inches="tight")
        
        return delta_df, abs_delta_df
    
    def recon_xyz(self,unwrap_phase,  
                  white_imgs, 
                  mask_cond, 
                  modulation= None, 
                  int_limit = None, 
                  resid_outlier_limit = None):
        '''
        Function to reconstruct 3d coordinates of calibration board and save as point cloud for each calibration pose. 

        Parameters
        ----------
        unwrap_phase = type: float. Unwrapped phase maps of each calibration pose.
        distance = type: float. Distance between camera - projector system and calibration board. 
        delta_distance = type: float. Volumetric distance around the given object.
        white_imgs = type: float. True intensity image for texture mapping.
        mask_cond = type: string. Mask condition for reconstruction based on 'intensity' or 'modulation' . 
                                  Intensity based mask is applied for reconstructing selected regions based on surface texture. 
                                  Eg: if appropriate int_limit is set and mask_condition =  'intensity' the white 
                                  region of the calibration board can be reconstructed. 
        modulation = type: float. Modulation image for each calibration pose for applying mask to build the calibration board.
                                  Default value is None and used if 'intensity' is used as 'mask_cond'.
        int_limit  = type: float. Minimum intensity value to extract white region. 
        resid_outlier_limit type: float. This parameter is used to eliminate outlier points (points too far).

        Returns
        -------
        cordi_lst = type: float. List of 3d coordinates of board points.
        color_lst = type: float. List of color for each 3d point.

        '''
        calibration = np.load(os.path.join(self.path,'{}_calibration_param.npz'.format(self.type_unwrap)))
        c_mtx = calibration["arr_0"]
        c_dist = calibration["arr_1"]
        p_mtx = calibration["arr_2"]
        cp_rot_mtx = calibration["arr_3"]
        cp_trans_mtx = calibration["arr_4"]
        if self.type_unwrap == 'phase':
            phi0 = -np.pi
        elif (self.type_unwrap == 'multifreq') or (self.type_unwrap == 'multiwave'):
            phi0 = 0
        cordi_lst = []
        color_lst = []
        for i,(u,w) in tqdm(enumerate (zip(unwrap_phase,white_imgs)),desc='building board 3d coordinates'): 
            u_copy = deepcopy(u)
            w_copy = deepcopy(w)
            roi_mask = np.full(u_copy.shape, False)
            if mask_cond == 'modulation': 
                if len(modulation) > 0:
                    roi_mask[modulation[i][-1] > self.limit] = True
                else:
                    print('Please provide modulation images for mask.')
            elif mask_cond == 'intensity' :
                  if w.size != 0:
                      roi_mask[w > int_limit]= True
                  else:
                      print('Please provide intensity (texture) image.')
            else:
                roi_mask[:] = True # all pixels are sellected.
                print("The inpput mask_cond is not supported, no mask is applied.")
            u_copy[~roi_mask] = np.nan
            x, y, z = rc.reconstruction_obj(u_copy, c_mtx, c_dist, p_mtx, cp_rot_mtx, cp_trans_mtx, phi0, self.pitch[-1])
            
            w_copy[~roi_mask] = False
            x[~roi_mask] = np.nan
            y[~roi_mask] = np.nan
            z[~roi_mask] = np.nan
            
            cordi = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
            nan_mask = np.isnan(cordi)
            up_cordi = cordi[~nan_mask.all(axis =1)]
            xyz = list(map(tuple, up_cordi)) 
            inte_img = w_copy / np.nanmax(w_copy)
            inte_rgb = np.stack((inte_img,inte_img,inte_img),axis = -1)
            rgb_intensity_vect = np.vstack((inte_rgb[:,:,0].ravel(), inte_rgb[:,:,1].ravel(),inte_rgb[:,:,2].ravel())).T
            up_rgb_intensity_vect = rgb_intensity_vect[~nan_mask.all(axis =1)]
            color = list(map(tuple, up_rgb_intensity_vect))
            cordi_lst.append(up_cordi)
            color_lst.append(up_rgb_intensity_vect)
            if mask_cond == 'modulation':
                point_cloud_dir = os.path.join(self.path, 'modulation_mask')
                if not os.path.exists(point_cloud_dir):
                    os.makedirs(point_cloud_dir)                
            elif mask_cond == 'intensity' : 
                point_cloud_dir = os.path.join(self.path, 'intensity_mask')
                if not os.path.exists(point_cloud_dir):
                    os.makedirs(point_cloud_dir)
            else:
                point_cloud_dir = os.path.join(self.path, 'no_mask')
                if not os.path.exists(point_cloud_dir):
                    os.makedirs(point_cloud_dir)  
            PlyData(
                [
                    PlyElement.describe(np.array(xyz, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'points'),
                    PlyElement.describe(np.array(color, dtype=[('r', 'f4'), ('g', 'f4'), ('b', 'f4')]), 'color'),
                ]).write(os.path.join(point_cloud_dir,'obj_%d.ply'%i))    
        if mask_cond == 'intensity':
            residual_lst, outlier_lst = self.white_center_planefit(cordi_lst, resid_outlier_limit)
        return cordi_lst, color_lst
    
    def white_center_planefit(self, cordi_lst, resid_outlier_limit):
        '''
        Function to fit plane to extracted white region of calibration board. The function computes the plane and 
        calculates distance of points to the plane (residue) for each calibration pose.
        The function then plots the histogram residue of all calibration poses.

        Parameters
        ----------
        cordi_lst = type: float. List of 3d coordinates of white points.
        resid_outlier_limit type: float. This parameter is used to eliminate outlier points (points too far).

        Returns
        -------
        residual_lst = type float. List of residue from each calibration pose.
        outlier_lst = type: float. List of outlier points from each calibration pose.

        '''
        residual_lst = []
        outlier_lst = []
        for i in tqdm(cordi_lst,desc='residual calculation for each pose'):
            xcord = i[:,0]
            xcord = xcord[~np.isnan(xcord)]
            ycord = i[:,1]
            ycord = ycord[~np.isnan(ycord)]
            zcord = i[:,2]
            zcord = zcord[~np.isnan(zcord)]
            fit,residual = plane_fit(xcord,ycord,zcord)
            outliers = residual [(residual < -resid_outlier_limit)|(residual > resid_outlier_limit)]
            updated_resid = residual[(residual>-resid_outlier_limit) & (residual < resid_outlier_limit)]
            residual_lst.append(updated_resid)
            outlier_lst.append(outliers)
        plane_resid_plot(residual_lst)
        return residual_lst, outlier_lst
    
    def pp_distance_analysis(self, center_cordi_lst, val_label):
        '''
        Function to compute given point to point distance on the calibration board over all calibration poses and 
        plot error plot.

        Parameters
        ----------
        center_cordi_lst = type: float. Array of x,y,z coordinates of detected circle centers in each calibration pose.
        val_label = type: float. Any distance between two circle centers.

        Returns
        -------
        distances = type: pandas dataframe. Dataframe of distance calculated for each calibration pose.

        '''
        dist =[]
        true_cordinates = self.world_points(self.dist_betw_circle, self.board_gridrows, self.board_gridcolumns)
        true_dist = distance.pdist(true_cordinates, 'euclidean')
        true_dist = np.sort(true_dist)
        dist_diff = np.diff(true_dist)
        dist_pos = np.where(dist_diff> abs(0.4))[0] + 1
        dist_split = np.split(true_dist, dist_pos)
        true_val,true_count = np.unique(true_dist, return_counts=True)
        true_val = np.around(true_val,decimals = 3)
        true_df = pd.DataFrame(dist_split).T
        true_df.columns = true_val
        if val_label != true_val.all():
            distances = dist_l(center_cordi_lst, true_val)
            dist.append(distances)
            fig,ax= plt.subplots()
            x = distances[val_label]-val_label
            sns.histplot(x)
            mean = x.mean()
            std = x.std()
            #ax.set_xlim(mean-0.2,mean+0.5)
            ax.text(0.75,0.75,'Mean:{0:.3f}'.format(mean),fontsize=20,horizontalalignment='center',
                     verticalalignment='center',transform = ax.transAxes)
            ax.text(0.75,0.65,'Std:{0:.3f}'.format(std),fontsize=20,horizontalalignment='center',
                     verticalalignment='center',transform = ax.transAxes)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            ax.set_xlabel('Measured error for true value {}mm'.format(val_label),fontsize=20)
            ax.set_ylabel('Count',fontsize=20)
        else:
            print ('Invalid point to point distance. For given calibration board distance values are {}'.format(true_val))
            distances = None
        return distances
      
def dist_l(center_cordi_lst, true_val):
    '''
    Function to build a pandas dataframe of calculated point to point distance between each circle centers.

    Parameters
    ----------
    center_cordi_lst = type: float. Array of x,y,z coordinates of detected circle centers in each calibration pose.
    val_label = type: float. Any distance between two circle centers.

    Returns
    -------
    dist_df : TYPE
        DESCRIPTION.

    '''
    dist_df = pd.DataFrame()
    for i in range(len(center_cordi_lst)):
        dist = distance.pdist(center_cordi_lst[i],'euclidean')
        #group into different distances
        dist = np.sort(dist)
        dist_diff = np.diff(dist)
        dist_pos = np.where(dist_diff> abs(0.4))[0] + 1 # to find each group starting(+1 since difference, [0] np where returns tuple)
        dist_split = np.split(dist, dist_pos)
        temp_df = pd.DataFrame(dist_split).T
        dist_df = dist_df.append(temp_df, ignore_index = True)
    dist_df = dist_df.iloc[:,0:len(true_val)]
    dist_df.columns = true_val
    return dist_df

def obj(X, p):
    '''
    Objective function for plane fitting. Used to calculate distance of a point from a plane.

    Parameters
    ----------
    X = type: float. 3d coordinates array.
    p = type: float. List of plane parameters.

    Returns
    -------
    distances = type: pandas dataframe. Dataframe of distance calculated for each calibration pose.

    '''
    plane_xyz = p[0:3]
    distance = (plane_xyz * X.T).sum(axis = 1) + p[3]
    return distance/ np.linalg.norm(plane_xyz)

def residuals(p, X):
    '''
    Function to compute residuals for optimization.

    Parameters
    ----------
    p = type:float. Plane parameters.
    X = type: float. 3d coordinates.

    Returns
    -------
    Distance of point from plane.

    '''
    return obj(X, p)

def plane_fit(xcord,ycord,zcord):
    '''
    Function to get optimized plane solution and calculate residue of each point with respect to the fitted plane.

    Parameters
    ----------
    xcord = type: float. List of X coordinates of each point.
    ycord = type: float. List of Y coordinates of each point.
    zcord = type: float. List of Z coordinates of each point.

    Returns
    -------
    coeff = type: float. Fitted plane coefficients.
    resid = type: float. Residue of each point from plane.

    '''
    p0 = [1, 1, 1, 1]
    xyz = np.vstack((xcord,ycord,zcord))
    coeff =  leastsq(residuals , p0, args = ( xyz))[0]
    resid = obj(xyz,coeff)
    return coeff, resid  

def plane_resid_plot(residual_lst):
    '''
    Function to generate diagonistic plot of residuls from white region plane fitting.

    Parameters
    ----------
    residual_lst = type: List of rsiduals for each calibration pose.

    Returns
    -------
    None.

    '''
    residuals =  np.concatenate (residual_lst, axis =0)
    rms = np.sqrt(np.mean(residuals**2))
    fig,ax= plt.subplots()
    plt.title('Residual from fitted planes', fontsize = 20)
    sns.histplot(residuals,legend = False)
    mean = np.mean(residuals)
    std = np.std(residuals)
    ax.text(0.75,0.75,'Mean:{0:.3f}'.format(mean),fontsize=20,horizontalalignment='center',
             verticalalignment='center',transform = ax.transAxes)
    ax.text(0.75,0.65,'Std:{0:.3f}'.format(std),fontsize=20,horizontalalignment='center',
             verticalalignment='center',transform = ax.transAxes)
    ax.text(0.75,0.55,'RMS:{0:.3f}'.format(rms),fontsize=20,horizontalalignment='center',
             verticalalignment='center',transform = ax.transAxes)
    ax.set_xlabel('Residual',fontsize = 20)
    ax.set_ylabel('Count',fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    ax.set_xlim(-1,1)
    return
    
        
         
    
        