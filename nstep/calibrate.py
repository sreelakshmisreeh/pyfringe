#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 11:02:18 2022

@author: Sreelakshmi
"""

import calib.py
import cv2

def calibrations(no_pose,limit,N,pitch,width,height,total_step_height,kernel_v,kernel_h,
                 bobmin=2000,bobmax=100000):
    #world coordinates
    objp=calib.asymmetric_world_points()
    #images for calibration. camera: intensity image; proj:unwrapped phase map images 
    unwrap_v_lst,unwrap_h_lst,white_lst=calib.projcam_calib_img(no_pose,limit,N,pitch,
                                                                width,height,
                                                                total_step_height,
                                                                kernel_v,kernel_h)
    #calibration of camera
    camr_error,cam_objpts,cam_imgpts,cam_mtx,cam_dist,cam_rvecs,cam_tvecs=calib.camera_calib(objp,
                                                                                         white_lst,
                                                                                         bobmin,
                                                                                         bobmax)
    
    
    #projector calibration
    centers=[i.reshape(44,2).astype('int') for i in cam_imgpts]
    projr_error,proj_objpts,proj_imgpts,proj_mtx,proj_dist,proj_rvecs,proj_tvecs,missing=calib.proj_calib(objp,
                                                                                                          white_lst,
                                                                                                          unwrap_v_lst,unwrap_h_lst,
                                                                                                          centers,pitch,height,width)
    #Removing outlier images
    cam_imgpts2=[val for n, val in enumerate(cam_imgpts) if n not in missing]
    #Stereo calibrate with same pose in camera and projector
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.0001)
    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
    retu,cam_mtx2,cam_dist2,proj_mtx2,proj_dist2,cam_proj_rmat,cam_proj_tvec,E,F=cv2.stereoCalibrate(proj_objpts,cam_imgpts2,proj_imgpts,
                                                                                                    cam_mtx,cam_dist,proj_mtx,proj_dist,
                                                                                                    white_lst[0].shape[::-1],
                                                                                                    flags=stereocalibration_flags,
                                                                                                    criteria=criteria)
    
    return(cam_mtx2,cam_dist2,proj_mtx2,proj_dist2,cam_proj_rmat, cam_proj_tvec)
    
    
    
    
    

