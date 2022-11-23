# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 15:14:01 2022

@author: kl001
"""
import cv2
import numpy as np
import os
import shutil
import glob
import sys
sys.path.append(r'C:\Users\kl001\pyfringe\functions')
sys.path.append(r'C:\Users\kl001\Documents\pyfringe_test')
import calib
import reconstruction_copy as rc

def copy_tofolder (sample_index, source_folder, dest_folder):
    #empty destination folder
    for f in os.listdir(dest_folder):
        os.remove(os.path.join(dest_folder, f))
    #copy contents    
    to_be_moved =  [glob.glob(os.path.join(source_folder,'capt%d_*.jpg'%x)) for x in sample_index]
    flat_list = [item for sublist in to_be_moved for item in sublist]
    for t in flat_list:
        shutil.copy(t, dest_folder)
    return

def sample_intrinsics_extrinsics(delta_pose,sub_sample_size, no_sample_sets, proj_width, proj_height, limit, type_unwrap, phase_st, N_list, pitch_list, board_gridrows, board_gridcolumns, bobdetect_areamin, bobdetect_convexity, dist_betw_circle, kernel_v, kernel_h, source_folder, dest_folder ):
    left = np.arange(0, delta_pose)
    right = np.arange(delta_pose, 2*delta_pose)
    down = np.arange(2*delta_pose, 3*delta_pose)
    up = np.arange(3*delta_pose, 4*delta_pose)
    cam_mtx_sample = []
    cam_dist_sample = []
    proj_mtx_sample = []
    proj_dist_sample = []
    st_rmat_sample = []
    st_tvec_sample = []
    proj_h_mtx_sample = []
    cam_h_mtx_sample = []
    calib_inst = calib.calibration(proj_width, proj_height, limit, type_unwrap, N_list, pitch_list, board_gridrows, board_gridcolumns, dist_betw_circle, dest_folder)
    for i in range(no_sample_sets):
        sample_index_l = np.random.choice(left, size = sub_sample_size, replace = False)
        sample_index_r = np.random.choice(right, size = sub_sample_size, replace = False)
        sample_index_d = np.random.choice(down, size = sub_sample_size, replace = False)
        sample_index_u = np.random.choice(up, size = sub_sample_size, replace = False)
        sample_index = np.sort(np.concatenate((sample_index_l,sample_index_r, sample_index_d,sample_index_u)))
        copy_tofolder(sample_index, source_folder, dest_folder)
        objp = calib_inst.world_points(dist_betw_circle, board_gridrows, board_gridcolumns)
        unwrapv_lst, unwraph_lst, white_lst, avg_lst, mod_lst, gamma_lst, wrapped_phase_lst = calib_inst.projcam_calib_img_multifreq(sample_index[-1], limit, N_list, pitch_list, proj_width, proj_height, kernel_v, kernel_h, dest_folder)
        proj_img_lst = calib_inst.projector_img(unwrapv_lst, unwraph_lst, white_lst, proj_width, proj_height, pitch_list[-1], phase_st)
        #Camera calibration
        camr_error, cam_objpts, cam_imgpts, cam_mtx, cam_dist, cam_rvecs, cam_tvecs = calib_inst.camera_calib(objp,
                                                                                             white_lst,
                                                                                             bobdetect_areamin,
                                                                                             bobdetect_convexity, 
                                                                                             board_gridrows, board_gridcolumns)
        
        #Projector calibration
        proj_ret, proj_imgpts,proj_mtx, proj_dist, proj_rvecs, proj_tvecs = calib_inst.proj_calib(cam_objpts, cam_imgpts, unwrapv_lst, unwraph_lst, proj_img_lst, pitch_list[-1], phase_st, board_gridrows, board_gridcolumns) 
        #Stereo calibration
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.0001)
        stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K3+cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5+cv2.CALIB_FIX_K6
        
        st_retu,st_cam_mtx,st_cam_dist,st_proj_mtx,st_proj_dist,st_cam_proj_rmat,st_cam_proj_tvec,E,F=cv2.stereoCalibrate(cam_objpts,cam_imgpts,proj_imgpts,
                                                                                                        cam_mtx,cam_dist,proj_mtx,proj_dist,
                                                                                                        white_lst[0].shape[::-1],
                                                                                                        flags=stereocalibration_flags,
                                                                                                        criteria=criteria) 
        proj_h_mtx = np.dot(proj_mtx, np.hstack((st_cam_proj_rmat, st_cam_proj_tvec)))
         
        cam_h_mtx = np.dot(cam_mtx,np.hstack((np.identity(3), np.zeros((3,1)))))
        cam_mtx_sample.append(cam_mtx)
        cam_dist_sample.append(cam_dist)
        proj_mtx_sample.append(proj_mtx)
        proj_dist_sample.append(proj_dist)
        st_rmat_sample.append(st_cam_proj_rmat)
        st_tvec_sample.append(st_cam_proj_tvec)
        proj_h_mtx_sample.append(proj_h_mtx)
        cam_h_mtx_sample.append(cam_h_mtx)
    return np.array(cam_mtx_sample), np.array(cam_dist_sample), np.array(proj_mtx_sample), np.array(proj_dist_sample), np.array(st_rmat_sample), np.array(st_tvec_sample), np.array(proj_h_mtx_sample), np.array(cam_h_mtx_sample)

def sample_statistics(sample):
    mean = np.mean(sample, axis = 0)
    std = np.std(sample, axis = 0)
    return mean, std


def main():
    proj_width = 800; proj_height = 1280
    #type of unwrapping 
    type_unwrap =  'multifreq'
    # modulation mask limit
    sub_sample_size = 5 # sample to be taken each direction
    no_sample_sets = 100
    root_dir = r'C:\Users\kl001\Documents\pyfringe_test\white_camera_error\varying_B\bootstrap' 
    source_folder = os.path.join(root_dir, '%s_calib_images' %type_unwrap)
    dest_folder = os.path.join(source_folder, 'sub_calib')
    sigma_path =  r'C:\Users\kl001\Documents\pyfringe_test\white_camera_error\varying_B\mean_err\mean_pixel_std.npy'
    pitch_list =[1375, 275, 55, 11] 
    N_list = [3, 3, 3, 9]
    phase_st = 0 
    delta_pose = 25 # no of poses in each direction
    bobdetect_areamin = 100; bobdetect_convexity = 0.75
    dist_betw_circle = 25; #Distance between centers
    board_gridrows = 5; board_gridcolumns = 15 # calibration board parameters 
    kernel_v = 1; kernel_h=1
    savedir =  os.path.join(source_folder, 'obj_reconstruction\plane')          
    #savedir = r'C:\Users\kl001\Documents\pyfringe_test\white_camera_error\varying_B\bootstrap\obj_reconstruction\plane'
    if not os.path.exists(savedir):
        os.makedirs(savedir)  
        
    quantile_limit = 5.5
    limit = rc.B_cutoff_limit(sigma_path, quantile_limit, N_list, pitch_list)
    
    cam_mtx_sample, cam_dist_sample, proj_mtx_sample, proj_dist_sample, st_rmat_sample, st_tvec_sample, proj_h_mtx_sample, cam_h_mtx_sample = sample_intrinsics_extrinsics(delta_pose,sub_sample_size, no_sample_sets, proj_width, proj_height, limit, type_unwrap, phase_st, N_list, pitch_list, board_gridrows, board_gridcolumns, bobdetect_areamin, bobdetect_convexity, dist_betw_circle, kernel_v, kernel_h, source_folder, dest_folder )
    cam_mtx_mean, cam_mtx_std = sample_statistics(cam_mtx_sample)
    cam_dist_mean, cam_dist_std = sample_statistics(cam_dist_sample)
    proj_mtx_mean, proj_mtx_std = sample_statistics(proj_mtx_sample)
    proj_dist_mean, proj_dist_std = sample_statistics(proj_dist_sample)
    st_rmat_mean, st_rmat_std = sample_statistics(st_rmat_sample)
    st_tvec_mean, st_tvec_std = sample_statistics(st_tvec_sample)
    proj_h_mtx_mean, proj_h_mtx_std = sample_statistics(proj_h_mtx_sample)
    cam_h_mtx_mean, cam_h_mtx_std = sample_statistics(cam_h_mtx_sample)
    np.savez(os.path.join(source_folder,'sample_calibration_param.npz'),cam_mtx_sample, cam_dist_sample, proj_mtx_sample, proj_dist_sample, st_rmat_sample, st_tvec_sample, proj_h_mtx_sample, cam_h_mtx_sample) 
    np.savez(os.path.join(source_folder,'mean_calibration_param.npz'),cam_mtx_mean, cam_mtx_std, cam_dist_mean, cam_dist_std,proj_mtx_mean, proj_mtx_std ,proj_dist_mean, proj_dist_std, st_rmat_mean, st_rmat_std, st_tvec_mean, st_tvec_std) 
    np.savez(os.path.join(source_folder,'h_matrix_param.npz'),cam_h_mtx_mean, cam_h_mtx_std, proj_h_mtx_mean, proj_h_mtx_std)

if __name__ == '__main__':
    main()