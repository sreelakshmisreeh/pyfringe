# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 09:34:29 2023

@author: kl001
"""

import cv2
import numpy as np 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pandas as pd
import sys
sys.path.append(r"C:\Users\kl001\pyfringe")
import reconstruction as rc
import nstep_fringe as nstep
from tqdm import tqdm, trange
from plyfile import PlyData, PlyElement

EPSILON = -0.5

def random_ext_intinsics(calibration_mean, calibration_std):
    """
    Function to random generate intrinsics and extrinsics.
    """
    
    camera_mtx = np.random.normal(loc=calibration_mean["cam_mtx_mean"], scale=calibration_std["cam_mtx_std"])
    camera_dist = np.random.normal(loc=calibration_mean["cam_dist_mean"], scale=calibration_std["cam_dist_std"])
    proj_mtx = np.random.normal(loc=calibration_mean["proj_mtx_mean"], scale=calibration_std["proj_mtx_std"])
    camera_proj_rot_mtx = np.random.normal(loc=calibration_mean["st_rmat_mean"], scale=calibration_std["st_rmat_std"])
    camera_proj_trans = np.random.normal(loc=calibration_mean["st_tvec_mean"], scale=calibration_std["st_tvec_std"])
    proj_h_mtx = np.dot(proj_mtx, np.hstack((camera_proj_rot_mtx, camera_proj_trans)))
    camera_h_mtx = np.dot(camera_mtx, np.hstack((np.identity(3), np.zeros((3, 1)))))
    return camera_mtx, camera_dist, proj_mtx, camera_proj_rot_mtx, camera_proj_trans, proj_h_mtx, camera_h_mtx

def subsample_mean_std(cord_list, intensity_list):
    """
    Helper function to calculate sub sample statistics.
    """
    cord_arr  = np.array(cord_list)
    pool_mean = np.mean(cord_arr, axis=0)
    pool_std = np.std(cord_arr, axis=0)
    intensity_arr = np.array(intensity_list)
    pool_inten_mean = np.mean(intensity_arr, axis=0)
    return pool_mean, pool_std, pool_inten_mean

def random_reconst_int_ext(images_arr,
                           random_calib_param,
                           reconst_inst):
    """
    Sub function to do reconstruction based on generated images.

    """
    if random_calib_param != None:
        reconst_inst.cam_mtx = random_calib_param[0]
        reconst_inst.cam_dist = random_calib_param[1]
        reconst_inst.proj_mtx = random_calib_param[2]
        reconst_inst.camproj_rot_mtx = random_calib_param[3]
        reconst_inst.camproj_trans_mtx = random_calib_param[4]
        reconst_inst.cam_h_mtx = random_calib_param[5]
        reconst_inst.proj_h_mtx = random_calib_param[6]
    modulation_vector, orig_img, phase_map, mask = nstep.phase_cal(images_arr,
                                                                   reconst_inst.limit,
                                                                   reconst_inst.N_list,
                                                                   False)
    phase_map[0][phase_map[0] < EPSILON] = phase_map[0][phase_map[0] < EPSILON] + 2 * np.pi
    reconst_inst.mask = mask
    unwrap_vector, k_arr = nstep.multifreq_unwrap(reconst_inst.pitch_list,
                                                 phase_map,
                                                 reconst_inst.kernel,
                                                 reconst_inst.fringe_direc,
                                                 reconst_inst.mask,
                                                 cam_width=reconst_inst.cam_width,
                                                 cam_height=reconst_inst.cam_height)
    orig_img = orig_img[-1]
    reconst_inst.mask = mask
    coords, inte_rgb, cordi_sigma = reconst_inst.complete_recon(unwrap_vector,
                                                                orig_img, 
                                                                None,
                                                                None)
        
    mask = reconst_inst.mask
    return coords, inte_rgb, mask

def virtual_scan_int_ext(no_drop_scans,
                         batch_size,
                         proj_width,
                         proj_height,
                         cam_width,
                         cam_height,
                         pitch_list,
                         N_list,
                         limit,
                         model_path,
                         type_unwrap,
                         calib_path,
                         obj_path,
                         scan_object,
                         dark_bias_path):
    """
    Function to generate pattern
    """
    path = glob.glob(os.path.join(obj_path,'*.tiff'))
    initial_data = no_drop_scans * sum(N_list)
    data_size =  int((len(path)/sum(N_list)) - no_drop_scans)
    path = np.reshape(path[initial_data:], (data_size,sum(N_list)))
    sample_size =int(data_size/batch_size)
    calibration_mean = np.load(os.path.join(calib_path,'{}_mean_calibration_param.npz'.format(type_unwrap)))
    calibration_std = np.load(os.path.join(calib_path,'{}_std_calibration_param.npz'.format(type_unwrap)))
    reconst_inst = rc.Reconstruction(proj_width=proj_width,
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
                                      processing='cpu',
                                      dark_bias_path=dark_bias_path,
                                      calib_path=calib_path,
                                      model_path=model_path,
                                      object_path=obj_path,
                                      temp=False,
                                      save_ply=False,
                                      probability=False)
    full_coords = []
    full_inte = []
    mask_list = np.full((cam_height, cam_width), True)
    pool_mean_lst=[];pool_std_lst=[];pool_inten_mean_lst=[]
    for i in tqdm(range(path.shape[0]), desc="virtual scan"):
        
        images = np.array([cv2.imread(file,0) for file in path[i]])
        camera_mtx, camera_dist, proj_mtx, camera_proj_rot_mtx, camera_proj_trans, proj_h_mtx, camera_h_mtx = random_ext_intinsics(calibration_mean, 
                                                                                                                                   calibration_std)
        random_calib_param = [camera_mtx, camera_dist, proj_mtx, camera_proj_rot_mtx, camera_proj_trans, camera_h_mtx, proj_h_mtx]
        
        coords, inte, mask = random_reconst_int_ext(images_arr=images,
                                                    random_calib_param=random_calib_param,
                                                    reconst_inst=reconst_inst)
        mask_list &=mask
        retrived_cord = np.array([nstep.recover_image(coords[:,i], mask, cam_height, cam_width) for i in range(coords.shape[-1])])
        retrived_int = np.array([nstep.recover_image(inte[:,i], mask, cam_height, cam_width) for i in range(coords.shape[-1])])
        full_coords.append(retrived_cord)
        full_inte.append(retrived_int)
        mask_list &= mask
        if len(full_coords) == data_size:
            print(len(full_coords))
            pool_means, pool_std, pool_inten_mean = subsample_mean_std(full_coords, full_inte)
            pool_mean_lst.append(pool_means)
            pool_std_lst.append(pool_std)
            pool_inten_mean_lst.append(pool_inten_mean)
            full_coords=[];full_inte=[];
                                                                  
    
    mean_cords = np.sum(pool_mean_lst, axis=0)/batch_size
    std_cords = np.sum(pool_std_lst, axis=0)/batch_size
    mean_intensity = np.sum(pool_inten_mean_lst, axis=0)/batch_size
    mean_cords_vector = np.array([mean_cords[i][mask_list] for i in range(0,mean_cords.shape[0])])
    std_cords_vector = np.array([std_cords[i][mask_list] for i in range(0,std_cords.shape[0])])
    mean_intensity_vector = np.array([mean_intensity[i][mask_list] for i in range(0,mean_intensity.shape[0])])
    
    return mean_cords, std_cords, mean_cords_vector, std_cords_vector, mask_list, mean_intensity_vector

def main():
    
    
    proj_width = 912  
    proj_height = 1140 
    cam_width = 1920
    cam_height = 1200
    type_unwrap = 'multifreq'
    scan_object = "concrete"
    model_path = r"C:\Users\kl001\Documents\pyfringe_test\mean_pixel_std\exp_30_fp_42_retake\lut_models.pkl"
    dark_bias_path = r"C:\Users\kl001\Documents\pyfringe_test\mean_pixel_std\exp_30_fp_42_retake\black_bias\avg_dark.npy"
    calib_path = r"C:\Users\kl001\Documents\pyfringe_test\multifreq_calib_images"
    limit = 20
    no_drop_scans = 200
    batch_size = 8
    pitch_list = [1200, 18]
    N_list = [3,3]
    data_path = r"E:\green_concrete"
    mean_cords, std_cords, mean_cords_vector, std_cords_vector, mask, mean_inten = virtual_scan_int_ext(no_drop_scans,
                                                                                                        batch_size,
                                                                                                        proj_width,
                                                                                                        proj_height,
                                                                                                        cam_width,
                                                                                                        cam_height,
                                                                                                        pitch_list,
                                                                                                        N_list,
                                                                                                        limit,
                                                                                                        model_path,
                                                                                                        type_unwrap,
                                                                                                        calib_path,
                                                                                                        data_path,
                                                                                                        scan_object,
                                                                                                        dark_bias_path)
    
    np.save(os.path.join(data_path,'monte_mean_cords_int_ext_{}.npy'.format(scan_object)), mean_cords)
    np.save(os.path.join(data_path,'monte_std_cords_int_ext_{}.npy'.format(scan_object)), std_cords)
    np.save(os.path.join(data_path,'monte_mean_cords_vector_int_ext_{}.npy'.format(scan_object)), mean_cords_vector)
    np.save(os.path.join(data_path,'monte_std_cords_vector_int_ext_{}.npy'.format(scan_object)), std_cords_vector)
    np.save(os.path.join(data_path,'monte_mean_inten_vector_int_ext_{}.npy'.format(scan_object)), mean_inten)
    np.save(os.path.join(data_path,'monte_mask_int_ext_{}.npy'.format(scan_object)), mask)
    xyz = list(map(tuple, mean_cords_vector.T)) 
    color = list(map(tuple, mean_inten.T))
    xyz_sigma = list(map(tuple, std_cords_vector.T))
    PlyData(
            [
                PlyElement.describe(np.array(xyz, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'points'),
                PlyElement.describe(np.array(color, dtype=[('r', 'f4'), ('g', 'f4'), ('b', 'f4')]), 'color'),
                PlyElement.describe(np.array(xyz_sigma, dtype=[('dx', 'f4'), ('dy', 'f4'), ('dz', 'f4')]), 'std'),
            ]).write(os.path.join(data_path, 'random_mean_obj_int_ext.ply'))
    return

if __name__ == '__main__':
    main()
