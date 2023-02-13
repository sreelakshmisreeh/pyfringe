# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 16:03:45 2022

@author: kl001
"""
import cv2
import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pandas as pd
import reconstruction as rc
import nstep_fringe_cp as nstep_cp
import nstep_fringe as nstep
from tqdm import tqdm, trange
from plyfile import PlyData, PlyElement

EPSILON = -0.5
#To be tested
def image_read(data_path, N_list, scan_object):
    """
    Function to calculate each image statistics
    Parameters
    ----------
    data_path : str
                Path name for N scans
    N_list : list
    Returns
    -------
    full_images : np.ndarray.
                 Array of all images
    images_mean : np.ndarray.
                  Mean of each pattern images
    images_std : np.ndarray.
                 Standard deviation of each pattern image
    """
    path = glob.glob(os.path.join(data_path,'capt_000_*'))
    number_sections = int(len(path)/np.sum(N_list))
    images = cp.asarray([cv2.imread(file,0) for file in path])
    full_images = images.reshape(number_sections, -1, images.shape[-2], images.shape[-1])
    images_mean = cp.asnumpy(cp.mean(full_images, axis = 0)) # full_images = no. of scans, sum(pitch_list), images shapes
    images_std = cp.asnumpy(cp.std(full_images, axis = 0))
    save_path = os.path.join(data_path,'images_stat_{}.npz'.format(scan_object))
    np.savez(save_path, images_mean=images_mean, images_std=images_std)
    print("\n Pattern image statistics saved at %s "%save_path)
    return cp.asnumpy(full_images), images_mean, images_std

def random_images(images_mean, images_std):
    """
    Function to generate pattern images based mean and std
    Parameters
    ----------
    images_mean : np.ndarray.
                  Mean of each pattern images
    images_std : np.ndarray.
                 Standard deviation of each pattern image
    Returns
    -------
    random_img : np.ndarray.
                 Array of randomly generated pattern images.

    """
    mean_colm = images_mean.ravel()
    std_colm = images_std.ravel()
    df = pd.DataFrame( np.column_stack((mean_colm, std_colm)), columns = ['pixel_mean','pixel_std'])
    df['RV'] = np.random.normal(loc=df['pixel_mean'], scale=df['pixel_std'])
    rv_array = df['RV'].to_numpy()
    random_img = rv_array.reshape(images_mean.shape[0],images_mean.shape[1],images_mean.shape[2])
    return random_img

def random_ext_intinsics(calibration_mean, calibration_std):
    """
    Function to random generate intrinsics and extrinsics
    """
    
    camera_mtx = np.random.normal(loc=calibration_mean["cam_mtx_mean"], scale=calibration_std["cam_mtx_std"])
    camera_dist = np.random.normal(loc=calibration_mean["cam_dist_mean"], scale=calibration_std["cam_dist_std"])
    proj_mtx = np.random.normal(loc=calibration_mean["proj_mtx_mean"], scale=calibration_std["proj_mtx_std"])
    camera_proj_rot_mtx = np.random.normal(loc=calibration_mean["st_rmat_mean"], scale=calibration_std["st_rmat_std"])
    camera_proj_trans = np.random.normal(loc=calibration_mean["st_tvec_mean"], scale=calibration_std["st_tvec_std"])
    proj_h_mtx = np.dot(proj_mtx, np.hstack((camera_proj_rot_mtx, camera_proj_trans)))
    camera_h_mtx = np.dot(camera_mtx, np.hstack((np.identity(3), np.zeros((3, 1)))))
    return camera_mtx, camera_dist, proj_mtx, camera_proj_rot_mtx, camera_proj_trans, proj_h_mtx, camera_h_mtx

def random_reconst(proj_width, 
                   proj_height, 
                   pitch_list, 
                   N_list, 
                   limit,
                   sigma_path, 
                   type_unwrap, 
                   calib_path, 
                   obj_path, 
                   random_img,
                   random_calib_param,
                   reconst_inst):
    """
    Sub function to do reconstruction based on generated images.

    """
    cam_width = random_img.shape[-2]
    cam_height = random_img.shape[-1]
    images_arr = cp.asarray(random_img)
    reconst_inst.cam_mtx = cp.asarray(random_calib_param[0])
    reconst_inst.cam_dist = cp.asarray(random_calib_param[1])
    reconst_inst.proj_mtx = cp.asarray(random_calib_param[2])
    reconst_inst.camproj_rot_mtx = cp.asarray(random_calib_param[3])
    reconst_inst.camproj_trans_mtx = cp.asarray(random_calib_param[4])
    reconst_inst.cam_h_mtx = cp.asarray(random_calib_param[5])
    reconst_inst.proj_h_mtx = cp.asarray(random_calib_param[6])
    modulation_vector, orig_img, phase_map, mask = nstep_cp.phase_cal_cp(images_arr,
                                                                         limit,
                                                                         N_list,
                                                                         False)
    phase_map[0][phase_map[0] < EPSILON] = phase_map[0][phase_map[0] < EPSILON] + 2 * np.pi
    unwrap_vector, k_arr = nstep_cp.multifreq_unwrap_cp(pitch_list,
                                                        phase_map,
                                                        kernel_size=7,
                                                        direc='v',
                                                        mask=mask,
                                                        cam_width=cam_width,
                                                        cam_height=cam_height)
    orig_img = cp.asnumpy(orig_img[-1])
    coords, inte_rgb, cordi_sigma, mask = reconst_inst. complete_recon(unwrap_vector, 
                                                                 mask,
                                                                 orig_img, 
                                                                 modulation_vector,    
                                                                 temperature_image=None)
    mask = cp.asnumpy(mask) 
    return cp.asnumpy(coords), cp.asnumpy(inte_rgb), mask

def virtual_scan(total_virtual_scans,
                 proj_width,
                 proj_height,
                 pitch_list,
                 N_list,
                 limit,
                 sigma_path,
                 type_unwrap,
                 calib_path,
                 obj_path,
                 scan_object):
    """
    Function to generate pattern
    """
    image_stat = np.load(os.path.join(obj_path,'images_stat_{}.npz'.format(scan_object)))
    image_mean = image_stat["images_mean"]
    image_std = image_stat["images_std"]
    cam_width = image_mean.shape[-2]
    cam_height = image_mean.shape[-1]
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
                                  data_type='jpeg',
                                  processing='gpu',
                                  calib_path=calib_path,
                                  sigma_path=sigma_path,
                                  object_path=obj_path,
                                  temp=False,
                                  save_ply=False,
                                  probability=False)
    full_coords = []
    full_inte = []
    mask_list = []
    mask_list = np.full((cam_height, cam_width), True)
    for i in trange(0, total_virtual_scans):
        random_img = random_images(image_mean, image_std)
        camera_mtx, camera_dist, proj_mtx, camera_proj_rot_mtx, camera_proj_trans, proj_h_mtx, camera_h_mtx = random_ext_intinsics(calibration_mean, 
                                                                                                                                   calibration_std)
        random_calib_param = [camera_mtx, camera_dist, proj_mtx, camera_proj_rot_mtx, camera_proj_trans, camera_h_mtx, proj_h_mtx]
        coords, inte, mask = random_reconst(proj_width, 
                                            proj_height, 
                                            pitch_list, 
                                            N_list, 
                                            limit,
                                            sigma_path, 
                                            type_unwrap, 
                                            calib_path, 
                                            obj_path, 
                                            random_img=random_img,
                                            random_calib_param=random_calib_param,
                                            reconst_inst=reconst_inst)
        mask_list &=mask
        retrived_cord = np.array([nstep.recover_image(coords[:,i], mask, cam_height, cam_width) for i in range(coords.shape[-1])])
        retrived_int = np.array([nstep.recover_image(inte[:,i], mask, cam_height, cam_width) for i in range(coords.shape[-1])])
        full_coords.append(retrived_cord)
        full_inte.append(retrived_int)
        mask_list.append(mask)
    full_coords = np.array(full_coords)
    full_inte = np.array(full_inte)
    mean_cords = np.mean(full_coords, axis=0)
    std_cords = np.std(full_coords, axis=0)
    mean_intensity = np.mean(full_inte, axis=0)
    mean_cords_vector = np.array([mean_cords[i][mask] for i in range(0,mean_cords.shape[0])])
    std_cords_vector = np.array([std_cords[i][mask] for i in range(0,std_cords.shape[0])])
    mean_intensity_vector = np.array([mean_intensity[i][mask] for i in range(0,mean_intensity.shape[0])])
    
    return mean_cords, std_cords, mean_cords_vector, std_cords_vector, mask, mean_intensity_vector

def main():
    pitch_list = [1000, 110, 16]
    N_list = [3,3,9]
    proj_width = 912  
    proj_height = 1140 
    type_unwrap = 'multifreq'
    save_dir = r"C:\Users\kl001\Documents\pyfringe_test\monte_carlo"
    scan_object = "plane"
    data_path = os.path.join(save_dir,scan_object)
    sigma_path = r"C:\Users\kl001\Documents\pyfringe_test\mean_pixel_std\mean_std_pixel.npy"
    calib_path = r"C:\Users\kl001\Documents\pyfringe_test\multifreq_calib_images"
    quantile_limit = 4.5
    limit = nstep.B_cutoff_limit(sigma_path, quantile_limit, N_list, pitch_list)
    total_virtual_scans = 500
    full_images, images_mean, images_std = image_read(data_path, N_list, scan_object)
    mean_cords, std_cords, mean_cords_vector, std_cords_vector, mask, mean_inten = virtual_scan(total_virtual_scans,
                                                                                                proj_width,
                                                                                                proj_height,
                                                                                                pitch_list,
                                                                                                N_list,
                                                                                                limit,
                                                                                                sigma_path,
                                                                                                type_unwrap,
                                                                                                calib_path,
                                                                                                obj_path=data_path,
                                                                                                scan_object=scan_object)
    
    np.save(os.path.join(data_path,'monte_mean_cords_{}.npy'.format(scan_object)), mean_cords)
    np.save(os.path.join(data_path,'monte_std_cords_{}.npy'.format(scan_object)), std_cords)
    np.save(os.path.join(data_path,'monte_mean_cords_vector_{}.npy'.format(scan_object)), mean_cords_vector)
    np.save(os.path.join(data_path,'monte_std_cords_vector_{}.npy'.format(scan_object)), std_cords_vector)
    np.save(os.path.join(data_path,'monte_mean_inten_vector_{}.npy'.format(scan_object)), mean_inten)
    np.save(os.path.join(data_path,'monte_mask_{}.npy'.format(scan_object)), mask)
    xyz = list(map(tuple, mean_cords_vector.T)) 
    color = list(map(tuple, mean_inten.T))
    PlyData(
            [
                PlyElement.describe(np.array(xyz, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'points'),
                PlyElement.describe(np.array(color, dtype=[('r', 'f4'), ('g', 'f4'), ('b', 'f4')]), 'color'),
            ]).write(os.path.join(data_path, 'random_mean_obj.ply'))
    return

if __name__ == '__main__':
    main()
