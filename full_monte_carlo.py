# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 16:03:45 2022

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

def image_read(data_path, no_drop_scans, no_batch, N_list, scan_object):
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
    
    path = sorted(glob.glob(os.path.join(data_path,'*.tiff')), key=lambda x:int(os.path.basename(x)[5:8]))
    initial_data = no_drop_scans * sum(N_list)
    length = int((len(path)-initial_data)/no_batch)
    path = np.reshape(path[initial_data:], (no_batch,length))
    images_mean = []; images_std=[]
    
    for i in range(no_batch):
        images = np.array([cv2.imread(file,0) for file in tqdm(path[i], desc="loading raw data")])
        full_images = images.reshape(-1, sum(N_list), images.shape[-2], images.shape[-1])
        images_mean.append(np.mean(full_images, axis = 0))
        images_std.append(np.std(full_images, axis = 0))
    full_img_mean =   np.sum(np.array(images_mean), axis=0)/ no_batch
    full_img_std =   np.sum(np.array(images_std), axis=0)/ no_batch
    save_path = os.path.join(data_path,'images_stat_{}.npz'.format(scan_object))
    np.savez(save_path, images_mean=full_img_mean, images_std=full_img_std)
    print("\n Pattern image statistics saved at %s "%save_path)
    return images_mean, images_std

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

def subsample_mean_std(cord_list, intensity_list, sample_size):
    cord_arr  = np.array(cord_list)
    pool_mean = np.mean(cord_arr, axis=0)
    pool_std = np.std(cord_arr, axis=0)
    intensity_arr = np.array(intensity_list)
    pool_inten_mean = np.mean(intensity_arr, axis=0)
    return pool_mean, pool_std, pool_inten_mean

def random_reconst(random_img,
                   random_calib_param,
                   reconst_inst):
    """
    Sub function to do reconstruction based on generated images.

    """
    images_arr = random_img
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

def virtual_scan(total_virtual_scans,
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
    image_stat = np.load(os.path.join(obj_path,'images_stat_{}.npz'.format(scan_object)))
    image_mean = image_stat["images_mean"]
    image_std = image_stat["images_std"]
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
    for i in tqdm(range(0, total_virtual_scans), desc="virtual scan"):
        random_img = random_images(image_mean, image_std)
        
        camera_mtx, camera_dist, proj_mtx, camera_proj_rot_mtx, camera_proj_trans, proj_h_mtx, camera_h_mtx = random_ext_intinsics(calibration_mean, 
                                                                                                                               calibration_std)
        random_calib_param = [camera_mtx, camera_dist, proj_mtx, camera_proj_rot_mtx, camera_proj_trans, camera_h_mtx, proj_h_mtx]
        
        coords, inte, mask = random_reconst(random_img=random_img,
                                            random_calib_param=random_calib_param,
                                            reconst_inst=reconst_inst)
        mask_list &=mask
        retrived_cord = np.array([nstep.recover_image(coords[:,i], mask, cam_height, cam_width) for i in range(coords.shape[-1])])
        retrived_int = np.array([nstep.recover_image(inte[:,i], mask, cam_height, cam_width) for i in range(coords.shape[-1])])
        full_coords.append(retrived_cord)
        full_inte.append(retrived_int)
        mask_list &= mask
    #full_coords = np.array(full_coords)
    #full_inte = np.array(full_inte)
    # mean_cords = np.mean(full_coords, axis=0)
    # std_cords = np.std(full_coords, axis=0)
    # mean_intensity = np.mean(full_inte, axis=0)
    batch_size = 10
    sample_size =int( total_virtual_scans/batch_size)
    pool_means, pool_std, pool_inten_mean = map(np.array,zip(*[subsample_mean_std(full_coords[i*sample_size:(i+1)*sample_size], full_inte[i*sample_size:(i+1)*sample_size],
                                                                  sample_size) for i in tqdm(range(batch_size),desc="Mean and std of coords")]))
    
    mean_cords = np.sum(pool_means, axis=0)/batch_size
    std_cords = np.sum(pool_std, axis=0)/batch_size
    mean_intensity = np.sum(pool_inten_mean, axis=0)/batch_size
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
    total_virtual_scans = 500
    no_drop_scans = 200
    no_batch = 8
    pitch_list = [1200, 18]
    N_list = [3,3]
    data_path = r"E:\green_concrete"
        
    #images_mean, images_std = image_read(data_path, no_drop_scans, no_batch , N_list, scan_object)
    mean_cords, std_cords, mean_cords_vector, std_cords_vector, mask, mean_inten = virtual_scan(total_virtual_scans,
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
                                                                                                obj_path=data_path,
                                                                                                scan_object=scan_object,
                                                                                                dark_bias_path=dark_bias_path)
    
    np.save(os.path.join(data_path,'monte_mean_cords_{}.npy'.format(scan_object)), mean_cords)
    np.save(os.path.join(data_path,'monte_std_cords_{}.npy'.format(scan_object)), std_cords)
    np.save(os.path.join(data_path,'monte_mean_cords_vector_{}.npy'.format(scan_object)), mean_cords_vector)
    np.save(os.path.join(data_path,'monte_std_cords_vector_{}.npy'.format(scan_object)), std_cords_vector)
    np.save(os.path.join(data_path,'monte_mean_inten_vector_{}.npy'.format(scan_object)), mean_inten)
    np.save(os.path.join(data_path,'monte_mask_{}.npy'.format(scan_object)), mask)
    xyz = list(map(tuple, mean_cords_vector.T)) 
    color = list(map(tuple, mean_inten.T))
    xyz_sigma = list(map(tuple, std_cords_vector.T))
    PlyData(
            [
                PlyElement.describe(np.array(xyz, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'points'),
                PlyElement.describe(np.array(color, dtype=[('r', 'f4'), ('g', 'f4'), ('b', 'f4')]), 'color'),
                PlyElement.describe(np.array(xyz_sigma, dtype=[('dx', 'f4'), ('dy', 'f4'), ('dz', 'f4')]), 'std'),
            ]).write(os.path.join(data_path, 'random_mean_obj.ply'))
    return

if __name__ == '__main__':
    main()

