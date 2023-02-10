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
EPSILON = -0.5
TAU = 5.5
#To be tested
def image_read(data_path, N_list):
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
    images = np.array([cv2.imread(file,0) for file in path])
    full_images = images.reshape(number_sections, -1, images.shape[-2], images.shape[-1])
    images_mean = np.mean(full_images, axis = 0) # full_images = no. of scans, sum(pitch_list), images shapes
    images_std = np.std(full_images, axis = 0)
    np.savez(os.path.join(data_path,'images_stat.npz'),images_mean=images_mean, images_std=images_std)
    return full_images, images_mean, images_std

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


def random_reconst(proj_width, 
                   proj_height, 
                   pitch_list, 
                   N_list, 
                   limit,
                   sigma_path, 
                   type_unwrap, 
                   calib_path, 
                   obj_path, 
                   random_img):
    """
    Sub function to do reconstruction based on generated images.

    """
    cam_width = random_img[0].shape[2]
    cam_height = random_img[0].shape[1]
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
    #images_arr = cp.asarray(random_img)
    full_cord_list = []
    full_inte_list = []
    mask_list = np.full((cam_height, cam_width), True)
    for img in random_img:
        images_arr = img
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
        
        retrived_cord = np.array([nstep_cp.recover_image_cp(coords[:,i], mask, cam_height, cam_width) for i in range(coords.shape[-1])])
        retrived_int = np.array([nstep_cp.recover_image_cp(inte_rgb[:,i], mask, cam_height, cam_width) for i in range(coords.shape[-1])])
        full_cord_list.append(retrived_cord)
        full_inte_list.append(retrived_int)
        mask_list &=mask
    return np.array(full_cord_list), np.array(full_inte_list), mask_list 

def virtual_scan(total_scans,
                 proj_width,
                 proj_height,
                 pitch_list,
                 N_list,
                 limit,
                 sigma_path,
                 type_unwrap,
                 calib_path,
                 obj_path):
    """
    Function to generate pattern
    """
    image_stat = np.load(os.path.join(obj_path,'images_stat.npz'))
    random_img = [random_images(image_stat["images_mean"], image_stat["images_std"]) for i in range(0,total_scans)]
    full_coords, full_inte, mask = random_reconst(proj_width, 
                                                  proj_height, 
                                                  pitch_list, 
                                                  N_list, 
                                                  limit,
                                                  sigma_path, 
                                                  type_unwrap, 
                                                  calib_path, 
                                                  obj_path, 
                                                  random_img=random_img)
    mean_cords = np.mean(full_coords, axis=0)
    std_cords = np.std(full_coords, axis=0)
    mean_cords_vector = np.array([mean_cords[i][mask] for i in range(0,mean_cords.shape[0])])
    std_cords_vector = np.array([std_cords[i][mask] for i in range(0,std_cords.shape[0])])
    
    return mean_cords, std_cords, mean_cords_vector, std_cords_vector, mask

def main():
    pitch_list = [1000, 110, 16]
    N_list = [3,3,9]
    proj_width = 912  
    proj_height = 1140 
    cam_width = 1920 
    cam_height = 1200
    type_unwrap = 'multifreq'
    save_dir = r"C:\Users\kl001\Documents\pyfringe_test\monte_carlo"
    scan_object = "plane"
    data_path = os.path.join(save_dir,scan_object)
    sigma_path = r"C:\Users\kl001\Documents\pyfringe_test\mean_pixel_std\mean_std_pixel.npy"
    calib_path = r"C:\Users\kl001\Documents\pyfringe_test\multifreq_calib_images"
    quantile_limit = 4.5
    limit = nstep.B_cutoff_limit(sigma_path, quantile_limit, N_list, pitch_list)
    total_scans = 5
    full_images, images_mean, images_std = image_read(os.path.join(data_path,'images'), N_list)
    mean_cords, std_cords, mean_cords_vector, std_cords_vector, mask = virtual_scan(total_scans,
                                                                                    proj_width,
                                                                                    proj_height,
                                                                                    pitch_list,
                                                                                    N_list,
                                                                                    limit,
                                                                                    sigma_path,
                                                                                    type_unwrap,
                                                                                    calib_path,
                                                                                    obj_path=data_path)
    
    np.save(os.path.join(data_path,'monte_mean_cords.npy'), mean_cords)
    np.save(os.path.join(data_path,'monte_std_cords.npy'), std_cords)
    np.save(os.path.join(data_path,'monte_mean_cords_vector.npy'), mean_cords_vector)
    np.save(os.path.join(data_path,'monte_std_cords_vector.npy'), std_cords_vector)
    np.save(os.path.join(data_path,'monte_mask.npy'), mask)
    return

if __name__ == '__main__':
    main()