# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 10:47:29 2023

@author: kl001
"""

import numpy as np
import cupy as cp
import sys
sys.path.append(r'C:\Users\kl001\pyfringe')
import image_acquisation as acq
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import glob
import cv2
import pickle
from tqdm import tqdm
import nstep_fringe_cp as nstep_cp


os.environ["KMP_DUPLICATE_LIB_OK"] = "True" # needed when openCV and matplotlib used at the same time
EPSILON = -0.5

def capture(image_index_list, 
            pattern_num_list, 
            savedir,
            number_scan,
            proj_exposure_period,
            proj_frame_period):
    """
    #This is for running modulation experiment.
    #Ground truth: 
    #gt_pitch_list =[1000, 110, 16] 
    # gt_N_list = [16, 16, 16]
    """

    result = acq.run_proj_single_camera(savedir=savedir,
                                         preview_option='Once',
                                         number_scan=number_scan,
                                         acquisition_index=0,
                                         image_index_list=image_index_list,
                                         pattern_num_list=pattern_num_list,
                                         cam_gain=0,
                                         cam_bufferCount=15,
                                         cam_capt_timeout=10,
                                         cam_black_level=0,
                                         cam_ExposureCompensation=0,
                                         proj_exposure_period=proj_exposure_period,#Check image aquisation option2 for recomended value
                                         proj_frame_period=proj_frame_period,#
                                         do_insert_black=True,
                                         led_select=4,
                                         preview_image_index=31,
                                         focus_image_index=None,
                                         image_section_size=None,
                                         pprint_status=True,
                                         save_npy=False,
                                         save_jpeg=True)
    return result

def gt_read_data(data_path, N_list):
    path = sorted(glob.glob(os.path.join(data_path,'capt_000_*')), key=lambda x:int(os.path.basename(x)[-11:-5]))
    number_sections = int(len(path)/np.sum(N_list))
    images = np.array([cv2.imread(file,0) for file in path])
    full_images = images.reshape(number_sections, -1, images.shape[-2], images.shape[-1])
    return full_images
    
def ground_truth_calc(iteration, 
                      pitch_list, 
                      N_list, 
                      limit,
                      data_path, 
                      cam_width, 
                      cam_height):
    full_images = gt_read_data(data_path, N_list)
    unwrap_img_arr = []
    mod_img_arr = []
    orig_img_arr = []
    k_map_arr = []
    for i in tqdm(range(0,iteration), desc = 'ground truth iterations'):
       modulation, orig_img, phase_map, mask = nstep_cp.phase_cal_cp(full_images[i], 
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
       mod_img = nstep_cp.recover_image_cp(modulation[-1], mask, cam_height, cam_width)
       unwrap_img = nstep_cp.recover_image_cp(unwrap_vector, mask, cam_height, cam_width)
       k_map = nstep_cp.recover_image_cp(k_arr, mask, cam_height, cam_width)
          
       mod_img_arr.append(cp.asnumpy(mod_img))
       unwrap_img_arr.append(cp.asnumpy(unwrap_img))
       orig_img_arr.append(cp.asnumpy(orig_img[-1]))
       k_map_arr.append(cp.asnumpy(k_map))
    mean_modu_img = np.mean(mod_img_arr, axis=0)
    mean_unwrap_img = np.mean(unwrap_img_arr, axis=0)
    var_unwrap_img = np.var(unwrap_img_arr, axis=0)
    mean_orig_img = np.mean(orig_img_arr, axis=0)
    mean_k_map = np.mean(k_map_arr, axis=0)
    np.save(os.path.join(data_path,'gt_modulation.npy'), mean_modu_img)
    np.save(os.path.join(data_path,'gt_unwrap.npy'), mean_unwrap_img)
    np.save(os.path.join(data_path,'gt_orig_img.npy'),mean_orig_img)
    np.save(os.path.join(data_path,'gt_var.npy'),var_unwrap_img)
    return mean_modu_img, mean_unwrap_img, mean_orig_img, var_unwrap_img, mean_k_map

def phase_loop(iterations,
               images_ref,
               images_high,
               gt_limit, 
               new_N_list,
               gt_scaled_ref, 
               new_freq_list, 
               new_pitch_list, 
               cam_height, 
               cam_width):
    
    ref_phasemap = []
    high_phasemap = []
    mod_array = []
    gt_scaled_ref = cp.asarray(gt_scaled_ref)
    for i in range(iterations):
        img_stack = cp.vstack((images_ref[:,i], images_high[:,i]))
        modulation_vector, orig_img, phase_map, mask = nstep_cp.phase_cal_cp(img_stack,
                                                                             gt_limit,
                                                                             new_N_list,
                                                                             False)
        phase_map[0][phase_map[0] < EPSILON] = phase_map[0][phase_map[0] < EPSILON] + 2 * np.pi
        
        updated_phase_arr = cp.array([gt_scaled_ref[mask]]+[phase_map[-1]])
        unwrap_vector, k_arr = nstep_cp.multifreq_unwrap_cp(new_pitch_list,
                                                            updated_phase_arr,
                                                            kernel_size=9,
                                                            direc='v',
                                                            mask=mask,
                                                            cam_width=cam_width,
                                                            cam_height=cam_height)
        
        ref_phasemap.append(cp.asnumpy(nstep_cp.recover_image_cp(phase_map[0], mask, cam_height, cam_width)))
        high_phasemap.append(cp.asnumpy(nstep_cp.recover_image_cp(unwrap_vector, mask, cam_height, cam_width)))
        mod_array.append(cp.asnumpy(nstep_cp.recover_image_cp(modulation_vector[-1], mask, cam_height, cam_width)))
        
    return ref_phasemap, high_phasemap, mod_array

def full_phasemap(mod_freq_list, 
                  iterations, 
                  gt_limit, 
                  gt_scaled_ref, 
                  cam_height, 
                  cam_width,
                  mod_pitch_list,
                  mod_savedir):
    all_path = sorted(glob.glob(os.path.join(mod_savedir,'*.jpeg')),key=lambda x:int(os.path.basename(x)[-11:-5])) 
    path_ref = all_path[::len(mod_pitch_list)*3] + all_path[1::len(mod_pitch_list)*3] + all_path[2::len(mod_pitch_list)*3]
    images_ref = cp.array([cv2.imread(file,0) for file in path_ref])
    images_ref = images_ref.reshape(3,iterations,1200,1920)
    
    for freq_no in tqdm(range (1, len(mod_freq_list)), desc="phase maps"):
        path_high = all_path[(3*freq_no)::len(mod_pitch_list)*3] + all_path[(3*freq_no+1)::len(mod_pitch_list)*3] + all_path[(3*freq_no+2)::len(mod_pitch_list)*3]
        print("\nReading images")
        images_high = cp.asarray([cv2.imread(file,0) for file in path_high])
        images_high = images_high.reshape(3,iterations,1200,1920)
        print("Reading complete")
        new_N_list = [3,3]
        new_freq_list  = np.append(mod_freq_list[0],mod_freq_list[freq_no])
        new_pitch_list = np.append(mod_pitch_list[0], mod_pitch_list[freq_no])
       
        ref_phasemap, high_phasemap, mod_array = phase_loop(iterations, 
                                                            images_ref,
                                                            images_high,
                                                            gt_limit,
                                                            new_N_list,
                                                            gt_scaled_ref,
                                                            new_freq_list,
                                                            new_pitch_list, 
                                                            cam_height, 
                                                            cam_width)
        print("Saving data") 
        if freq_no == 1:
            np.save(os.path.join(mod_savedir,'ref_unwrap.npy'), ref_phasemap)
        np.save(os.path.join(mod_savedir,'high_unwrap%d.npy'%(mod_freq_list[freq_no])),high_phasemap)
        np.save(os.path.join(mod_savedir,'mod_array%d.npy'%(mod_freq_list[freq_no])), mod_array)
    return

def ref_var_calc(mod_savedir,camx, deltax, camy, deltay):
    print("Calculating reference variance")
    ref_phasemap =  np.load(os.path.join(mod_savedir,'ref_unwrap.npy'))
    ref_phasemap = ref_phasemap[:,camy:camy+deltay,camx:camx+deltax]
    ref_var = np.var(ref_phasemap, axis=0)
    np.save(os.path.join(mod_savedir,'ref_var.npy'), ref_var)
    return ref_var

def high_var_calc(mod_savedir, index_list, mod_freq_list, camx, deltax, camy, deltay):
    
    high_phasemap = np.array([np.load(os.path.join(mod_savedir,'high_unwrap%d.npy'%(mod_freq_list[freq_no]))) 
                     for freq_no in tqdm(range(index_list[0],index_list[1]),"high phasemap")])
    high_phasemap = high_phasemap[:,:,camy:camy+deltay, camx:camx+deltax]
    sub_high_var = np.var(high_phasemap, axis=1)
    
    return sub_high_var
        
def total_high_var(mod_savedir, mod_freq_list, camx, deltax, camy, deltay):
    print("Variance set 1")
    high_var1 = high_var_calc(mod_savedir, [1,int(len(mod_freq_list)/2)], mod_freq_list, camx, deltax, camy, deltay)
    print("Variance set 2")
    high_var2 = high_var_calc(mod_savedir, [int(len(mod_freq_list)/2), len(mod_freq_list)], mod_freq_list, camx, deltax, camy, deltay)
    total_var = np.vstack((high_var1, high_var2))
    np.save(os.path.join(mod_savedir,'high_var.npy'), total_var)
    return total_var

def subsample_mean_var(ik_avg_100, ik_std_100, sample_size):
    new_img = np.array([np.random.normal(loc=ik_avg_100, scale=ik_std_100) for i in range(sample_size)])
    pool_mean = np.mean(new_img, axis=0)
    pool_var = np.var(new_img, axis=0)
    return pool_mean, pool_var

def var_ik_avg_ik(images, mod_savedir, virtual_scans, no_subsamples):
    """
    images.shape = (iterations, len(mod_freq_lst),3, 1200,1920)
    """
    print("ik initial std calculations")
    ik_std_100 =  np.std(images, axis=0)
    print("ik initial average calculations")
    ik_avg_100 =  np.mean(images, axis=0)
    sample_size = int(virtual_scans/no_subsamples)
    pool_means, pool_vars = map(np.array,zip(*[subsample_mean_var(ik_avg_100, ik_std_100, sample_size) for i in tqdm(range(no_subsamples),desc="virtual images")]))
    ik_avg = np.sum(pool_means, axis=0)/no_subsamples
    ik_var = np.sum(pool_vars, axis=0)/ no_subsamples
    np.save(os.path.join(mod_savedir,'ik_var.npy'), ik_var)
    np.save(os.path.join(mod_savedir,'ik_avg.npy'), ik_avg)
    return ik_var, ik_avg


def var_phase(single_ik_var, image_array, N):
    """
    image_array.shape =(len(mode_freq_lst,3,1200.1920))
    """
    sin_lst_k =  np.sin(2*np.pi*(np.tile(np.arange(1,N+1),N)-np.repeat(np.arange(1,N+1),N))/N)
    sin_lst =  np.sin(2*np.pi*np.arange(1,N+1)/N)
    cos_lst =  np.cos(2*np.pi*np.arange(1,N+1)/N)
    denominator_cs = (np.einsum("i,jikl->jkl",sin_lst, image_array))**2 + (np.einsum("i,jikl->jkl",cos_lst, image_array))**2
    each_int = [np.einsum("j,ijkl->ikl",sin_lst_k[i*N: (i+1)*N], image_array)/(denominator_cs)for i in range(N)]
    each_int_reshape = each_int.reshape(each_int.shape[0],each_int.shape[-2]*each_int.shape[-1])
    sigmasq_phi = (np.sum([np.einsum("j,ijkl->ikl",sin_lst_k[i*N:(i+1)*N], image_array)**2 *single_ik_var[:,i]/(denominator_cs**2) for i in range(N)],axis=0))
    return sigmasq_phi, np.array(each_int)

def mean_sigmas(images, mod_savedir, N, mod_freq_list, virtual_scans, no_subsamples):
    print("Mean sigmas")
    ik_var, ik_avg = var_ik_avg_ik(images, mod_savedir, virtual_scans, no_subsamples)
    mean_sigmasq_phi, mean_each_int = var_phase(ik_var, ik_avg, N)
    mean_sigmasq_norm_phi = np.einsum("i,ijk->ijk",(1/(mod_freq_list**2)),mean_sigmasq_phi)
    temp = np.einsum("i,jk->ijk",(mod_freq_list**2) ,mean_sigmasq_phi[0]) #[0] reference freq
    mean_delta_phi = ( temp[1:]+  mean_sigmasq_phi[1:])
    np.save(os.path.join(mod_savedir,'mean_sigma_sq_phi.npy'),mean_sigmasq_phi)
    np.save(os.path.join(mod_savedir,'mean_sigmasq_norm_phi.npy'),mean_sigmasq_norm_phi)
    np.save(os.path.join(mod_savedir,'mean_delta_phi.npy'),mean_delta_phi)
    return 

def singlescan_sigmasqphi(lut_model, images, N, iterations, mod_freq_list, mod_savedir):
    single_sigmasq_phi_lst = [];  single_each_int_lst = []; ik_var_lst=[]
    images_shape = images.shape
    for i in tqdm(range(iterations),desc="single_sigmasqphi"):
        single_ik_std =np.array([lut_model[k].predict(images[i,k].ravel()).reshape(images_shape[-3],images_shape[-2],images_shape[-1]) 
                                 for k in range(len(mod_freq_list))])
        single_ik_var = single_ik_std**2
        ik_var_lst.append(single_ik_var)
        single_sigmasq_phi, single_each_int = var_phase(single_ik_var, images[i], N)
        single_sigmasq_phi_lst.append(single_sigmasq_phi)
        single_each_int_lst.append(single_each_int)
    single_sigmasq_phi_lst = np.array(single_sigmasq_phi_lst)
    np.save(os.path.join(mod_savedir,'single_sigmasq_phi_lst.npy'),single_sigmasq_phi_lst)
    #np.save(os.path.join(mod_savedir,'single_ik_var_lst.npy'),np.array(ik_var_lst))
    return 

def single_norm_delta(mod_savedir,mod_freq_list ):
    single_sigmasq_phi_lst = np.load(os.path.join(mod_savedir,"single_sigmasq_phi_lst.npy"))
    single_norm_sigmasq = np.einsum("i,jikl->jikl",(1/(mod_freq_list)**2), single_sigmasq_phi_lst)
    temp = np.einsum("i,jkl->jikl",mod_freq_list[1:]**2, single_sigmasq_phi_lst[:,0])
    single_sigma_sq_delta = temp + single_sigmasq_phi_lst[:,1:]
    np.save(os.path.join(mod_savedir,'single_norm_sigmasq.npy'),single_norm_sigmasq)
    np.save(os.path.join(mod_savedir,'single_sigma_sq_delta.npy'),single_sigma_sq_delta)
    return


def main():
    cam_width = 1920
    cam_height = 1200
    proj_width = 912  
    proj_height = 1140
    proj_exposure_period =33000 #29000
    proj_frame_period = 40000#36000
    gt_pitch_list = np.array([1200, 120, 12]) 
    #gt_pitch_list = np.array([1000, 110, 16]) 
    gt_freq = proj_width/gt_pitch_list
    gt_N_list = [16, 16, 16]
    gt_savedir =  r"E:\white board5\ground_truth"
    mod_savedir = r"E:\white board5\freq_variation"
    mod_pitch_list = np.array([1200, 912, 450, 275, 225, 150, 120, 110, 90, 75,65, 55,50, 45, 40, 35,30, 25, 20, 19,18])
    mod_freq_list = proj_width/mod_pitch_list
    #mod_freq_list = np.append(proj_width/gt_pitch_list[0],np.arange(1,59,3))
    #mod_pitch_list = np.ceil(proj_width / mod_freq_list)
    
    N_ini = 3
    mod_N_list = np.array([N_ini]*len(mod_pitch_list))
    
    model_path = r"E:\result_lut_calib\lut_models_pitch_dict.pkl"
    with open(model_path, "rb") as tt:
        model_dict = pickle.load(tt)
   
    option = input("Please choose: \n1:Ground truth capture\n2:Varying freq capture\n3: Phase map calculations\n4: Variance calculations")
    if option == '1':
        gt_iterations = int(input("No. of scans"))
        gt_image_index_list = np.repeat(np.arange(32,48),3).tolist() #change
        #gt_image_index_list = np.repeat(np.arange(0,16),3).tolist()
        gt_pattern_num_list = [0,1,2] * len(set(gt_image_index_list))
        result = capture(image_index_list=gt_image_index_list, 
                          pattern_num_list=gt_pattern_num_list, 
                          savedir=gt_savedir,
                          number_scan=gt_iterations,
                          proj_exposure_period=proj_exposure_period,
                          proj_frame_period=proj_frame_period)
    if option == '2':
        number_scan = int(input("No. of scans"))
        mod_image_index_list = np.repeat(np.append(1,np.arange(4,24)),3).tolist()
        #mod_image_index_list = np.repeat(np.arange(16,37),3).tolist()
        mod_pattern_num_list = [0,1,2] * len(set(mod_image_index_list))
        result = capture(image_index_list=mod_image_index_list, 
                          pattern_num_list=mod_pattern_num_list, 
                          savedir=mod_savedir,
                          number_scan=number_scan,
                          proj_exposure_period=proj_exposure_period,
                          proj_frame_period=proj_frame_period)

    if option == '3':
        iterations =  int(input("No. of scans"))
        gt_limit = float(input("Background limit"))
       
        gt_modulation, gt_unwrap, gt_orig_img, gt_var, gt_k_map = ground_truth_calc(iteration=iterations, 
                                                                                    pitch_list=gt_pitch_list, 
                                                                                    N_list=gt_N_list, 
                                                                                    limit=gt_limit,
                                                                                    data_path=gt_savedir, 
                                                                                    cam_width=cam_width, 
                                                                                    cam_height=cam_height)
        gt_scaled_ref = gt_unwrap *gt_pitch_list[-1]/mod_pitch_list[0]
        full_phasemap(mod_freq_list, 
                      iterations, 
                      gt_limit, 
                      gt_scaled_ref, 
                      cam_height, 
                      cam_width,
                      mod_pitch_list,
                      mod_savedir)
    if option == '4':
        iterations =  int(input("No. of scans"))
        virtual_scans = int(input("No. of virtual scans"))
        no_subsamples = int(input("No. of sub virtual scan"))
        if virtual_scans%no_subsamples != 0:
            print("Total virtual scans cannot be divided into given sub virtual scans ")
        camx = int(input("camx"))
        camy = int(input("camy"))
        deltax = int(input("deltax"))
        deltay = int(input("deltay"))
        all_path = sorted(glob.glob(os.path.join(mod_savedir,'*.jpeg')),key=lambda x:int(os.path.basename(x)[-11:-5])) 
        images = np.array([cv2.imread(file,0) for file in all_path])
        images = images.reshape(iterations, len(mod_freq_list), N_ini, 1200, 1920)
        imag_region = images[:,:,:,camy : camy + deltay, camx : camx + deltax]
        obs_ref_var = ref_var_calc(mod_savedir, camx, deltax, camy, deltay)
        obs_high_var = total_high_var(mod_savedir, mod_freq_list,camx, deltax, camy, deltay)
        obs_temp = np.einsum("i,jk->ijk",mod_freq_list[1:]**2 , obs_ref_var)
        obs_sigma_delta = obs_temp + obs_high_var
        np.save(os.path.join(mod_savedir,"obs_sigma_delta.npy"),obs_sigma_delta)
        mean_sigmas(imag_region, mod_savedir, N_ini, mod_freq_list, virtual_scans, no_subsamples)
        lut_model = [model_dict[p] for p in mod_pitch_list]
        singlescan_sigmasqphi(lut_model, imag_region, N_ini, iterations, mod_freq_list, mod_savedir)
        single_norm_delta(mod_savedir,mod_freq_list )
        
    return

if __name__ == '__main__':
    main()
