# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 08:29:56 2023

@author: kl001
"""

import numpy as np
import sys
sys.path.append(r"C:\Users\kl001\pyfringe")
import image_acquisation as acq
import nstep_fringe as nstep
import reconstruction as reconstruct
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os 
import pickle
import cv2
from tqdm import tqdm

def capture_data():
    result = True
    image_index_list = np.repeat(np.arange(30, 35), 3).tolist()
    pattern_num_list = [0, 1, 2] * len(set(image_index_list))
    savedir = r"C:\Users\kl001\Documents\pyfringe_test\summer_camp"
    result &= acq.run_proj_single_camera(savedir=savedir,
                                         preview_option='Once',
                                         number_scan=1,
                                         acquisition_index=0,
                                         image_index_list=image_index_list,
                                         pattern_num_list=pattern_num_list,
                                         cam_gain=0,
                                         cam_bufferCount=15,
                                         cam_capt_timeout=10,
                                         cam_black_level=0,
                                         cam_ExposureCompensation=0,
                                         proj_exposure_period=30000,#27084,Check option 2 for recomended value
                                         proj_frame_period=40000,#34000,#33334,
                                         do_insert_black=True,
                                         led_select=4,
                                         preview_image_index=16,
                                         focus_image_index=None,
                                         image_section_size=None,
                                         pprint_status=True,
                                         save_npy=False,
                                         save_jpeg=True,
                                         clear_dir=False)
    return result
def reconst():
    pitch_list = [1200, 120, 12]
    N_list = [3, 3, 9]
    limit = float(input("\nEnter background limit:"))
    proj_width = 912  
    proj_height = 1140 
    cam_width = 1920 
    cam_height = 1200
    type_unwrap = 'multifreq'
    #obj_path = r'C:\Users\kl001\Documents\grasshopper3_python\images'
    obj_path = r'C:\Users\kl001\Documents\pyfringe_test\summer_camp'
    calib_path = r'C:\Users\kl001\Documents\pyfringe_test\multifreq_calib_images'
    model_path = r"E:\result_lut_calib\lut_models_pitch_dict.pkl"
    reconst_inst = reconstruct.Reconstruction(proj_width=proj_width,
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
                                              processing='cpu',
                                              calib_path=calib_path,
                                              object_path=obj_path,
                                              model_path=model_path,
                                              temp=False,
                                              save_ply=True,
                                              probability=False)
    obj_cordi, obj_color, cordi_sigma, mask = reconst_inst.obj_reconst_wrapper()
    return

def main():
    result = capture_data()
    if result:
        reconst()
    return
if __name__ == '__main__':
    main()