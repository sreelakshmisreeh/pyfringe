# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 10:47:29 2023

@author: kl001
"""

import numpy as np
import image_acquisation as acq
import matplotlib.pyplot as plt
import seaborn as sns
import os
import image_acquisation as acq
os.environ["KMP_DUPLICATE_LIB_OK"] = "True" # needed when openCV and matplotlib used at the same time

#This is for checking running modulation experiment. This is new projector firmware with ground truth.
#Ground truth: 
#gt_pitch_list =[1000, 110, 16] 
# gt_N_list = [16, 16, 16]

savedir = r'C:\Users\kl001\Documents\pyfringe_test\modulation\ground_truth'
image_index_list = np.repeat(np.arange(0,16),3).tolist()
pattern_num_list = [0,1,2] * len(set(image_index_list))
savedir = r'C:\Users\kl001\Documents\grasshopper3_python\images'
result = acq.run_proj_single_camera(savedir=savedir,
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
                                     proj_exposure_period=27500,#Check image aquisation option2 for recomended value
                                     proj_frame_period=34000,#
                                     do_insert_black=True,
                                     led_select=4,
                                     preview_image_index=21,
                                     focus_image_index=34,
                                     image_section_size=None,
                                     pprint_status=True,
                                     save_npy=False,
                                     save_jpeg=True)

#%%
#This is for varying frequencies from
#pitch_list = [1000,912,228,131,92,71,57,48,42,37,33,30,27,25,23,22,20,19,18,17,16]
proj_width = 912
proj_height = 1140
savedir = r'C:\Users\kl001\Documents\pyfringe_test\modulation\allfreq'
mod_image_index_list = np.repeat(np.arange(16,37),3).tolist()
mod_pattern_num_list = [0,1,2] * len(set(image_index_list))
savedir = r'C:\Users\kl001\Documents\grasshopper3_python\images'
result = acq.run_proj_single_camera(savedir=savedir,
                                     preview_option='Once',
                                     number_scan=1,
                                     acquisition_index=0,
                                     image_index_list=mod_image_index_list,
                                     pattern_num_list=mod_pattern_num_list,
                                     cam_gain=0,
                                     cam_bufferCount=15,
                                     cam_capt_timeout=10,
                                     cam_black_level=0,
                                     cam_ExposureCompensation=0,
                                     proj_exposure_period=27500,#Check image aquisation option2 for recomended value it will be diffrent since there are more images
                                     proj_frame_period=34000,#
                                     do_insert_black=True,
                                     led_select=4,
                                     preview_image_index=21,
                                     focus_image_index=34,
                                     image_section_size=None,
                                     pprint_status=True,
                                     save_npy=False,
                                     save_jpeg=True)