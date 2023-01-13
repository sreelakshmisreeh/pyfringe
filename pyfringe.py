# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 10:47:29 2023

@author: kl001
"""

import numpy as np
import image_acquisation as acq
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True" # needed when openCV and matplotlib used at the same time
#%%

#Gamma curve
savedir = r'C:\Users\kl001\Documents\pyfringe_test\gamma_images'
gamma_image_index_list = np.repeat(np.arange(5,22),3).tolist()
gamma_pattern_num_list = [0,1,2] * len(set(gamma_image_index_list))
result = acq.gamm_curve(gamma_image_index_list,gamma_pattern_num_list, savedir)
#%%
#Calibration
savedir = r'C:\Users\kl001\Documents\pyfringe_test\multifreq_calib_images'
image_index_list =  np.repeat(np.arange(22,34),3).tolist()
pattern_num_list = [0,1,2] * len(set(image_index_list))
number_scan = 1
result = acq.calib_capture(image_index_list,pattern_num_list,savedir,number_scan)