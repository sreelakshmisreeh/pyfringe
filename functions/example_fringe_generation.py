#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 11:13:53 2022

@author: Sreelakshmi
"""

import sys
sys.path.append('/Users/Sreelakshmi/Documents/Github/pyfringe/functions')
import nstep_fringe as nstep
import cv2
import numpy as np
import matplotlib.pyplot as plt



width = 800; height = 1280
inte_rang = [50,240]
# Set type of unwrapping
type_unwrap = 'phase'
EPSILON = 0.5

#multifrequency unwrapping parameters
if type_unwrap == 'multifreq':
    pitch_list =[1375, 275, 55, 11] 
    N_list = [3, 3, 3, 9]
    kernel_v = 1; kernel_h=1
    phase_st = 0
    path = '/Users/Sreelakshmi/Documents/Raspberry/reconstruction/testing_data/multifreq_calib_images'
    
# multiwavelength unwrapping parameters
if type_unwrap == 'multiwave':
    pitch_list = [139,21,18]
    N_list =[5,5,9]
    kernel_v = 40; kernel_h=30  
    phase_st = 0
    path = '/Users/Sreelakshmi/Documents/Raspberry/reconstruction/testing_data/multiwave_calib_images'

# phase coding unwrapping parameters
if type_unwrap == 'phase':
    pitch_list =[20]
    N_list =[9]
    kernel_v = 30; kernel_h=35
    phase_st = -np.pi
    path = '/Users/Sreelakshmi/Documents/Raspberry/reconstruction/testing_data/phase_calib_images'

# To generate patterns in both directions based on type of unwrapping.
fringe_arr, delta_deck_list = nstep.calib_generate(width, height, type_unwrap, N_list, pitch_list, phase_st, inte_rang, path)
#%%
# To veiw generated fringe patterns. Can also be plotted using matplotlib
cv2.startWindowThread()
cv2.namedWindow('proj',cv2.WINDOW_NORMAL)
for i,img in enumerate(fringe_arr):
       cv2.imshow('proj',img)
       cv2.waitKey()
cv2.destroyAllWindows()
cv2.waitKey(1)
#%%
## To generate patterns in one directions based on type of unwrapping for reconstruction.
direc = 'h'
fringe_arr_d, delta_deck_list_d = nstep.recon_generate(width, height, type_unwrap, N_list, pitch_list, phase_st, inte_rang, direc, path)

# To veiw generated fringe patterns.
for i,img in enumerate(fringe_arr_d):
    plt.figure()
    plt.imshow(img,cmap = 'gray')

#%% Tesing generated patterns
fringe_arr =  fringe_arr.astype(np.float64)
if type_unwrap == 'multifreq':
    
    multi_delta_deck_1 = nstep.delta_deck_gen(N_list[0], fringe_arr.shape[1], fringe_arr.shape[2])
    multi_delta_deck_2 = nstep.delta_deck_gen(N_list[1], fringe_arr.shape[1], fringe_arr.shape[2])
    multi_delta_deck_3 = nstep.delta_deck_gen(N_list[2], fringe_arr.shape[1], fringe_arr.shape[2])
    multi_delta_deck_4 = nstep.delta_deck_gen(N_list[3], fringe_arr.shape[1], fringe_arr.shape[2])
    
    multi_phase_v1 = nstep.phase_cal(fringe_arr[0 : N_list[0]], N_list[0], multi_delta_deck_1 )
    multi_phase_h1 = nstep.phase_cal(fringe_arr[N_list[0] : 2 * N_list[0]], N_list[0], multi_delta_deck_1 )
    multi_phase_v2 = nstep.phase_cal(fringe_arr[2 * N_list[0] : 2 * N_list[0] + N_list[1]], N_list[1], multi_delta_deck_2 )
    multi_phase_h2 = nstep.phase_cal(fringe_arr[2 * N_list[0] + N_list[1] : 2 * N_list[0] + 2 * N_list[1]], N_list[1], multi_delta_deck_2 )
    multi_phase_v3 = nstep.phase_cal(fringe_arr[2 * N_list[0] + 2 * N_list[1] : 2 * N_list[0] + 2 * N_list[1] + N_list[2]], N_list[2], multi_delta_deck_3 )
    multi_phase_h3 = nstep.phase_cal(fringe_arr[2 * N_list[0] + 2 * N_list[1] + N_list[2] : 2 * N_list[0] + 2 * N_list[1] + 2 * N_list[2]], N_list[2], multi_delta_deck_3 )
    multi_phase_v4 = nstep.phase_cal(fringe_arr[2 * N_list[0] + 2 * N_list[1] + 2 * N_list[2] : 2 * N_list[0] + 2 * N_list[1] + 2 * N_list[2] + N_list[3]], N_list[3], multi_delta_deck_4 )
    multi_phase_h4 = nstep.phase_cal(fringe_arr[2 * N_list[0] + 2 * N_list[1] + 2 * N_list[2] + N_list[3] : 2 * N_list[0] + 2 * N_list[1] + 2 * N_list[2] + 2 * N_list[3]], N_list[3], multi_delta_deck_4 )
    
    multi_phase_v1[multi_phase_v1< EPSILON] = multi_phase_v1[multi_phase_v1 < EPSILON ] + 2 * np.pi
    multi_phase_h1[multi_phase_h1< EPSILON] = multi_phase_h1[multi_phase_h1 < EPSILON ] + 2 * np.pi
    
    phase_arr_v = np.stack([multi_phase_v1, multi_phase_v2, multi_phase_v3, multi_phase_v4])
    phase_arr_h = np.stack([multi_phase_h1, multi_phase_h2, multi_phase_h3, multi_phase_h4])
    
    multifreq_unwrap_v, k_arr_v = nstep.multifreq_unwrap(pitch_list, phase_arr_v)
    multifreq_unwrap_h, k_arr_h = nstep.multifreq_unwrap(pitch_list, phase_arr_h)

if type_unwrap == 'multiwave':
    eq_wav12 = (pitch_list[-1] * pitch_list[1]) / (pitch_list[1]-pitch_list[-1])
    eq_wav123 = pitch_list[0] *eq_wav12 / (pitch_list[0] - eq_wav12)
    pitch_list = np.insert(pitch_list,0,eq_wav123)
    pitch_list = np.insert(pitch_list,2,eq_wav12)
   
    multi_delta_deck_3 = nstep.delta_deck_gen(N_list[0], fringe_arr.shape[1], fringe_arr.shape[2])
    multi_delta_deck_2 = nstep.delta_deck_gen(N_list[1], fringe_arr.shape[1], fringe_arr.shape[2])
    multi_delta_deck_1 = nstep.delta_deck_gen(N_list[2], fringe_arr.shape[1], fringe_arr.shape[2])
    
    multi_phase_v3 = nstep.phase_cal(fringe_arr[0 : N_list[0]], N_list[0], multi_delta_deck_3 )
    multi_phase_h3 = nstep.phase_cal(fringe_arr[N_list[0] : 2 * N_list[0]], N_list[0], multi_delta_deck_3 )
    multi_phase_v2 = nstep.phase_cal(fringe_arr[2 * N_list[0] : 2 * N_list[0] + N_list[1]], N_list[1], multi_delta_deck_2 )
    multi_phase_h2 = nstep.phase_cal(fringe_arr[2 * N_list[0] + N_list[1] : 2 * N_list[0] + 2 * N_list[1]], N_list[1], multi_delta_deck_2 )
    multi_phase_v1 = nstep.phase_cal(fringe_arr[2 * N_list[0] + 2 * N_list[1] : 2 * N_list[0] + 2 * N_list[1] + N_list[2]], N_list[2], multi_delta_deck_1 )
    multi_phase_h1 = nstep.phase_cal(fringe_arr[2 * N_list[0] + 2 * N_list[1] + N_list[2] : 2 * N_list[0] + 2 * N_list[1] + 2 * N_list[2]], N_list[2], multi_delta_deck_1 )
    
    multi_phase_v12 = np.mod(multi_phase_v1 - multi_phase_v2, 2 * np.pi)
    multi_phase_h12 = np.mod(multi_phase_h1 - multi_phase_h2, 2 * np.pi)
    multi_phase_v123 = np.mod(multi_phase_v12 - multi_phase_v3, 2 * np.pi)
    multi_phase_h123 = np.mod(multi_phase_h12 - multi_phase_h3, 2 * np.pi)
    
    multi_phase_v123 = nstep.edge_rectification(multi_phase_v123, 'v')
    multi_phase_h123 = nstep.edge_rectification(multi_phase_h123, 'h')
    
    phase_arr_v = np.stack([multi_phase_v123, multi_phase_v3, multi_phase_v12,multi_phase_v2, multi_phase_v1])
    phase_arr_h = np.stack([multi_phase_h123, multi_phase_h3, multi_phase_h12,multi_phase_h2, multi_phase_h1])
    
    unwrap_v, k_arr_v = nstep.multiwave_unwrap(pitch_list, phase_arr_v, kernel_v, 'v')
    unwrap_h, k_arr_h = nstep.multiwave_unwrap(pitch_list, phase_arr_h, kernel_h, 'h')
    
if type_unwrap == 'phase':
      
    capt_delta_deck1 = nstep.delta_deck_gen(N_list[0], fringe_arr.shape[1], fringe_arr.shape[2])
    unwrap_v, unwrap_h, k0_v, k0_h, cos_wrap_v, cos_wrap_h, step_wrap_v, step_wrap_h = nstep.ph_temp_unwrap(fringe_arr[0 : N_list[0]], fringe_arr[N_list[0] : 2 * N_list[0]], fringe_arr[2 * N_list[0] : 3 * N_list[0]], fringe_arr[3 * N_list[0] : 4 * N_list[0]], pitch_list[0], height, width, capt_delta_deck1, kernel_v, kernel_h)
    plt.figure()
    plt.plot(cos_wrap_v[500])
    plt.plot(step_wrap_v[500])
    plt.xlabel('Dimension',fontsize = 15)
    plt.ylabel('Phase',fontsize = 15)
    plt.title('Wrapped phases for phase coded (vertical fringes)', fontsize = 20)
    plt.figure()
    plt.plot(cos_wrap_h[:,500])
    plt.plot(step_wrap_h[:,500])
    plt.title('Wrapped phases for phase coded (horizontal fringes)', fontsize = 20)
    plt.xlabel('Dimension',fontsize = 15)
    plt.ylabel('Phase',fontsize = 15)
    
plt.figure()    
plt.imshow(unwrap_v)
plt.title('Unwrapped phase map (vertical fringes)', fontsize = 20)
plt.xlabel('Dimension',fontsize = 15)
plt.ylabel('Phase',fontsize = 15)
plt.figure()
plt.imshow(unwrap_h)
plt.title('Unwrapped phase map (horizontal fringes)', fontsize = 20)
plt.xlabel('Dimension',fontsize = 15)
plt.ylabel('Phase',fontsize = 15)

plt.figure()
plt.plot(unwrap_v[500])
plt.xlabel('Dimension',fontsize = 15)
plt.ylabel('Phase',fontsize = 15)
plt.title('Unwrapped phase map cross section for {} \n (vertical fringes)'.format(type_unwrap), fontsize = 20)
plt.figure()
plt.plot(unwrap_h[:,500])
plt.title('Wrapped phase map cross section for {} \n (horizontal fringes)'.format(type_unwrap), fontsize = 20)
plt.xlabel('Dimension',fontsize = 15)
plt.ylabel('Phase',fontsize = 15)

