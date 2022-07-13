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
inte_rang = [50,250]
# path to save pattern
path = '/Users/Sreelakshmi/Documents/Raspberry/reconstruction/July_5_cali_img'
#%% Phase coded unwrapping
#type of unwrapping 
type_unwrap =  'phase'
ph_pitch_list =[18]
ph_N_list =[9]

ph_phase_st = -np.pi
# To generate phase codedpatterns in both directions
ph_fringe_arr, ph_delta_deck_list = nstep.calib_generate(width, height, type_unwrap, ph_N_list, ph_pitch_list, ph_phase_st, inte_rang, path)

# To veiw generated fringe patterns. Can also be plotted using matplotlib
cv2.startWindowThread()
cv2.namedWindow('proj',cv2.WINDOW_NORMAL)
for i,img in enumerate(ph_fringe_arr):
       cv2.imshow('proj',img)
       cv2.waitKey()
cv2.destroyAllWindows()
cv2.waitKey(1)

## To generate phase codedpatterns in one directions
direc = 'h'
ph_fringe_arr_d, ph_delta_deck_list_d = nstep.recon_generate(width, height, type_unwrap, ph_N_list, ph_pitch_list, ph_phase_st, inte_rang, direc, path)

# To veiw generated fringe patterns.
for i,img in enumerate(ph_fringe_arr_d):
    plt.figure()
    plt.imshow(img)

#%% Multifrequency unwrapping
#type of unwrapping 
type_unwrap =  'multifreq'
mf_pitch_list =[1375, 275, 55, 11] 
mf_N_list = [3, 3, 3, 9]
mf_phase_st = 0
# To generate multifrequency patterns in both directions
mf_fringe_arr, mf_delta_deck_list = nstep.calib_generate(width, height, type_unwrap, mf_N_list, mf_pitch_list, mf_phase_st, inte_rang, path)

# To veiw generated fringe patterns.
cv2.startWindowThread()
cv2.namedWindow('proj',cv2.WINDOW_NORMAL)
for i,img in enumerate(mf_fringe_arr):
       cv2.imshow('proj',img)
       cv2.waitKey()       
cv2.destroyAllWindows()
cv2.waitKey(1)      

## To generate multifrequency patterns in one directions
direc = 'h'
mf_fringe_arr_d, mf_delta_deck_list_d = nstep.recon_generate(width, height, type_unwrap, mf_N_list, mf_pitch_list, mf_phase_st, inte_rang, direc, path)

for i,img in enumerate(mf_fringe_arr_d):
    plt.figure()
    plt.imshow(img)

#%% Multiwavelength unwrapping

#type of unwrapping 
type_unwrap =  'multiwave'
mw_pitch_list = [139,21,18]
mw_N_list =[5,5,9]
mw_phase_st = 0
# To generate multiwave patterns in both directions
mw_fringe_arr, delta_deck_list = nstep.calib_generate(width, height, type_unwrap, mw_N_list, mw_pitch_list, mw_phase_st, inte_rang, path)

# To veiw generated fringe patterns.
cv2.startWindowThread()
cv2.namedWindow('proj',cv2.WINDOW_NORMAL)
for i,img in enumerate(mw_fringe_arr):
       cv2.imshow('proj',img)
       cv2.waitKey()
cv2.destroyAllWindows()
cv2.waitKey(1)
## To generate multiwave patterns in one directions
direc = 'h'
mw_fringe_arr_d, mw_delta_deck_list_d = nstep.recon_generate(width, height, type_unwrap, mw_N_list, mw_pitch_list, mw_phase_st, inte_rang, direc, path)
for i,img in enumerate(mw_fringe_arr_d):
    plt.figure()
    plt.imshow(img)






