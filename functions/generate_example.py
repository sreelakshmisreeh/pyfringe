#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 11:13:53 2022

@author: Sreelakshmi
"""

import sys
sys.path.append('/Users/Sreelakshmi/Documents/Github/pyfringe/')
import nstep_fringe as nstep
import cv2




#%% Phase coded
width = 800; height = 1280
#type of unwrapping 
#type_unwrap =  'multiwave'
type_unwrap =  'multifreq'
pitch_list =[1375, 275, 55, 11] 
#pitch_list = [139,21,18]
N_list = [3, 3, 3, 9]
#N_list =[5,5,9]
path = '/Users/Sreelakshmi/Documents/Raspberry/reconstruction/July_5_cali_img'
phase_st = 0
inte_rang = [50,250]

fringe_lst, delta_deck_list = nstep.calib_generate(width, height, type_unwrap, pitch_list, N_list,phase_st, inte_rang, path)

#%%
cv2.startWindowThread()
cv2.namedWindow('proj',cv2.WINDOW_NORMAL)
for i,img in enumerate(fringe_lst):
       cv2.imshow('proj',img)
       cv2.waitKey()
       
  
cv2.destroyAllWindows()
cv2.waitKey(1)
#%%
direc = 'h'
fringe_lst, delta_deck_list = recon_generate(width, height, type_unwrap, pitch_list, N_list, phase_st, inte_rang, direc, path)
#%%
cv2.startWindowThread()
cv2.namedWindow('proj',cv2.WINDOW_NORMAL)
for i,img in enumerate(fringe_lst):
       cv2.imshow('proj',img)
       cv2.waitKey()
       
  
cv2.destroyAllWindows()
cv2.waitKey(1)








