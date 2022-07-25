#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 13:07:09 2022

@author: Sreelakshmi
"""

import sys
sys.path.append('/Users/Sreelakshmi/Documents/Github/pyfringe/functions')
import reconstruction as rc
import numpy as np
import matplotlib.pyplot as plt

width = 800; height = 1280
inte_rang = [50,240]
# Set type of unwrapping
type_unwrap = 'multifreq'
limit = 2
dist = 700
delta_dist =300
direc = 'v'
#multifrequency unwrapping parameters
if type_unwrap == 'multifreq':
    pitch_list =[1375, 275, 55, 11] 
    N_list = [3, 3, 3, 9]
    kernel = 1
    phase_st = 0
    calib_path = '/Users/Sreelakshmi/Documents/Raspberry/reconstruction/testing_data/reconstruction/'
    obj_path = '/Users/Sreelakshmi/Documents/Raspberry/reconstruction/testing_data/reconstruction/multifreq'
    
# multiwavelength unwrapping parameters
if type_unwrap == 'multiwave':
    pitch_list = [139,21,18]
    N_list =[5,5,9]
    kernel = 30;
    phase_st = 0
    calib_path = '/Users/Sreelakshmi/Documents/Raspberry/reconstruction/testing_data/reconstruction/'
    obj_path = '/Users/Sreelakshmi/Documents/Raspberry/reconstruction/testing_data/reconstruction/multiwave'

# phase coding unwrapping parameters
if type_unwrap == 'phase':
    pitch_list =[18]
    N_list =[9]
    kernel = 30
    phase_st = -np.pi
    calib_path = '/Users/Sreelakshmi/Documents/Raspberry/reconstruction/testing_data/reconstruction/'
    obj_path = '/Users/Sreelakshmi/Documents/Raspberry/reconstruction/testing_data/reconstruction/phase'


    
obj_cordi, obj_color = rc.obj_reconst_wrapper(width, height, pitch_list, N_list, limit, dist, delta_dist, phase_st, direc, type_unwrap, calib_path, obj_path, kernel)


    
    
    
    
    