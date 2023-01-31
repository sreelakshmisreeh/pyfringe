# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 20:13:34 2023

@author: kl001
"""

import numpy as np
import cupy as cp
import nstep_fringe as nstep
import nstep_fringe_cp as nstep_cp
import pickle


def toy_generate():
    """
    Function to generate toy data
    :return fringe_arr: Array of vertical horizontal fringe patterns of two levels of pitch 50 and 20. 
                        Each has N = 3 images. First 6 patterns are high pitch vertical and horizontal.
    :rtype fringe_arr: np.ndarray: float
    """
    height = 118
    width = 118
    inte_rang =[5, 254]
    phase_st = 0
    N_list = [3, 3]
    pitch_list = [50, 20]
    delta = 2 * np.pi * np.arange(1, N_list[0] + 1) / N_list[0]
    one_block = np.ones((N_list[0], height, width))
    delta_deck = np.stack((delta[0]*one_block[0], delta[1] * one_block[1], delta[2]*one_block[2]))
    fringe_lst=[]
    for p, n in zip(pitch_list, N_list):
        cos_v, absolute_phi_v = nstep.cos_func(inte_rang, p, 'v', phase_st, delta_deck)
        cos_h, absolute_phi_h = nstep.cos_func(inte_rang, p, 'h', phase_st, delta_deck)
        cos_v = np.pad(cos_v, pad_width=((0, 0), (5, 5), (5, 5)), mode='constant', constant_values=np.nan) #118+10=128
        cos_h = np.pad(cos_h, pad_width=((0, 0), (5, 5), (5, 5)), mode='constant', constant_values=np.nan) #118+10=128
        fringe_lst.append(np.vstack((cos_v, cos_h)))
    fringe_arr = np.ceil(np.vstack(fringe_lst))
    #np.save('test_data/toy_data.npy', fringe_arr)
    return fringe_arr

def main():
    test_limit = 0.9
    pitch_list = [50, 20]
    fringe_arr_np = toy_generate()
    fringe_arr_cp = cp.asarray(fringe_arr_np)
    modulation_np_v1, white_img_np_v1, sin_lst_np_v1, cos_lst_np_v1, mask_np_v1 = nstep.phase_cal(fringe_arr_np[0:3], test_limit)
    modulation_cp_v1, white_img_cp_v1, sin_lst_cp_v1, cos_lst_cp_v1, mask_cp_v1 = nstep_cp.phase_cal_cp(fringe_arr_cp[0:3], test_limit)
    modulation_np_v2, white_img_np_v2, sin_lst_np_v2, cos_lst_np_v2, mask_np_v2 = nstep.phase_cal(fringe_arr_np[6:9], test_limit)
    modulation_cp_v2, white_img_cp_v2, sin_lst_cp_v2, cos_lst_cp_v2, mask_cp_v2 = nstep_cp.phase_cal_cp(fringe_arr_cp[6:9], test_limit)
    mask_np_v = mask_np_v1 & mask_np_v2
    flag_np_v = np.where(mask_np_v == True)
    sin_lst_np_v = np.array([sin_lst_np_v1[flag_np_v], sin_lst_np_v2[flag_np_v]])
    cos_lst_np_v = np.array([cos_lst_np_v1[flag_np_v], cos_lst_np_v2[flag_np_v]])
    phase_np_v = -np.arctan2(sin_lst_np_v, cos_lst_np_v)
    mask_cp_v = mask_cp_v1 & mask_cp_v2
    flag_cp_v = cp.where(mask_cp_v == True)
    sin_lst_cp_v = cp.array([sin_lst_cp_v1[flag_cp_v], sin_lst_cp_v2[flag_cp_v]])
    cos_lst_cp_v = cp.array([cos_lst_cp_v1[flag_cp_v], cos_lst_cp_v2[flag_cp_v]])
    phase_cp_v = -cp.arctan2(sin_lst_cp_v, cos_lst_cp_v)
    if (phase_np_v.all() == cp.asnumpy(phase_cp_v).all()):
        print('\n All vertical phase maps match')
        multifreq_unwrap_np_v, k_arr_np_v = nstep.multifreq_unwrap(pitch_list, phase_np_v, 1, 'v')
        multifreq_unwrap_cp_v, k_arr_cp_v = nstep_cp.multifreq_unwrap_cp(pitch_list, phase_cp_v, 1, 'v')
        if multifreq_unwrap_np_v.all() == cp.asnumpy(multifreq_unwrap_cp_v).all():
            print('\n Vertical unwrapped phase maps match')
            vertical_fringes_np = {"phase_map_np_v": phase_np_v,
                                   "multifreq_unwrap_np_v": multifreq_unwrap_np_v}
            vertical_fringes_cp = {"phase_map_cp_v": phase_cp_v,
                                   "multifreq_unwrap_cp_v": multifreq_unwrap_cp_v}
            with open(r'test_data\vertical_fringes_np.pickle', 'wb') as f:
                pickle.dump(vertical_fringes_np, f)
            with open(r'test_data\vertical_fringes_cp.pickle', 'wb') as f:
                pickle.dump(vertical_fringes_cp, f)
        else:
            print('\n Vertical unwrapped phase map mismatch ')  
    else:
        print('\n Vertical phase map mismatch')
    modulation_np_h1, white_img_np_h1, sin_lst_np_h1, cos_lst_np_h1, mask_np_h1 = nstep.phase_cal(fringe_arr_np[3:6], test_limit)
    modulation_cp_h1, white_img_cp_h1, sin_lst_cp_h1, cos_lst_cp_h1, mask_cp_h1 = nstep_cp.phase_cal_cp(fringe_arr_cp[3:6], test_limit)
    modulation_np_h2, white_img_np_h2, sin_lst_np_h2, cos_lst_np_h2, mask_np_h2 = nstep.phase_cal(fringe_arr_np[9:12], test_limit)
    modulation_cp_h2, white_img_cp_h2, sin_lst_cp_h2, cos_lst_cp_h2, mask_cp_h2 = nstep_cp.phase_cal_cp(fringe_arr_cp[9:12], test_limit)
    mask_np_h = mask_np_h1 & mask_np_h2
    flag_np_h = np.where(mask_np_h == True)
    sin_lst_np_h = np.array([sin_lst_np_h1[flag_np_h], sin_lst_np_h2[flag_np_h]])
    cos_lst_np_h = np.array([cos_lst_np_h1[flag_np_h], cos_lst_np_h2[flag_np_h]])
    phase_np_h = -np.arctan2(sin_lst_np_h, cos_lst_np_h)
    mask_cp_h = mask_cp_h1 & mask_cp_h2
    flag_cp_h = cp.where(mask_cp_h == True)
    sin_lst_cp_h = cp.array([sin_lst_cp_h1[flag_cp_h], sin_lst_cp_h2[flag_cp_h]])
    cos_lst_cp_h = cp.array([cos_lst_cp_h1[flag_cp_h], cos_lst_cp_h2[flag_cp_h]])
    phase_cp_h = -cp.arctan2(sin_lst_cp_h, cos_lst_cp_h)
    if (phase_np_h.all() == cp.asnumpy(phase_cp_h).all()):
        print('\n All horizontal phase maps match')
        multifreq_unwrap_np_h, k_arr_np_h = nstep.multifreq_unwrap(pitch_list, phase_np_h, 1, 'h')
        multifreq_unwrap_cp_h, k_arr_cp_h = nstep_cp.multifreq_unwrap_cp(pitch_list, phase_cp_h, 1, 'h')
        if multifreq_unwrap_np_h.all() == cp.asnumpy(multifreq_unwrap_cp_h).all():
            print('\n Horizontal unwrapped phase maps match')
            horizontal_fringes_np = {"phase_map_np_h": phase_np_h,
                                     "multifreq_unwrap_np_h": multifreq_unwrap_np_h}
            horizontal_fringes_cp = {"phase_map_cp_h": phase_cp_h,
                                     "multifreq_unwrap_cp_h": multifreq_unwrap_cp_v}
            with open(r'test_data\horizontal_fringes_np.pickle', 'wb') as f:
                pickle.dump(horizontal_fringes_np, f)
            with open(r'test_data\horizontal_fringes_cp.pickle', 'wb') as f:
                pickle.dump(horizontal_fringes_cp, f)
        else:
            print('\n Horizontal unwrapped phase map mismatch ')  
    else:
        print('\n Horizontal phase map mismatch')

    return


if __name__ == '__main__':
    main()
