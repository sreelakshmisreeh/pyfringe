# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 20:13:34 2023

@author: kl001
"""

import numpy as np
import cupy as cp
import nstep_fringe as nstep
import nstep_fringe_cp as nstep_cp


def toy_generate():
    """
    Function to generate toy data
    :return fringe_arr: Array of vertical horizontal fringe patterns of two levels of pitch 50 and 20. 
                        Each has N = 3 images. First 6 patterns are high pitch vertical and horizontal.
    :rtypr fringe_arr: np.ndarray: float
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
    np.save('test_data/toy_data.npy',fringe_arr)
    return fringe_arr

def main():
    test_limit = 0.9
    pitch_list = [50, 20]
    N_list = [3, 3]
    fringe_arr_np = toy_generate()
    fringe_arr_cp = cp.asarray(fringe_arr_np)
    delta_deck_np = nstep.delta_deck_gen(N_list[0], height=fringe_arr_np.shape[1], width=fringe_arr_np.shape[2])
    delta_deck_cp = nstep_cp.delta_deck_gen_cp(N_list[0], height=fringe_arr_cp.shape[1], width=fringe_arr_cp.shape[2])
    if delta_deck_np.all() == cp.asnumpy(delta_deck_cp).all():
        print('\n Delta deck generation match')
        masked_img_np_v1, modulation_np_v1, average_int_np_v1, phase_map_np_v1 = nstep.phase_cal(fringe_arr_np[0:3], delta_deck_np, test_limit)
        masked_img_cp_v1, modulation_cp_v1, average_int_cp_v1, phase_map_cp_v1 = nstep_cp.phase_cal_cp(fringe_arr_cp[0:3], delta_deck_cp, test_limit)
        masked_img_np_v2, modulation_np_v2, average_int_np_v2, phase_map_np_v2 = nstep.phase_cal(fringe_arr_np[6:9], delta_deck_np, test_limit)
        masked_img_cp_v2, modulation_cp_v2, average_int_cp_v2, phase_map_cp_v2 = nstep_cp.phase_cal_cp(fringe_arr_cp[6:9], delta_deck_cp, test_limit)
        
        if (phase_map_np_v1.all() == cp.asnumpy(phase_map_cp_v1).all()) & (phase_map_np_v2.all() == cp.asnumpy(phase_map_cp_v2).all()):
            print('\n All vertical phase maps match')
            phase_arr_np = [phase_map_np_v1, phase_map_np_v2]
            phase_arr_cp = [phase_map_cp_v1, phase_map_cp_v2]
            multifreq_unwrap_np_v, k_arr_np_v = nstep.multifreq_unwrap(pitch_list, phase_arr_np, 1, 'v')
            multifreq_unwrap_cp_v, k_arr_cp_v = nstep_cp.multifreq_unwrap_cp(pitch_list, phase_arr_cp, 1, 'v')
            if multifreq_unwrap_np_v.all() == cp.asnumpy(multifreq_unwrap_cp_v).all():
                print('\n Vertical unwrapped phase maps match')
            else:
                print('\n Vertical unwrapped phase map mismatch ')  
        else:
            print('\n Vertical phase map mismatch')
        masked_img_np_h1, modulation_np_h1, average_int_np_h1, phase_map_np_h1 = nstep.phase_cal(fringe_arr_np[3:6], delta_deck_np, test_limit)
        masked_img_cp_h1, modulation_cp_h1, average_int_cp_h1, phase_map_cp_h1 = nstep_cp.phase_cal_cp(fringe_arr_cp[3:6], delta_deck_cp, test_limit)
        masked_img_np_h2, modulation_np_h2, average_int_np_h2, phase_map_np_h2 = nstep.phase_cal(fringe_arr_np[9:12], delta_deck_np, test_limit)
        masked_img_cp_h2, modulation_cp_h2, average_int_cp_h2, phase_map_cp_h2 = nstep_cp.phase_cal_cp(fringe_arr_cp[9:12], delta_deck_cp, test_limit)
        
        if (phase_map_np_h1.all() == cp.asnumpy(phase_map_cp_h1).all()) & (phase_map_np_h2.all() == cp.asnumpy(phase_map_cp_h2).all()):
            print('\n All horizontal phase maps match')
            phase_arr_np = [phase_map_np_h1, phase_map_np_h2]
            phase_arr_cp = [phase_map_cp_h1, phase_map_cp_h2]
            multifreq_unwrap_np_h, k_arr_np_h = nstep.multifreq_unwrap(pitch_list, phase_arr_np, 1, 'h')
            multifreq_unwrap_cp_h, k_arr_cp_h = nstep_cp.multifreq_unwrap_cp(pitch_list, phase_arr_cp, 1, 'h')
            if multifreq_unwrap_np_h.all() == cp.asnumpy(multifreq_unwrap_cp_h).all():
                print('\n Horizontal unwrapped phase maps match')
            else:
                print('\n Horizontal unwrapped phase map mismatch ')  
        else:
            print('\n Horizontal phase map mismatch')
    else:
        print('Numpy and cupy delta deck arrays does not match')
    return


if __name__ == '__main__':
    main()
