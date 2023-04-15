# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 08:35:41 2023

@author: kl001
"""

import numpy as np
import os
import sys
import image_acquisation as acq
import pwlf
import pickle
import matplotlib.pyplot as plt
import glob
import cv2



def capture(image_index_list,
            pattern_num_list,
            savedir,
            number_scan,
            proj_exposure_period,
            proj_frame_period):

    for acq_index, image_list in enumerate(image_index_list):
        result = acq.run_proj_single_camera(savedir=savedir,
                                             preview_option='Once',
                                             number_scan=number_scan,
                                             acquisition_index=acq_index,
                                             image_index_list=image_list,
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
        print("-------------------------------------",image_list,"------------------------------------------------")
    return result

def piecewise_fitting(sorted_int, sorted_std, pitch_list, freq_list, savedir):
    model_list = []; results = []; slopes =[]; intercepts = []
    for k in range(len(pitch_list)):
        my_pwlf = pwlf.PiecewiseLinFit(sorted_int[k], sorted_std[k])
        res = my_pwlf.fit(2)
        results.append(res)
        slope = my_pwlf.calc_slopes()
        slopes.append(slope)
        inter = my_pwlf.intercepts
        intercepts.append(inter)
        xHat = np.linspace(min(sorted_int[k]),max(sorted_int[k]), num=10000)
        yHat = my_pwlf.predict(xHat)
        model_list.append(my_pwlf)
        fig, ax = plt.subplots()
        ax.plot(sorted_int[k], sorted_std[k],"o",c="b", alpha=0.5)
        ax.plot(xHat, yHat, '-', c="r")
        ax.set_xlabel("Captured intensity", fontsize=20)
        ax.set_ylabel("$\sigma$", fontsize=20)
        ax.set_title("Pitch = %d , Frequency = %.3f"%(pitch_list[k],freq_list[k]))
        ax.text(0.1,0.9, "$R^2=%.3f$"%(my_pwlf.r_squared()), fontsize=15, transform=ax.transAxes)
        ax.text(0.1,0.8, "Turning point=%.3f"%(res[1]), fontsize=15, transform=ax.transAxes)
        ax.text(0.1,0.7, "Slopes=%.3f, %.3f"%(slope[0], slope[1]), fontsize=15, transform=ax.transAxes)
        ax.text(0.1,0.6, "Intercepts=%.3f, %.3f"%(inter[0], inter[1]), fontsize=15, transform=ax.transAxes)
        ax.set_ylim(0,4.5)
        ax.set_xlim(0,255)
        plt.tight_layout()
        fig.savefig(os.path.join(savedir,"sigma_%d.png"%(int(freq_list[k]))))
    np.save(os.path.join(savedir,"lut_breakpts.npy"),np.array(results))
    np.save(os.path.join(savedir,"lut_slopes.npy"),np.array(slopes))
    np.save(os.path.join(savedir,"lut_intercepts.npy"),np.array(intercepts))
    with open(os.path.join(savedir,"lut_models.pkl"), "wb") as tt:
         pickle.dump(model_list,tt)
    pitch_dict = {pitch_list[i]: model_list[i] for i in range(len(model_list))}
    freq_dict = {freq_list[i]: model_list[i] for i in range(len(model_list))}
    with open(os.path.join(savedir,"lut_models_pitch_dict.pkl"), "wb") as tt:
         pickle.dump(pitch_dict,tt)
    with open(os.path.join(savedir,"lut_models_freq_dict.pkl"), "wb") as tt:
         pickle.dump(freq_dict,tt)         
    return model_list, np.array(results), np.array(slopes), np.array(intercepts)

def main():
    option = input("Please choose: 1:Capture data \n2: Analysis")
    iterations = int(input("No. of scans"))
    capt_savedir =  r"E:\lut_calibration"
    res_savedir = r"E:\result_lut_calib"
    if option == "1":
        proj_exposure_period = 29000
        proj_frame_period = 36000
        
        image_index_list =   np.repeat(np.arange(0,31),3).reshape(31,3).tolist() 
        pattern_num_list = [0,1,2] 
            
        result = capture(image_index_list=image_index_list, 
                         pattern_num_list=pattern_num_list,
                         savedir=capt_savedir,
                         number_scan=iterations,
                         proj_exposure_period=proj_exposure_period,
                         proj_frame_period=proj_frame_period)
    if option == "2":
        
        camx = 500
        camy= 300
        deltay = 500
        deltax = 900
        lut_pitch = np.array([1375, 1200, 1100, 1000, 912, 450, 275, 225, 150, 120, 110, 90, 75,
                     65, 55, 50, 45, 40, 35,30, 25, 20, 19,18, 17, 16, 15, 14, 13, 12, 11]) 
        freq_list = 912/lut_pitch
        path = sorted(glob.glob(os.path.join(capt_savedir,'*.jpeg')), key=lambda x:int(os.path.basename(x)[6:8]))
        data_arr = np.array([cv2.imread(file,0) for file in path[2::3]]).reshape(len(lut_pitch),iterations,1200,1920)
        data_crop = data_arr[:,:,camy: camy + deltay, camx: camx + deltax]
        mean_captured = np.mean(data_crop, axis=1).reshape(len(lut_pitch),deltax*deltay)
        std_captured = np.std(data_crop, axis=1).reshape(len(lut_pitch),deltax*deltay)
        int_index = np.argsort(mean_captured, axis=-1)
        sorted_int = mean_captured[np.arange(mean_captured.shape[0])[:,None], int_index]
        sorted_std = std_captured[np.arange(std_captured.shape[0])[:,None], int_index]
        model_list, results, slopes, intercepts =  piecewise_fitting(sorted_int, sorted_std, lut_pitch, freq_list, res_savedir)
    return result

if __name__ == '__main__':
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
