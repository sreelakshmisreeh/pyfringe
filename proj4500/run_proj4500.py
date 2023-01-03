#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 14:14:03 2022

@author: Sreelakshmi
"""
import os
import sys
sys.path.append(r'C:\Users\kl001\Documents\pyfringe_test\proj4500')
sys.path.append(r'C:\Users\kl001\Documents\pyfringe_test')
import proj4500
import FringeAcquisition as fa
import cv2
from time import perf_counter_ns
import PySpin



#proj4500.power_up()
#proj4500.pattern_trigger_mode(exposure_period = 27084, frame_period = 33334)


def proj_cam_acquire_images(cam, acquisition_index, savedir, triggerType, exposure_period = 27084, frame_period = 33334):
    """
    This function acquires and saves one image from a device. Note that camera 
    must be initialized before calling this function, i.e., cam.Init() must be 
    called before calling this function.

    :param cam: Camera to acquire images from.
    :param savedir: directory to save images
    :param acquisition_index: the index number of the current acquisition.
    :param triggerType: trigger type, must be one of {"software", "hardware"}
    :type cam: CameraPtr
    :tyoe savedir: str
    :type acquisition_index: int
    :type triggerType: str
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    print('*** IMAGE ACQUISITION ***\n')

    result = True        

    # live view        
    cam.BeginAcquisition()        
    while True:                
        ret, frame = fa.capture_image(cam)       
        img_show = cv2.resize(frame, None, fx=0.5, fy=0.5)
        cv2.imshow("press q to quit", img_show)    
        key = cv2.waitKey(1)        
        if key == ord("q"):
            break
    cam.EndAcquisition()
    cv2.destroyAllWindows()
    
    # Retrieve, convert, and save image
    fa.activate_trigger(cam)
    cam.BeginAcquisition()        
    
    if triggerType == "software":
        start = perf_counter_ns()            
        cam.TriggerSoftware.Execute()    
        ret, image_array = fa.capture_image(cam=cam)                
        end = perf_counter_ns()
        t = (end - start)/1e9
        print('time spent: %2.3f s' % t)                
        if ret:
            filename = 'Acquisition-%02d.jpg' %acquisition_index
            save_path = os.path.join(savedir, filename)                    
            cv2.imwrite(save_path, image_array)
            print('Image saved at %s' % save_path)
        else:
            print('Capture failed')
    
    if triggerType == "hardware":
        count = 0        
        start = perf_counter_ns()   
        proj4500.pattern_trigger_mode(exposure_period = exposure_period, frame_period = frame_period)                        
        while count < 15:
            try:
                ret, image_array = fa.capture_image(cam=cam)
            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                ret = False
                image_array = None
                pass
                                
            if ret:
                print("extract successfully")
                filename = 'Acquisition-%02d-%02d.jpg' %(acquisition_index,count)
                save_path = os.path.join(savedir, filename)
                cv2.imwrite(save_path, image_array)
                print('Image saved at %s' % save_path)
                count += 1
                start = perf_counter_ns()
                print('waiting clock is reset')
            else:
                end = perf_counter_ns()
                waiting_time = (end - start)/1e9
                print('Capture failed, time spent %2.3f s before 10s timeout'%waiting_time)
                if waiting_time > 10:
                    print('timeout is reached, stop capturing image ...')
                    break
    
    cam.EndAcquisition()
    fa.deactivate_trigger(cam)        

    return result

def run_single_camera(cam, savedir, acquisition_index, triggerType):
    """
    Initialize and configurate a camera and take one image.

    :param cam: Camera to run on.
    :type cam: CameraPtr
    :param savedir: directory to save images
    :param acquisition_index: the index of acquisition
    :type savedir: str
    :type acquisition_index: int
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True
        # Initialize camera
        cam.Init()
        # config camera
        result &= fa.cam_configuration(cam, triggerType)        
        # Acquire images        
        result &= proj_cam_acquire_images(cam, acquisition_index, savedir, triggerType)
        # Deinitialize camera        
        cam.DeInit()
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False
    return result

def main():
    triggerType = "hardware"
    result, system, cam_list, num_cameras = fa.sysScan()
    
    if result:
        # Run example on each camera
        savedir = r'C:\Users\kl001\Documents\grasshopper3_python\images'
        fa.clearDir(savedir)
        for i, cam in enumerate(cam_list):    
            print('Running example for camera %d...'%i)
            acquisition_index=0
            result &= run_single_camera(cam, savedir, acquisition_index, triggerType=triggerType) # only one acquisition            
            print('Camera %d example complete...'%i)
    
        # Release reference to camera
        # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
        # cleaned up when going out of scope.
        # The usage of del is preferred to assigning the variable to None.
        if cam_list:    
            del cam
        else:
            print('Camera list is empty! No camera is detected, please check camera connection.')    
    else:
        pass
    # Clear camera list before releasing system
    cam_list.Clear()
    # Release system instance
    system.ReleaseInstance()    
    return result

if __name__ == '__main__':
    if main():
        sys.exit(0)
    else:
        sys.exit(1)