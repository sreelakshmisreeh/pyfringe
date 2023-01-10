# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 14:14:57 2023

@author: kl001
"""
import os
import sys
import numpy as np
import gspy
import lcpy
import cv2
from time import perf_counter_ns
import usb.core
import PySpin

    

def proj_cam_preview(cam, 
                     lcr, 
                     proj_exposure_period, 
                     proj_frame_period, 
                     preview_type,
                     image_index):
    
    delta = 100      
  
    #set projector configuration
    lcr.set_pattern_config(num_lut_entries= 1, 
                           do_repeat = True, 
                           num_pats_for_trig_out2 = 1, 
                           num_images = 1)
    lcr.set_exposure_frame_period( proj_exposure_period, proj_frame_period)
    if preview_type == 'focus':
        lcr.image_flash_index([image_index],0)
        lcr.send_pattern_lut(trig_type = 0, 
                             bit_depth = 8, 
                             led_select = 0b111,
                             swap_location_list = [0],
                             image_index_list = [image_index], 
                             pattern_num_list = [0], 
                             starting_address = 0,  
                             do_insert_black = False) 
        
    elif preview_type == 'preview':
        lcr.image_flash_index([image_index],0)
        lcr.send_pattern_lut(trig_type = 0, 
                             bit_depth = 8, 
                             led_select = 0b111,
                             swap_location_list = [0],
                             image_index_list = [image_index], 
                             pattern_num_list = [0], 
                             starting_address = 0,  
                             do_insert_black = False)  
    ans = lcr.start_pattern_lut_validate()
    # live view   
    if (not int(ans)) or (int(ans,2)==8):
        cam.BeginAcquisition()
        lcr.pattern_display('start')
        while True:                
            ret, frame = gspy.capture_image(cam)       
            img_show = cv2.resize(frame, None, fx=0.5, fy=0.5)
            if preview_type == 'preview':
                img_show_color = cv2.cvtColor(img_show, cv2.COLOR_GRAY2BGR)
                img_show_color[img_show == 255] = [0,0,255] #over exposed
            else:
                img_show_color = img_show
            center_y = int(img_show.shape[0]/2)
            center_x = int(img_show.shape[1]/2)
            cv2.line(img_show_color,(center_x,center_y - delta),(center_x,center_y + delta),(0,0,0),5)
            cv2.line(img_show_color,(center_x - delta,center_y),(center_x + delta,center_y),(0,0,0),5)
            cv2.imshow("press q to quit", img_show_color)    
            key = cv2.waitKey(1)        
            if key == ord("q"):
                lcr.pattern_display('stop')
                break
        cam.EndAcquisition()
        cv2.destroyAllWindows()
    return True

def run_proj_cam_capt(cam, 
                      lcr, 
                      savedir,
                      acquisition_index, 
                      image_index_list, 
                      pattern_num_list, 
                      cam_capt_timeout,
                      proj_exposure_period, 
                      proj_frame_period, 
                      do_insert_black,
                      preview_image_index,
                      pprint_proj_status):
    
    
    """
    This function acquires and saves one image from a device. Note that camera 
    must be initialized before calling this function, i.e., cam.Init() must be 
    called before calling this function.

    :param cam: Camera to acquire images from.
    :param savedir: directory to save images
    :param acquisition_index: the index number of the current acquisition.
    :param cam_triggerType: camera trigger type, must be one of {"software", "hardware"}
    :param image_index_list: projector pattern sequence to create and project
    :param proj_exposure_period : projector exposure period in microseconds 
    :param proj_frame_period : projector frame period in microseconds 
    :type cam: CameraPtr
    :type savedir: str
    :type acquisition_index: int
    :type triggerType: str
    :type: image_index_list: projector pattern sequence list
    :type: proj_exposure_period: float
    type: proj_frame_period: float
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    
    nodemap = cam.GetNodeMap()
    triggerSourceSymbolic = gspy.get_IEnumeration_node_current_entry_name(nodemap, 'TriggerSource', verbose=False)
    
    print('*** IMAGE ACQUISITION ***\n')

    result = True      
    # Retrieve, convert, and save image
    gspy.activate_trigger(cam)
    cam.BeginAcquisition()        
    
    if triggerSourceSymbolic == "Software":
        start = perf_counter_ns()            
        cam.TriggerSoftware.Execute()               
        ret, image_array = gspy.capture_image(cam=cam)                
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
            result = False
    
    if triggerSourceSymbolic == "Line0":
        count = 0        
        total_dual_time_start = perf_counter_ns()
        start = perf_counter_ns() 
        #Configure projector
        image_LUT_entries, swap_location_list = lcpy.get_image_LUT_swap_location(image_index_list)
        lcr.set_pattern_config(num_lut_entries= len(image_index_list),
                               do_repeat = False,  
                               num_pats_for_trig_out2 = len(image_index_list),
                               num_images = len(image_LUT_entries))
        lcr.set_exposure_frame_period( proj_exposure_period, proj_frame_period)
        # To set new image LUT
        lcr.image_flash_index(image_LUT_entries,0)
        # To set pattern LUT table    
        lcr.send_pattern_lut(trig_type = 0 , 
                             bit_depth = 8, 
                             led_select = 0b111,
                             swap_location_list = swap_location_list, 
                             image_index_list = image_index_list, 
                             pattern_num_list = pattern_num_list, 
                             starting_address = 0,
                             do_insert_black = do_insert_black)
        if pprint_proj_status:# Print all projector current attributes set
            lcr.pretty_print_status()
        ans = lcr.start_pattern_lut_validate()
        #Check validation status
        if (not int(ans)) or (int(ans,2)==8):   
            lcr.pattern_display('start') 
                 
            while count < len(image_index_list):
                try:
                    ret, image_array = gspy.capture_image(cam=cam)
                except PySpin.SpinnakerException as ex:
                    print('Error: %s' % ex)
                    ret = False
                    image_array = None
                    pass
                                    
                if ret:
                    print("extract successfull")
                    filename = 'capt%d_%d.jpg' %(acquisition_index,count)
                    save_path = os.path.join(savedir, filename)
                    cv2.imwrite(save_path, image_array)
                    print('Image saved at %s' % save_path)
                    count += 1
                    start = perf_counter_ns()
                    print('waiting clock is reset')
                else:
                    end = perf_counter_ns()
                    waiting_time = (end - start)/1e9
                    print('Capture failed, time spent %2.3f s before %2.3f s timeout'%(waiting_time,cam_capt_timeout))
                    if waiting_time > cam_capt_timeout:
                        print('timeout is reached, stop capturing image ...')
                        break
            total_dual_time_end = perf_counter_ns()
            total_dual_time = (total_dual_time_end - total_dual_time_start)/1e9
            print('Total dual device time:%.3f'%total_dual_time)
        else:
            result = False
    
    cam.EndAcquisition()
    gspy.deactivate_trigger(cam)        

    return result

def proj_cam_acquire_images(cam,
                            lcr, 
                            savedir,
                            preview_option, 
                            number_scan,
                            acquisition_index,  
                            image_index_list,
                            pattern_num_list, 
                            cam_capt_timeout,
                            proj_exposure_period, 
                            proj_frame_period,
                            do_insert_black,
                            preview_image_index,
                            pprint_proj_status,
                            focus_image_index):
    result = True
    if not focus_image_index == None:
        result &= proj_cam_preview(cam, 
                             lcr, 
                             proj_exposure_period, 
                             proj_frame_period, 
                             'focus',
                             focus_image_index)
    
    if (number_scan == 1) & (preview_option == 'Once'):
        result &= proj_cam_preview(cam, 
                             lcr, 
                             proj_exposure_period, 
                             proj_frame_period,
                             'preview',
                             preview_image_index)
        
        result &= run_proj_cam_capt(cam, 
                                    lcr, 
                                    savedir,
                                    acquisition_index, 
                                    image_index_list, 
                                    pattern_num_list, 
                                    cam_capt_timeout,
                                    proj_exposure_period, 
                                    proj_frame_period, 
                                    do_insert_black,
                                    preview_image_index,
                                    pprint_proj_status)
        
        
    elif (number_scan > 1) & (preview_option == 'Always'):
        initial_acq_index = acquisition_index
        for i in range(number_scan):
            result &= proj_cam_preview(cam, 
                                 lcr, 
                                 proj_exposure_period, 
                                 proj_frame_period,
                                 'preview',
                                 preview_image_index)
            
            result &= run_proj_cam_capt(cam, 
                                        lcr, 
                                        savedir,
                                        (initial_acq_index + i), 
                                        image_index_list, 
                                        pattern_num_list, 
                                        cam_capt_timeout,
                                        proj_exposure_period, 
                                        proj_frame_period, 
                                        do_insert_black,
                                        preview_image_index,
                                        pprint_proj_status)
            
    elif (number_scan > 1) & (preview_option == 'Once'):
        initial_acq_index = acquisition_index
        result &= proj_cam_preview(cam, 
                             lcr, 
                             proj_exposure_period, 
                             proj_frame_period,
                             'preview',
                             preview_image_index)
        for i in range(number_scan):
            result &= run_proj_cam_capt(cam, 
                                        lcr, 
                                        savedir,
                                        (initial_acq_index + i), 
                                        image_index_list, 
                                        pattern_num_list, 
                                        cam_capt_timeout,
                                        proj_exposure_period, 
                                        proj_frame_period, 
                                        do_insert_black,
                                        preview_image_index,
                                        pprint_proj_status)
            
    elif (number_scan == 1) & (preview_option == 'Never'):
        result &= run_proj_cam_capt(cam, 
                                    lcr, 
                                    savedir,
                                    acquisition_index, 
                                    image_index_list, 
                                    pattern_num_list, 
                                    cam_capt_timeout,
                                    proj_exposure_period, 
                                    proj_frame_period, 
                                    do_insert_black,
                                    preview_image_index,
                                    pprint_proj_status)
    return result

def run_proj_single_camera(cam,
                           savedir,
                           preview_option, 
                           number_scan,
                           acquisition_index, 
                           cam_triggerType, 
                           image_index_list,
                           pattern_num_list,
                           cam_gain,
                           cam_bufferCount,
                           cam_capt_timeout,
                           proj_exposure_period, 
                           proj_frame_period,
                           do_insert_black,
                           preview_image_index,
                           pprint_proj_status,
                           focus_image_index ):
    """
    Initialize and configurate a camera and take one image.

    :param cam: Camera to run on.
    :type cam: CameraPtr
    :param savedir: directory to save images
    :param acquisition_index: the index of acquisition
    :param cam_triggerType: camera trigger type, must be one of {"software", "hardware"}
    :param image_index_list: projector pattern sequence to create and project
    :param proj_exposure_period : projector exposure period in microseconds 
    :param proj_frame_period : projector frame period in microseconds 
    :type savedir: str
    :type acquisition_index: int
    :type triggerType: str
    :type: image_index_list: projector pattern sequence list
    :type: proj_exposure_period: float
    type: proj_frame_period: float
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True
        device = usb.core.find(idVendor=0x0451, idProduct=0x6401) #finding the projector usb port
        device.set_configuration()

        lcr = lcpy.dlpc350(device)
        lcr.pattern_display('stop')
        
        # Initialize camera
        cam.Init()
        # config camera
        frameRate = 1e6/proj_frame_period
        result &= gspy.cam_configuration(cam=cam,
                                         triggerType=cam_triggerType,
                                         frameRate=frameRate, #proj_frame_period is in μs
                                         exposureTime=proj_exposure_period,
                                         gain=cam_gain,
                                         bufferCount=cam_bufferCount)        
        # Acquire images        
        result &= proj_cam_acquire_images(cam,
                                          lcr, 
                                          savedir,
                                          preview_option, 
                                          number_scan,
                                          acquisition_index,  
                                          image_index_list,
                                          pattern_num_list, 
                                          cam_capt_timeout,
                                          proj_exposure_period, 
                                          proj_frame_period,
                                          do_insert_black,
                                          preview_image_index,
                                          pprint_proj_status,
                                          focus_image_index)
        # Deinitialize camera        
        cam.DeInit()
        device.reset()
        del lcr
        del device
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False
    return result

def main():
    
    triggerType = "hardware"
    result, system, cam_list, num_cameras = gspy.sysScan()
    image_index_list = np.repeat(np.arange(0,5),3).tolist()
    pattern_num_list = [0,1,2] * len(set(image_index_list))
    if result:
        # Run example on each camera
        savedir = r'C:\Users\kl001\Documents\grasshopper3_python\images'
        gspy.clearDir(savedir)
        for i, cam in enumerate(cam_list):    
            print('Running example for camera %d...'%i)            
            result &= run_proj_single_camera(cam = cam,
                                             savedir = savedir,
                                             preview_option = 'Once', 
                                             number_scan = 1,
                                             acquisition_index = 0, 
                                             cam_triggerType = triggerType, 
                                             image_index_list = image_index_list,
                                             pattern_num_list = pattern_num_list,
                                             cam_gain = 0,
                                             cam_bufferCount = 15,
                                             cam_capt_timeout = 10,
                                             proj_exposure_period = 27084, 
                                             proj_frame_period = 33334,
                                             do_insert_black = True,
                                             preview_image_index = 21,
                                             pprint_proj_status = True,
                                             focus_image_index = 34 ) 
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