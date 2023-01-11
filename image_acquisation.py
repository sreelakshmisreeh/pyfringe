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
    result = lcr.set_pattern_config(num_lut_entries= 1, 
                                    do_repeat = True, 
                                    num_pats_for_trig_out2 = 1, 
                                    num_images = 1)
    result &= lcr.set_exposure_frame_period(proj_exposure_period, 
                                            proj_frame_period)
    if preview_type == 'focus':
        result &= lcr.send_img_lut([image_index],0)
        result &= lcr.send_pattern_lut(trig_type = 0, 
                                       bit_depth = 8, 
                                       led_select = 0b111,
                                       swap_location_list = [0],
                                       image_index_list = [image_index], 
                                       pattern_num_list = [0], 
                                       starting_address = 0,  
                                       do_insert_black = False) 
        
    elif preview_type == 'preview':
        result &= lcr.send_img_lut([image_index],0)
        result &= lcr.send_pattern_lut(trig_type = 0, 
                                       bit_depth = 8, 
                                       led_select = 0b111,
                                       swap_location_list = [0],
                                       image_index_list = [image_index], 
                                       pattern_num_list = [0], 
                                       starting_address = 0,  
                                       do_insert_black = False)  
    result &= lcr.start_pattern_lut_validate()
    # live view   
    if result:
        gspy.deactivate_trigger(cam)   
        cam.BeginAcquisition()
        result &= lcr.pattern_display('start')
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
            cv2.line(img_show_color,(center_x,center_y - delta),(center_x,center_y + delta),(0,255,0),5)
            cv2.line(img_show_color,(center_x - delta,center_y),(center_x + delta,center_y),(0,255,0),5)
            cv2.imshow("press q to quit", img_show_color)    
            key = cv2.waitKey(1)        
            if key == ord("q"):
                result &= lcr.pattern_display('stop')
                break
        cam.EndAcquisition()
        cv2.destroyAllWindows()
    return True

def run_proj_cam_capt(cam, 
                      lcr, 
                      savedir = None,
                      acquisition_index = 0, 
                      image_index_list = None, 
                      pattern_num_list = None, 
                      cam_capt_timeout = 10,
                      proj_exposure_period = 27084, 
                      proj_frame_period = 33334, 
                      do_insert_black = True,
                      pprint_proj_status = True,
                      do_validation = True):
    
    
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
    image_array_list = []    
    # Retrieve, convert, and save image
    gspy.activate_trigger(cam)
    cam.BeginAcquisition()
    
    if triggerSourceSymbolic == "Line0":
        count = 0        
        total_dual_time_start = perf_counter_ns()
        start = perf_counter_ns() 
        if do_validation: 
            #Configure projector
            if image_index_list and pattern_num_list:
                image_LUT_entries, swap_location_list = lcpy.get_image_LUT_swap_location(image_index_list)
                result &= lcr.set_pattern_config(num_lut_entries= len(image_index_list),
                                                 do_repeat = False,  
                                                 num_pats_for_trig_out2 = len(image_index_list),
                                                 num_images = len(image_LUT_entries))
                result &= lcr.set_exposure_frame_period(proj_exposure_period, 
                                                        proj_frame_period)
                # To set new image LUT
                result &= lcr.send_img_lut(image_LUT_entries,0)
                # To set pattern LUT table    
                result &= lcr.send_pattern_lut(trig_type = 0, 
                                               bit_depth = 8, 
                                               led_select = 0b111,
                                               swap_location_list = swap_location_list, 
                                               image_index_list = image_index_list, 
                                               pattern_num_list = pattern_num_list, 
                                               starting_address = 0,
                                               do_insert_black = do_insert_black)
                if pprint_proj_status:# Print all projector current attributes set
                    lcr.pretty_print_status()
                result &=  lcr.start_pattern_lut_validate()
            elif not image_index_list:
                print('\n image_index_list cannot be empty')
                result &= False
            elif not pattern_num_list:
                print('\n pattern_num_list cannot be empty')
                result &= False
        if result:   
            result &= lcr.pattern_display('start') 
            capturing_time_start = perf_counter_ns()     
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
                    image_array_list.append(image_array)
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
            image_capture_time = (total_dual_time_end - capturing_time_start)/1e9
            total_dual_time = (total_dual_time_end - total_dual_time_start)/1e9
            print('image capturing time:%.3f'%image_capture_time)
            print('Total dual device time:%.3f'%total_dual_time)
        else:
            result = False
    
    cam.EndAcquisition()
    gspy.deactivate_trigger(cam)     

    return result, image_array_list

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
                            focus_image_index,
                            do_validation = True):
    result = True
    proj_preview_exp_period = proj_exposure_period + 230
    proj_preview_frame_period = proj_preview_exp_period
    
    if not focus_image_index == None:
        result &= proj_cam_preview(cam, 
                             lcr, 
                             proj_preview_exp_period, 
                             proj_preview_frame_period, 
                             'focus',
                             focus_image_index)
    
    if (number_scan == 1) & (preview_option == 'Once'):
        result &= proj_cam_preview(cam, 
                                   lcr, 
                                   proj_preview_exp_period, 
                                   proj_preview_frame_period,
                                   'preview',
                                   preview_image_index)
        
        ret, n_scanned_image_list = run_proj_cam_capt(cam, 
                                                 lcr, 
                                                 savedir,
                                                 acquisition_index, 
                                                 image_index_list, 
                                                 pattern_num_list, 
                                                 cam_capt_timeout,
                                                 proj_exposure_period, 
                                                 proj_frame_period, 
                                                 do_insert_black,
                                                 pprint_proj_status,
                                                 do_validation)
        
        
    elif (number_scan > 1) & (preview_option == 'Always'):
        initial_acq_index = acquisition_index
        n_scanned_image_list = []
        validation_flag = do_validation
        for i in range(number_scan):
            if i !=0:
                validation_flag = False
            result &= proj_cam_preview(cam, 
                                       lcr, 
                                       proj_preview_exp_period, 
                                       proj_preview_frame_period,
                                       'preview',
                                       preview_image_index)
            
            ret, image_array_list = run_proj_cam_capt(cam, 
                                                      lcr, 
                                                      savedir,
                                                      initial_acq_index, 
                                                      image_index_list, 
                                                      pattern_num_list, 
                                                      cam_capt_timeout,
                                                      proj_exposure_period, 
                                                      proj_frame_period, 
                                                      do_insert_black,
                                                      pprint_proj_status,
                                                      do_validation = validation_flag)
            initial_acq_index +=1
            n_scanned_image_list.append(image_array_list)
            
    elif (number_scan > 1) & (preview_option == 'Once'):
        initial_acq_index = acquisition_index
        n_scanned_image_list = []
        validation_flag = do_validation
        result &= proj_cam_preview(cam, 
                                   lcr, 
                                   proj_preview_exp_period, 
                                   proj_preview_frame_period,
                                   'preview',
                                   preview_image_index)
        for i in range(number_scan):
            if i !=0:
                validation_flag = False
            ret, image_array_list= run_proj_cam_capt(cam, 
                                                     lcr, 
                                                     savedir,
                                                     initial_acq_index, 
                                                     image_index_list, 
                                                     pattern_num_list, 
                                                     cam_capt_timeout,
                                                     proj_exposure_period, 
                                                     proj_frame_period, 
                                                     do_insert_black,
                                                     pprint_proj_status,
                                                     do_validation = validation_flag)
            initial_acq_index +=1
            n_scanned_image_list.append(image_array_list)
    elif (number_scan > 1) & (preview_option == 'Never'):
        initial_acq_index = acquisition_index
        n_scanned_image_list = []
        validation_flag = do_validation
        for i in range(number_scan):
            if i !=0:
                validation_flag = False
            ret, image_array_list= run_proj_cam_capt(cam, 
                                                         lcr, 
                                                         savedir,
                                                         initial_acq_index, 
                                                         image_index_list, 
                                                         pattern_num_list, 
                                                         cam_capt_timeout,
                                                         proj_exposure_period, 
                                                         proj_frame_period, 
                                                         do_insert_black,
                                                         pprint_proj_status,
                                                         do_validation = validation_flag)
            initial_acq_index +=1
            n_scanned_image_list.append(image_array_list)
            
    elif (number_scan == 1) & (preview_option == 'Never'):
       ret, n_scanned_image_list= run_proj_cam_capt(cam, 
                                                    lcr, 
                                                    savedir,
                                                    acquisition_index, 
                                                    image_index_list, 
                                                    pattern_num_list, 
                                                    cam_capt_timeout,
                                                    proj_exposure_period, 
                                                    proj_frame_period, 
                                                    do_insert_black,
                                                    pprint_proj_status,
                                                    do_validation)
    result &= ret
    
    return result, n_scanned_image_list

def run_proj_single_camera(savedir,                          
                           image_index_list,
                           pattern_num_list,
                           cam_gain = 0,
                           cam_bufferCount = 15,
                           cam_capt_timeout = 10,
                           proj_exposure_period = 27084, 
                           proj_frame_period = 33334,
                           do_insert_black = True,
                           preview_image_index = 21,
                           number_scan = 1,
                           acquisition_index = 0,
                           pprint_proj_status = True,
                           preview_option = 'Once',
                           focus_image_index = None):
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
        result, system, cam_list, num_cameras = gspy.sysScan()
        cam = cam_list[0]
        gspy.clearDir(savedir)
        device = usb.core.find(idVendor=0x0451, idProduct=0x6401) #finding the projector usb port
        device.set_configuration()

        lcr = lcpy.dlpc350(device)
        result &= lcr.pattern_display('stop')
        
        # Initialize camera
        cam.Init()
        # config camera
        frameRate = 1e6/proj_frame_period  #proj_frame_period is in μs
        result &= gspy.cam_configuration(cam = cam,
                                         triggerType = 'hardware',
                                         frameRate = frameRate,
                                         exposureTime = proj_exposure_period,
                                         gain = cam_gain,
                                         bufferCount = cam_bufferCount)  
        print('result',result)
        # Acquire images        
        ret, n_scanned_image_list = proj_cam_acquire_images(cam,
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
        result &= ret
        # Deinitialize camera        
        cam.DeInit()
        device.reset()
        del lcr
        del device
        del cam
        cam_list.Clear()
        system.ReleaseInstance() 
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False
    return result, n_scanned_image_list

def main():
  
    image_index_list = np.repeat(np.arange(0,5),3).tolist()
    pattern_num_list = [0,1,2] * len(set(image_index_list))
    savedir = r'C:\Users\kl001\Documents\grasshopper3_python\images'
    result = True
    ret, n_scanned_image_list= run_proj_single_camera(savedir = savedir,
                                     preview_option = 'Always', 
                                     number_scan = 2,
                                     acquisition_index = 0, 
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
    result &= ret
 
    return result ,n_scanned_image_list

if __name__ == '__main__':
    result , n_scanned_image_list = main()
    if result:
        sys.exit(0)
    else:
        sys.exit(1)