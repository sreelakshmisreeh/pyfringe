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
                     nodemap,
                     s_node_map,
                     lcr, 
                     proj_exposure_period, 
                     proj_frame_period, 
                     preview_type,
                     image_index):
    '''
    Function is used to show a preview of camera projector setting before scanning.There two options :
    focus:  The projector projects the focus adjustment image at the given image index and the camera will be in free run vedio mode. 
            This can be used to adjust the projector and camera focus.
    preview:The projector projects constant image at the given image index and the camera will be in free run vedio mode.
            This option can be used to udjust camera exposure. If the camera is overexposed(255) those pixels will be flaged red.   
    :param cam : camera to acquire images from.
    :param nodemap: camara nodemap.
    :param s_node_map:camera stream nodemap.
    :param lcr : lcr4500 USB projector device.
    :param proj_exposure_period: projector exposure period.
    :param proj_frame_period: projector frame period.
    :param preview_type: 'focus' or 'preview'.
    :param image_index: image index on projector flash.
    :type cam: cameraPtr.
    :type nodemap:cNodemapPtr.
    :type s_node_map:cNodemapPtr.
    :type lcr : class instance.
    :type proj_exposure_period : int.
    :type proj_frame_period: int.
    :type preview_type: str.
    :type image_index: int.
    :return True if successful, False otherwise. 
    :rtype: bool
    '''
    
    delta = 100      
  
    #set projector configuration
    result = lcr.set_pattern_config(num_lut_entries= 1, 
                                    do_repeat = True, 
                                    num_pats_for_trig_out2 = 1, 
                                    num_images = 1)
    result &= lcr.set_exposure_frame_period(proj_exposure_period, 
                                            proj_frame_period)
    # config camera trigger for preview
    result &= gspy.trigger_configuration(nodemap=nodemap,
                                         s_node_map=s_node_map,
                                         triggerType="off")
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
                      nodemap,
                      s_node_map,
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
    This function projects and acquires images. Note that projector and camera must be initialized before 
    calling this function. The acquired will be returned as a list and if the savedir path is given the images will be saved.
    do_validation option can be used as a switch if multiple scans with same patterns are done to bypass reconfiguring projector LUT.

    :param cam: camera to acquire images from.
    :param nodemap: camara nodemap.
    :param s_node_map:camera stream nodemap.
    :param lcr: lcr4500 USB projector device.
    :param savedir: directory to save images. 
    :param acquisition_index: the index number of the current acquisition.
    :param image_index_list: projector pattern sequence to create and project.
    :param pattern_num_list: pattern number for each pattern in image_index_list.
    :param cam_capt_timeout: camera waiting time in seconds before termination.
    :param proj_exposure_period: projector exposure period in microseconds.
    :param proj_frame_period : projector frame period in microseconds. 
    :param do_insert_black: insert black-fill pattern after each pattern. This setting requires 230 us of time before the
                            start of the next pattern.
    :pprint_proj_status: pretty print projector current parameters.
    :do_validation : do validation of projector LUT before projection and capture. 
                    Warning: for each new pattern sequence this must be True to change the projector LUT.
    :type cam: CameraPtr
    :type nodemap:cNodemapPtr.
    :type s_node_map:cNodemapPtr.
    :type lcr : class instance.
    :type savedir: str.
    :type acquisition_index: int.
    :type image_index_list: list.
    :type pattern_num_list: list.
    :type cam_capt_timeout: float. 
    :type proj_exposure_period : int.
    :type proj_frame_period: int.
    :type do_insert_black: bool.
    :type pprint_proj_status:bool.
    :type do_validation:bool.
    :return result :True if successful, False otherwise. 
    :return image_array_list: list of image arrays captured.
    :rtype: bool
    :rtype: list
    """
    
    print('*** IMAGE ACQUISITION ***\n')

    result = True  
    image_array_list = []    
    # Retrieve, convert, and save image
    # config trigger for image acquasition
    result &= gspy.trigger_configuration(nodemap=nodemap,
                                    s_node_map=s_node_map,
                                    triggerType='hardware')
    gspy.activate_trigger(nodemap)
    cam.BeginAcquisition()
    count = 0        
    total_dual_time_start = perf_counter_ns()
    start = perf_counter_ns() 
    result &= lcr.pattern_display('stop')
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
                if savedir is not None:
                    filename = 'capt%d_%d.jpg' %(acquisition_index,count)
                    save_path = os.path.join(savedir, filename)
                    cv2.imwrite(save_path, image_array)
                    print('Image saved at %s' % save_path)
                image_array_list.append(image_array)
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
    gspy.deactivate_trigger(nodemap)   

    return result, image_array_list

def proj_cam_acquire_images(cam,
                            lcr, 
                            savedir,
                            preview_option, 
                            number_scan,
                            acquisition_index,  
                            image_index_list,
                            pattern_num_list, 
                            cam_gain,
                            cam_bufferCount,
                            cam_capt_timeout,
                            proj_exposure_period, 
                            proj_frame_period,
                            do_insert_black,
                            preview_image_index,
                            focus_image_index,
                            pprint_proj_status,
                            do_validation = True):
    '''
    Wrapper function combining preview option and object scanning. 
    The projector configuration and camera trigger mode for each is diffrent.
    
    :param cam: camera to acquire images from.
    :param lcr: lcr4500 USB projector device.
    :param savedir: directory to save images.
    :param preview_option: 'Once','Always,'Never'
    :param number_scan: number of times the projector camera system scans the object.
    :param acquisition_index: the index number of the current acquisition.
    :param image_index_list: projector pattern sequence to create and project.
    :param pattern_num_list: pattern number for each pattern in image_index_list.
    :param cam_gain: camera gain
    :param cam_bufferCount: camera buffer count
    :param cam_capt_timeout: camera waiting time in seconds before termination.
    :param proj_exposure_period: projector exposure period in microseconds.
    :param proj_frame_period: projector frame period in microseconds.
    :param do_insert_black: insert black-fill pattern after each pattern. This setting requires 230 us of time before the
                            start of the next pattern.
    :param preview_image_index: image to be projected for adjusting camera exposure.
    :param focus_image_index: image to be projected to adjust projector and camera focus. If set to None this will be skipped.
    :param pprint_proj_status: pretty print projector current parameters.
    :param do_validation: do validation of projector LUT before projection and capture.
                          Warning: for each new pattern sequence this must be True to change the projector LUT.
    :type cam: CameraPtr
    :type lcr: class instance.
    :type savedir: str
    :type preview_option: str
    :type number_scan: int
    :type acquisition_index: int
    :type image_index_list: list
    :type pattern_num_list: list
    :type cam_gain: float
    :type cam_bufferCount: int
    :type cam_capt_timeout: float
    :type proj_exposure_period: int
    :type proj_frame_period: int
    :type do_insert_black: bool
    :type preview_image_index: int
    :type focus_image_index: int
    :type pprint_proj_status: bool
    :type do_validation: bool
    :return result :True if successful, False otherwise.
    :return n_scanned_image_list: list of image arrays captured from all scans.
    :rtype: bool
    :rtype: list
    '''
    nodemap = cam.GetNodeMap()
    nodemap_tldevice = cam.GetTLDeviceNodeMap()
    s_node_map = cam.GetTLStreamNodeMap()
    gspy.print_device_info(nodemap_tldevice)
    frameRate = 1e6/proj_frame_period  #proj_frame_period is in μs
    
    proj_preview_exp_period = proj_exposure_period + 230
    proj_preview_frame_period = proj_preview_exp_period
    result = True
    #config camera
    result &= gspy.cam_configuration(nodemap=nodemap,
                                     s_node_map=s_node_map,
                                     frameRate=frameRate,
                                     exposureTime=proj_exposure_period,
                                     gain=cam_gain,
                                     bufferCount=cam_bufferCount)
    
    if not focus_image_index == None:
        result &= proj_cam_preview(cam, 
                                   nodemap,
                                   s_node_map,
                                   lcr, 
                                   proj_preview_exp_period, 
                                   proj_preview_frame_period, 
                                   'focus',
                                   focus_image_index)
    
    if (number_scan == 1) & ((preview_option == 'Once') or (preview_option == 'Always')):
        result &= proj_cam_preview(cam,
                                   nodemap,
                                   s_node_map,
                                   lcr, 
                                   proj_preview_exp_period, 
                                   proj_preview_frame_period,
                                   'preview',
                                   preview_image_index)
        
        ret, n_scanned_image_list = run_proj_cam_capt(cam,
                                                      nodemap,
                                                      s_node_map,
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
        # if preview option is Always the projector LUT has to be rewritten hence do_validation must be True
        initial_acq_index = acquisition_index
        n_scanned_image_list = []
        for i in range(number_scan):
            result &= proj_cam_preview(cam, 
                                       nodemap,
                                       s_node_map,
                                       lcr, 
                                       proj_preview_exp_period, 
                                       proj_preview_frame_period,
                                       'preview',
                                       preview_image_index)
            
            ret, image_array_list = run_proj_cam_capt(cam, 
                                                      nodemap,
                                                      s_node_map,
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
                                                      do_validation = do_validation)
            initial_acq_index +=1
            n_scanned_image_list.append(image_array_list)
            
    elif (number_scan > 1) & (preview_option == 'Once'):
        initial_acq_index = acquisition_index
        n_scanned_image_list = []
        validation_flag = do_validation
        result &= proj_cam_preview(cam, 
                                   nodemap,
                                   s_node_map,
                                   lcr, 
                                   proj_preview_exp_period, 
                                   proj_preview_frame_period,
                                   'preview',
                                   preview_image_index)
        for i in range(number_scan):
            if i !=0:
                validation_flag = False
            ret, image_array_list= run_proj_cam_capt(cam, 
                                                     nodemap,
                                                     s_node_map,
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
    elif (preview_option == 'Never'):
        initial_acq_index = acquisition_index
        n_scanned_image_list = []
        validation_flag = do_validation
        for i in range(number_scan):
            if i !=0:
                validation_flag = False
            ret, image_array_list= run_proj_cam_capt(cam,
                                                     nodemap,
                                                     s_node_map,
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
    Initialize and de-initialize projector and camera before and after capture.
    :param savedir: directory to save images.
    :param image_index_list: projector pattern sequence to create and project.
    :param pattern_num_list: pattern number for each pattern in image_index_list.
    :param cam_gain: camera gain
    :param cam_bufferCount:camera buffer count
    :param cam_capt_timeout: camera waiting time in seconds before termination.
    :param proj_exposure_period: projector exposure period in microseconds.
    :param proj_frame_period: projector frame period in microseconds.
    :param do_insert_black: insert black-fill pattern after each pattern. This setting requires 230 us of time before the
                            start of the next pattern.
    :param preview_image_index: image to be projected for adjusting camera exposure.
    :param number_scan: number of times the projector camera system scans the object.
    :param acquisition_index: the index number of the current acquisition.
    :param pprint_proj_status: pretty print projector current parameters.
    :param preview_option: 'Once','Always,'Never'
    :param focus_image_index: image to be projected to adjust projector and camera focus. If set to None this will be skipped.
    :type savedir: str
    :type image_index_list: list
    :type pattern_num_list: list
    :type cam_gain: float
    :type cam_bufferCount: int
    :type cam_capt_timeout: float
    :type proj_exposure_period: int
    :type proj_frame_period: int
    :type do_insert_black: bool
    :type preview_image_index: int
    :type number_scan: int
    :type acquisition_index: int
    :type pprint_proj_status: bool
    :type preview_option: bool
    :type focus_image_index: int
    :return result: True if successful, False otherwise.
    :return n_scanned_image_list: list of image arrays captured from all scans.
    :rtype :bool
    :rtype: list
    """
    try:
        result = True
        result, system, cam_list, num_cameras = gspy.sysScan()
        cam = cam_list[0]
        if savedir is not None:
            gspy.clearDir(savedir)
        device = usb.core.find(idVendor=0x0451, idProduct=0x6401) #finding the projector usb port
        device.set_configuration()

        lcr = lcpy.dlpc350(device)
        result &= lcr.pattern_display('stop')
        
        # Initialize camera
        cam.Init()
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
                                                            cam_gain,
                                                            cam_bufferCount,
                                                            cam_capt_timeout,
                                                            proj_exposure_period, 
                                                            proj_frame_period,
                                                            do_insert_black,
                                                            preview_image_index,
                                                            focus_image_index,
                                                            pprint_proj_status)
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
    '''
    Example main function.
    '''
  
    image_index_list = np.repeat(np.arange(0,5),3).tolist()
    pattern_num_list = [0,1,2] * len(set(image_index_list))
    savedir = r'C:\Users\kl001\Documents\grasshopper3_python\images'
    result = True
    ret, n_scanned_image_list= run_proj_single_camera(savedir=savedir,
                                                      preview_option='Once',
                                                      number_scan=1,
                                                      acquisition_index=0,
                                                      image_index_list=image_index_list,
                                                      pattern_num_list=pattern_num_list,
                                                      cam_gain=0,
                                                      cam_bufferCount=15,
                                                      cam_capt_timeout=10,
                                                      proj_exposure_period=27084,
                                                      proj_frame_period=33334,
                                                      do_insert_black=True,
                                                      preview_image_index=21,
                                                      pprint_proj_status=True,
                                                      focus_image_index=34 )
    result &= ret
 
    return result ,n_scanned_image_list

if __name__ == '__main__':
    result , n_scanned_image_list = main()
    if result:
        sys.exit(0)
    else:
        sys.exit(1)