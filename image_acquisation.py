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
import glob
from time import perf_counter_ns, sleep
import usb.core
import PySpin
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # needed when openCV and matplotlib used at the same time

"""    
NOTE: Regarding projector, there are four key time durations: 
1) 8-bit pattern frame period; 
2) pattern exposure time; 
3) black fill time; 
4) 24-bit image loading time 

8-bit pattern frame period = pattern exposure time + black fill time

Pattern exposure time is the duration of each pattern projected onto the object's surface and defines the camera triggering period (trigger width). 
No specific requirement for the shortest exposure time while longest exposure time will be determined by the minimum black fill time, hence it should be 
determined last (i.e., after the minimum back fill time and pattern frame period).

The minimum black fill time depends on the larger value of 
1) the DMD pattern loading time (230 us from the TI document) and 
2) the camera sensor readout time (conservatively 6250 us for Grasshopper3 GS3-U3-23S6M-C 163 FPS), hence our system uses 6250 us as the minimum black fill time.

The projector's 8-bit pattern frame period should satisfy the following requirement to accommodate image buffer loading:
(8-bit pattern frame period x 3) >= (worst/longest 24-bit image loading time)

Hence the workflow should be: 
1) the 24-bit image loading time should be characterized for all images, 
2) find the worst case and then add a small time period to it, say 500 us 
3) divide the resulting number by three as the 8-bit pattern frame period
4) exposure time <= (8-bit pattern frame period) - 6250 us, to avoid over exposure, usually a smaller number of preferred. 

Also camera requires certain time to activate its trigger mode, this issue is currently fixed by adding a sleep time after sending the trigger activation command
and before starting the projector. If this is not set the camera may drop some initial frames while switching between preview and acquisition mode.
"""

def proj_cam_preview(cam, 
                     nodemap,
                     s_node_map,
                     lcr, 
                     proj_exposure_period, 
                     proj_frame_period, 
                     led_select,
                     preview_type,
                     image_index,
                     cam_trig_reconfig=True,
                     pprint_status=True):
    """
    Function is used to show a preview of camera projector setting before scanning.There two options :
    focus:  The projector projects the focus adjustment image at the given image index and the camera will be in free run video mode.
            This can be used to adjust the projector and camera focus.
    preview:The projector projects constant image at the given image index and the camera will be in free run video mode.
            This option can be used to adjust camera exposure. If the camera is overexposed(255) those pixels will be flagged red.
    :param cam : camera to acquire images from.
    :param nodemap: camara nodemap.
    :param s_node_map:camera stream nodemap.
    :param lcr : lcr4500 USB projector device.
    :param proj_exposure_period: projector exposure period.
    :param proj_frame_period: projector frame period.
    :param led_select: projector light source color.
    :param preview_type: 'focus' or 'preview'.
    :param image_index: image index on projector flash.
    :param cam_trig_reconfig: switch to reconfigure camera trigger. If consecutively the function is called, it can be set to False.
    :pprint_status: pretty print projector and camera current parameters.
    :type cam: cameraPtr.
    :type nodemap:cNodemapPtr.
    :type s_node_map:cNodemapPtr.
    :type lcr : class instance.
    :type proj_exposure_period : int.
    :type proj_frame_period: int.
    :type led_select: int.
    :type preview_type: str.
    :type image_index: int.
    :type cam_trig_reconfig: bool.
    :type pprint_status: bool.
    :return True if successful, False otherwise. 
    :rtype: bool.
    """     
  
    # set projector configuration
    result = lcr.set_pattern_config(num_lut_entries=1,
                                    do_repeat=True,
                                    num_pats_for_trig_out2=1,
                                    num_images=1)
    result &= lcr.set_exposure_frame_period(exposure_period=proj_exposure_period,
                                            frame_period=proj_frame_period)
    # config camera trigger for preview
    if cam_trig_reconfig:
        result &= gspy.trigger_configuration(nodemap=nodemap,
                                             s_node_map=s_node_map,
                                             triggerType="off",
                                             verbose=pprint_status)
    
    if preview_type == 'focus':
        result &= lcr.send_img_lut([image_index], 0)
        result &= lcr.send_pattern_lut(trig_type=0,
                                       bit_depth=8,
                                       led_select=led_select,
                                       swap_location_list=[0],
                                       image_index_list=[image_index],
                                       pattern_num_list=[0],
                                       starting_address=0,
                                       do_insert_black=False)
        
    elif preview_type == 'preview':
        result &= lcr.send_img_lut([image_index], 0)
        result &= lcr.send_pattern_lut(trig_type=0,
                                       bit_depth=8,
                                       led_select=led_select,
                                       swap_location_list=[0],
                                       image_index_list=[image_index],
                                       pattern_num_list=[0],
                                       starting_address=0,
                                       do_insert_black=False)

    # Print all projector current attributes set
    if pprint_status:
        lcr.pretty_print_status()

    # validate the pattern lut
    result &= lcr.start_pattern_lut_validate()

    # live view
    if result:
        delta_time = 50
        delta_cross = 100
        mean_lst = []
        max_int = 0
        result &= lcr.pattern_display('start')
        cam.BeginAcquisition()
        while True:                
            ret, frame = gspy.capture_image(cam)       
            img_show = cv2.resize(frame, None, fx=0.5, fy=0.5)
            mean_lst.append(img_show)
            if len(mean_lst) == 20:
                mean_intensity = np.mean(np.array(mean_lst),axis=0)
                max_int = np.max(mean_intensity)
                mean_lst = []
            if preview_type == 'preview':
                img_show_color = cv2.cvtColor(img_show, cv2.COLOR_GRAY2BGR)
                img_show_color[img_show == 255] = [0, 0, 255]  # over exposed
            else:
                img_show_color = img_show

            # draw the cross
            center_y = int(img_show.shape[0]/2)
            center_x = int(img_show.shape[1]/2)
            cv2.line(img_show_color, (center_x, center_y - delta_cross), (center_x, center_y + delta_cross), (0, 255, 0), 5)
            cv2.line(img_show_color, (center_x - delta_cross, center_y), (center_x + delta_cross, center_y), (0, 255, 0), 5)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_show_color,'Exposure time:%s'%str(proj_exposure_period),(0,50),font,1,(0,255,255),2)  #text,coordinate,font,size of text,color,thickness of font
            cv2.putText(img_show_color,'Delta:%s'%str(delta_time),(0,100),font,1,(0,255,255),2)
            cv2.putText(img_show_color,'Max intensity:%s'%str(max_int),(0,150),font,1,(0,255,255),2)
            cv2.imshow("press q to quit", img_show_color)
            key = cv2.waitKey(1)
            if key ==ord("+"):
                proj_exposure_period +=delta_time
                result &= gspy.setExposureTime(nodemap, exposureTime=proj_exposure_period)
            elif key == ord("-"):
                proj_exposure_period -=delta_time
                result &= gspy.setExposureTime(nodemap, exposureTime=proj_exposure_period)
            elif key == ord(">"):
                delta_time +=5
            elif key == ord("<"):
                delta_time -=5
            elif key == ord("q"):
                break
        cam.EndAcquisition()
        cv2.destroyAllWindows()
        result &= lcr.pattern_display('stop')
    return result, proj_exposure_period

def run_proj_cam_capt(cam, 
                      nodemap,
                      s_node_map,
                      lcr, 
                      savedir,
                      acquisition_index=0,
                      image_index_list=None,
                      pattern_num_list=None,
                      cam_capt_timeout=10,
                      proj_exposure_period=27084,
                      proj_frame_period=33334,
                      do_insert_black=True,
                      led_select=4,
                      pprint_status=True,
                      do_repeat=False,
                      total_image_number=None,
                      image_section_size=None,
                      save_npy=True,
                      save_jpeg=False):
    
    """
    This function projects and acquires images. Note that projector and camera must be initialized before 
    calling this function. The acquired images will be saved as np array in the given savedir path.
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
    :param led_select: projector light source color.
    :param pprint_status: pretty print projector and camera current parameters.
    :param do_repeat: do repeat projection
    :param total_image_number: total number of images to capture. If None is given, using len(image_index_list).
    :image_section_size: the number of images that are packed into a single npy file. If None is given, using len(image_index_list).
    :param save_npy: Save images as .npy format
    :param save_jpeg: Save images as .jpeg format
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
    :type led_select: int.
    :type pprint_status: bool.
    :type do_repeat: bool.
    :type total_image_number: int.
    :type image_section_size: int
    :type save_npy: bool.
    :type save_jpeg: bool.
    :return result :True if successful, False otherwise. 
    :rtype: bool.
    """
    
    print('*** IMAGE ACQUISITION ***\n')

    # Check if the total number of images is valid
    number_of_patterns = len(image_index_list)
    if total_image_number is None:
        total_image_number = number_of_patterns
    if image_section_size is None:
        image_section_size = number_of_patterns
    if (total_image_number < number_of_patterns) or ((total_image_number % number_of_patterns) != 0):
        print("ERROR: total_image_number is not valid, it must be N x number_of_patterns.")
        return False
    if (not do_repeat) and (total_image_number > number_of_patterns):
        print("WARNING: Pattern sequence running once while the total number of images requested is larger than the number of patterns!")
    # Check if the saving options are valid
    if (not save_npy) and (not save_jpeg):
        print("ERROR: both save_npy and save_jpeg are false, at least one should be True")
        return False

    result = True
    total_dual_time_start = perf_counter_ns()
    # For projector safety, force the projector to stop first
    result &= lcr.pattern_display('stop')
    # Configure projector
    if image_index_list and pattern_num_list:
        image_LUT_entries, swap_location_list = lcpy.get_image_LUT_swap_location(image_index_list)
        result &= lcr.set_pattern_config(num_lut_entries=number_of_patterns,
                                         do_repeat=do_repeat,
                                         num_pats_for_trig_out2=number_of_patterns,
                                         num_images=len(image_LUT_entries))
        result &= lcr.set_exposure_frame_period(exposure_period=proj_exposure_period,
                                                frame_period=proj_frame_period)
        # To set new image LUT
        result &= lcr.send_img_lut(image_LUT_entries, 0)
        # To set pattern LUT table
        result &= lcr.send_pattern_lut(trig_type=0,
                                       bit_depth=8,
                                       led_select=led_select,
                                       swap_location_list=swap_location_list,
                                       image_index_list=image_index_list,
                                       pattern_num_list=pattern_num_list,
                                       starting_address=0,
                                       do_insert_black=do_insert_black)
        if pprint_status:  # Print all projector current attributes set
            lcr.pretty_print_status()
        result &= lcr.start_pattern_lut_validate()
    elif not image_index_list:
        print('\n image_index_list cannot be empty')
        result &= False
    elif not pattern_num_list:
        print('\n pattern_num_list cannot be empty')
        result &= False

    # config camera trigger for image acquisition
    result &= gspy.trigger_configuration(nodemap=nodemap,
                                         s_node_map=s_node_map,
                                         triggerType='hardware')
    if result:
        gspy.activate_trigger(nodemap)
        sleep(0.05)
        cam.BeginAcquisition()
        start = perf_counter_ns()
        count = 0
        image_array_list = []
        result &= lcr.pattern_display('start')
        capturing_time_start = perf_counter_ns()
        while count < total_image_number:
            try:
                if save_npy:
                    return_array = True
                else:
                    return_array = False
                if save_jpeg:
                    save_path_jpeg = os.path.join(savedir,
                                                  'capt_%03d_%06d.tiff' % (acquisition_index, count))
                else:
                    save_path_jpeg = None
                ret, image_array = gspy.capture_image(cam=cam, save_path=save_path_jpeg, return_array=return_array)
            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                ret = False
                image_array = None
                pass
            if ret:
                print("extract successful")
                # save one section when the counter reaches the section size
                if save_npy:
                    image_array_list.append(image_array)
                    if (count % image_section_size) == (image_section_size - 1):
                        section_id = count // image_section_size
                        save_path = os.path.join(savedir, 'capt_%03d_%06d.npy' % (acquisition_index, section_id))
                        np.save(save_path, image_array_list)
                        image_array_list = []
                count += 1
                start = perf_counter_ns()
                print('waiting clock is reset')
            else:
                end = perf_counter_ns()
                waiting_time = (end - start)/1e9
                print('Capture failed, time spent %2.3f s before %2.3f s timeout' % (waiting_time, cam_capt_timeout))
                if waiting_time > cam_capt_timeout:
                    print('timeout is reached, stop capturing image ...')
                    break
        capturing_time_end = perf_counter_ns()
        if do_repeat:
            result &= lcr.pattern_display('stop')
        image_capture_time = (capturing_time_end - capturing_time_start) / 1e9
        print('image capturing time:%.3f' % image_capture_time)

        # save the last section if the number of images is shorter than a section size
        if save_npy:
            if image_array_list:
                print('WARNING: The last image section is shorter with number of images less than %d.' % image_section_size)
                section_id = (count - 1) // image_section_size
                save_path = os.path.join(savedir, 'capt_%03d_%06d.npy' % (acquisition_index, section_id))
                np.save(save_path, image_array_list)
                print('Last section of scanned images saved as %s' % save_path)

        cam.EndAcquisition()
        gspy.deactivate_trigger(nodemap)
        total_dual_time_end = perf_counter_ns()
        total_dual_time = (total_dual_time_end - total_dual_time_start)/1e9
        print('Total dual device time:%.3f' % total_dual_time)
    return result

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
                            cam_black_level,
                            cam_ExposureCompensation,
                            proj_exposure_period, 
                            proj_frame_period,
                            do_insert_black,
                            led_select,
                            preview_image_index,
                            focus_image_index,
                            image_section_size=None,
                            pprint_status=True,
                            save_npy=True,
                            save_jpeg=False):
    """
    Wrapper function combining preview option and object scanning. 
    The projector configuration and camera trigger mode for each is different.
    
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
    :param cam_black_level: camera black level.
    :param cam_ExposureCompensation: camera exposure compensation
    :param proj_exposure_period: projector exposure period in microseconds.
    :param proj_frame_period: projector frame period in microseconds.
    :param do_insert_black: insert black-fill pattern after each pattern. This setting requires 230 us of time before the
                            start of the next pattern.
    :param led_select: projector light source color.
    :param preview_image_index: image to be projected for adjusting camera exposure.
    :param focus_image_index: image to be projected to adjust projector and camera focus. If set to None this will be skipped.
    :param pprint_status: pretty print projector and camera current parameters.
    :param image_section_size: the number of images that are packed into a single npy file. If None is given, using len(image_index_list).
    :param save_npy: Save images as .npy format
    :param save_jpeg: Save images as .jpeg
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
    :type cam_black_level: float
    :type cam_ExposureCompensation: float
    :type proj_exposure_period: int
    :type proj_frame_period: int
    :type do_insert_black: bool
    :type led_select: int
    :type preview_image_index: int/None
    :type focus_image_index: int
    :type pprint_status: bool
    :type image_section_size: int / None
    :type save_npy: bool
    :type save_jpeg: bool
    :return result :True if successful, False otherwise.
    :rtype: bool
    """
    nodemap = cam.GetNodeMap()
    nodemap_tldevice = cam.GetTLDeviceNodeMap()
    s_node_map = cam.GetTLStreamNodeMap()
    gspy.print_device_info(nodemap_tldevice)
    frameRate = 1e6/proj_frame_period  # proj_frame_period is in μs
    
    proj_preview_exp_period = proj_exposure_period 
    proj_preview_frame_period = proj_preview_exp_period
    result = True
    ret = True
    # config camera
    result &= gspy.cam_configuration(nodemap=nodemap,
                                     s_node_map=s_node_map,
                                     frameRate=frameRate,
                                     pgrExposureCompensation=cam_ExposureCompensation,
                                     exposureTime=proj_exposure_period,
                                     gain=cam_gain,
                                     blackLevel=cam_black_level,
                                     bufferCount=cam_bufferCount,
                                     verbose=pprint_status)
    cam_trig_reconfig = True
    if focus_image_index is not None:
        ret,_ = proj_cam_preview(cam=cam,
                                 nodemap=nodemap,
                                 s_node_map=s_node_map,
                                 lcr=lcr,
                                 proj_exposure_period=proj_preview_exp_period,
                                 proj_frame_period=proj_preview_frame_period,
                                 led_select=led_select,
                                 preview_type='focus',
                                 image_index=focus_image_index,
                                 cam_trig_reconfig=True,
                                 pprint_status=pprint_status)
    if (focus_image_index is not None) or (preview_option is not None):
        cam_trig_reconfig = False
    if (number_scan == 1) & ((preview_option == 'Once') or (preview_option == 'Always')):
        do_repeat = False
        total_image_number = len(image_index_list)
        image_section_size = total_image_number
        ret, proj_exposure_period = proj_cam_preview(cam=cam,
                                                     nodemap=nodemap,
                                                     s_node_map=s_node_map,
                                                     lcr=lcr,
                                                     proj_exposure_period=proj_preview_exp_period,
                                                     proj_frame_period=proj_preview_frame_period,
                                                     led_select=led_select,
                                                     preview_type='preview',
                                                     image_index=preview_image_index,
                                                     cam_trig_reconfig=cam_trig_reconfig,
                                                     pprint_status=pprint_status)
        
        ret &= run_proj_cam_capt(cam=cam,
                                 nodemap=nodemap,
                                 s_node_map=s_node_map,
                                 lcr=lcr,
                                 savedir=savedir,
                                 acquisition_index=acquisition_index,
                                 image_index_list=image_index_list,
                                 pattern_num_list=pattern_num_list,
                                 cam_capt_timeout=cam_capt_timeout,
                                 proj_exposure_period=proj_exposure_period,
                                 proj_frame_period=proj_frame_period,
                                 do_insert_black=do_insert_black,
                                 led_select=led_select,
                                 do_repeat=do_repeat,
                                 total_image_number=total_image_number,
                                 image_section_size=image_section_size,
                                 pprint_status=pprint_status,
                                 save_npy=save_npy,
                                 save_jpeg=save_jpeg)
        
    elif (number_scan > 1) & (preview_option == 'Always'):
        # if preview option is Always the projector LUT has to be rewritten hence do_validation must be True
        initial_acq_index = acquisition_index
        do_repeat = False
        total_image_number = len(image_index_list)
        image_section_size = total_image_number
        for i in range(number_scan):
            if i > 0:
                cam_trig_reconfig = True
                proj_preview_exp_period = proj_exposure_period
                proj_preview_frame_period = proj_preview_exp_period
            ret, proj_exposure_period = proj_cam_preview(cam=cam,
                                                         nodemap=nodemap,
                                                         s_node_map=s_node_map,
                                                         lcr=lcr,
                                                         proj_exposure_period=proj_preview_exp_period,
                                                         proj_frame_period=proj_preview_frame_period,
                                                         led_select=led_select,
                                                         preview_type='preview',
                                                         image_index=preview_image_index,
                                                         cam_trig_reconfig=cam_trig_reconfig,
                                                         pprint_status=pprint_status)
            
            ret &= run_proj_cam_capt(cam=cam,
                                     nodemap=nodemap,
                                     s_node_map=s_node_map,
                                     lcr=lcr,
                                     savedir=savedir,
                                     acquisition_index=initial_acq_index,
                                     image_index_list=image_index_list,
                                     pattern_num_list=pattern_num_list,
                                     cam_capt_timeout=cam_capt_timeout,
                                     proj_exposure_period=proj_exposure_period,
                                     proj_frame_period=proj_frame_period,
                                     do_insert_black=do_insert_black,
                                     led_select=led_select,
                                     do_repeat=do_repeat,
                                     total_image_number=total_image_number,
                                     image_section_size=image_section_size,
                                     pprint_status=pprint_status,
                                     save_npy=save_npy,
                                     save_jpeg=save_jpeg)
            initial_acq_index += 1
            
    elif (number_scan > 1) & (preview_option == 'Once'):
        do_repeat = True
        total_image_number = number_scan * len(image_index_list)
        ret, proj_exposure_period = proj_cam_preview(cam=cam,
                                                     nodemap=nodemap,
                                                     s_node_map=s_node_map,
                                                     lcr=lcr,
                                                     proj_exposure_period=proj_preview_exp_period,
                                                     proj_frame_period=proj_preview_frame_period,
                                                     led_select=led_select,
                                                     preview_type='preview',
                                                     image_index=preview_image_index,
                                                     pprint_status=pprint_status)
        ret &= run_proj_cam_capt(cam=cam,
                                 nodemap=nodemap,
                                 s_node_map=s_node_map,
                                 lcr=lcr,
                                 savedir=savedir,
                                 acquisition_index=acquisition_index,
                                 image_index_list=image_index_list,
                                 pattern_num_list=pattern_num_list,
                                 cam_capt_timeout=cam_capt_timeout,
                                 proj_exposure_period=proj_exposure_period,
                                 proj_frame_period=proj_frame_period,
                                 do_insert_black=do_insert_black,
                                 led_select=led_select,
                                 do_repeat=do_repeat,
                                 total_image_number=total_image_number,
                                 image_section_size=image_section_size,
                                 pprint_status=pprint_status,
                                 save_npy=save_npy,
                                 save_jpeg=save_jpeg)
            
    elif preview_option == 'Never':
        if number_scan == 1:
            do_repeat = False
            total_image_number = len(image_index_list)
            image_section_size = total_image_number
        else:
            do_repeat = True
            total_image_number = number_scan * len(image_index_list)
        ret = run_proj_cam_capt(cam=cam,
                                nodemap=nodemap,
                                s_node_map=s_node_map,
                                lcr=lcr,
                                savedir=savedir,
                                acquisition_index=acquisition_index,
                                image_index_list=image_index_list,
                                pattern_num_list=pattern_num_list,
                                cam_capt_timeout=cam_capt_timeout,
                                proj_exposure_period=proj_exposure_period,
                                proj_frame_period=proj_frame_period,
                                do_insert_black=do_insert_black,
                                led_select=led_select,
                                do_repeat=do_repeat,
                                total_image_number=total_image_number,
                                image_section_size=image_section_size,
                                pprint_status=pprint_status,
                                save_npy=save_npy,
                                save_jpeg=save_jpeg)
            
    result &= ret
    
    return result

def run_proj_single_camera(savedir,                          
                           image_index_list,
                           pattern_num_list,
                           cam_gain=0,
                           cam_bufferCount=15,
                           cam_capt_timeout=10,
                           cam_black_level=0,
                           cam_ExposureCompensation=0,
                           proj_exposure_period=27084,
                           proj_frame_period=33334,
                           do_insert_black=True,
                           led_select=4,
                           preview_image_index=21,
                           number_scan=1,
                           acquisition_index=0,
                           preview_option='Once',
                           focus_image_index=None,
                           image_section_size=None,
                           pprint_status=True,
                           save_npy=True,
                           save_jpeg=False,
                           clear_dir=True):
    """
    Initialize and de-initialize projector and camera before and after capture.
    :param savedir: directory to save images.
    :param image_index_list: projector pattern sequence to create and project.
    :param pattern_num_list: pattern number for each pattern in image_index_list.
    :param cam_gain: camera gain
    :param cam_bufferCount:camera buffer count
    :param cam_capt_timeout: camera waiting time in seconds before termination.
    :param cam_black_level: camera black level.
    :param cam_ExposureCompensation: camera exposure compensation
    :param proj_exposure_period: projector exposure period in microseconds.
    :param proj_frame_period: projector frame period in microseconds.
    :param do_insert_black: insert black-fill pattern after each pattern. This setting requires 230 us of time before the
                            start of the next pattern.
    :param led_select: projector light source color.
    :param preview_image_index: image to be projected for adjusting camera exposure.
    :param number_scan: number of times the projector camera system scans the object.
    :param acquisition_index: the index number of the current acquisition.
    :param pprint_status: pretty print projector and camera current parameters.
    :param preview_option: 'Once','Always,'Never'
    :param focus_image_index: image to be projected to adjust projector and camera focus. If set to None this will be skipped.
    :param image_section_size: the number of images that are packed into a single npy file. If None is given, using len(image_index_list).
    :param save_npy: Save images as .npy format
    :param save_jpeg: Save images as .jpeg
    :param clear_dir: Clear given directory
    :type savedir: str
    :type image_index_list: list
    :type pattern_num_list: list
    :type cam_gain: float
    :type cam_bufferCount: int
    :type cam_capt_timeout: float
    :type cam_black_level: float
    :type cam_ExposureCompensation: float
    :type proj_exposure_period: int
    :type proj_frame_period: int
    :type do_insert_black: bool
    :type led_select: int.
    :type preview_image_index: int/None
    :type number_scan: int
    :type acquisition_index: int
    :type pprint_status: bool
    :type preview_option: str
    :type focus_image_index: int / None
    :type image_section_size: int/ None
    :type save_npy: bool
    :type save_jpeg: bool
    :type clear_dir: bool
    :return result: True if successful, False otherwise.
    :rtype :bool
    """
    try:
        result, system, cam_list, num_cameras = gspy.sysScan()
        cam = cam_list[0]
        if clear_dir:
            gspy.clearDir(savedir)
        device = usb.core.find(idVendor=0x0451, idProduct=0x6401)  # find the projector usb port
        device.set_configuration()

        lcr = lcpy.dlpc350(device)
        result &= lcr.pattern_display('stop')
        
        # Initialize camera
        cam.Init()
        print('result', result)
        # Acquire images        
        ret = proj_cam_acquire_images(cam=cam,
                                      lcr=lcr,
                                      savedir=savedir,
                                      preview_option=preview_option,
                                      number_scan=number_scan,
                                      acquisition_index=acquisition_index,
                                      image_index_list=image_index_list,
                                      pattern_num_list=pattern_num_list,
                                      cam_gain=cam_gain,
                                      cam_bufferCount=cam_bufferCount,
                                      cam_capt_timeout=cam_capt_timeout,
                                      cam_black_level=cam_black_level,
                                      cam_ExposureCompensation=cam_ExposureCompensation,
                                      proj_exposure_period=proj_exposure_period,
                                      proj_frame_period=proj_frame_period,
                                      do_insert_black=do_insert_black,
                                      led_select=led_select,
                                      preview_image_index=preview_image_index,
                                      focus_image_index=focus_image_index,
                                      image_section_size=image_section_size,
                                      pprint_status=pprint_status,
                                      save_npy=save_npy,
                                      save_jpeg=save_jpeg)
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
    return result

def gamma_curve(gamma_image_index_list,
                gamma_pattern_num_list,
                savedir,
                cam_width=1920,
                cam_height=1200,
                half_cross_length=100):
    """
    Function to generate gamma curve
    """
    
    camx = int(cam_width/2)
    camy = int(cam_height/2)
   
    result = run_proj_single_camera(savedir=savedir,
                                    preview_option='Once',
                                    number_scan=1,
                                    acquisition_index=0,
                                    image_index_list=gamma_image_index_list,
                                    pattern_num_list=gamma_pattern_num_list,
                                    cam_gain=0,
                                    cam_bufferCount=15,
                                    cam_capt_timeout=10,
                                    cam_black_level=0,
                                    cam_ExposureCompensation=0,
                                    proj_exposure_period=25000,
                                    proj_frame_period=34000,
                                    do_insert_black=True,
                                    led_select=4,
                                    preview_image_index=21,
                                    focus_image_index=None,
                                    pprint_status=True,
                                    save_npy=True,
                                    save_jpeg=False)
    if result:
        n_scanned_image_list = np.load(os.path.join(savedir, 'capt_000_000000.npy'))
        camera_captured = n_scanned_image_list[:, camy - half_cross_length: camy + half_cross_length, camx - half_cross_length: camx + half_cross_length]
        mean_intensity = np.mean(camera_captured.reshape((camera_captured.shape[0], -1)), axis=1)
        x_axis = np.arange(5, 256, 5)
        y = np.polyfit(x_axis, mean_intensity, 1)
        plt.figure(figsize=(16, 9))
        plt.scatter(x_axis, mean_intensity, label='captured mean per frame')
        plt.plot(x_axis, y[0]*x_axis+y[1], label='linear fit', color='r')
        plt.xlabel("Input Intensity", fontsize=20)
        plt.ylabel("Output Intensity", fontsize=20)
        plt.title("Projector gamma curve", fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(fontsize=15)
        plt.show()
        plt.savefig(os.path.join(savedir, 'gamma_curve.png'))
        np.save(os.path.join(savedir, 'gamma_curve.npy'), mean_intensity)
    else:
        print('ERROR: Capture failure')
    return result

def calib_capture(image_index_list,
                  pattern_num_list,
                  savedir,
                  number_scan,
                  acquisition_index):
    """
    Function to capture calibration images.
    """
    result = run_proj_single_camera(savedir=savedir,
                                    preview_option='Always',
                                    number_scan=number_scan,
                                    acquisition_index=acquisition_index,
                                    image_index_list=image_index_list,
                                    pattern_num_list=pattern_num_list,
                                    cam_gain=0,
                                    cam_bufferCount=15,
                                    cam_capt_timeout=10,
                                    cam_black_level=0,
                                    cam_ExposureCompensation=0,
                                    proj_exposure_period=25000,
                                    proj_frame_period=34000,#66668,#33334,
                                    do_insert_black=True,
                                    led_select=4,
                                    preview_image_index=21,
                                    focus_image_index=None,
                                    pprint_status=True,
                                    save_npy=True)
    return result
def meanpixel_var(savedir,
                  image_index,
                  pattern_no,
                  no_images,
                  proj_exposure_period=27000,
                  proj_frame_period=34000,
                  cam_width=1920,
                  cam_height=1200,
                  half_cross_length=100,
                  acquisition_index=0):
    """
    Function to determine additive noise std of camera sensor using constant image (image_index).
    :param savedir : path to save npy file.
    :param image_index: pattern to be projected.
    :param no_images: no of images to be used in the calculation.
    :param acquisition_index: the index number of the current acquisition.
    :param cam_width : camera width
    :param cam_height: camera height
    :half_cross_length: half window size for calculation.
    :type savedir: str
    :type image_index: int
    :type pattern_no: int
    :type no_images: int
    :type acquisition_index: int
    :type cam_width: int
    :type cam_height: int
    :type half_cross_length: int
    :return mean_std_pixel: mean of std of each pixel within the given window.
    :return std_pixel: temporal intensity std map for the window region
    :rtype mean_std_pixel:float
    :rtype std_pixel: array of float
    """
    image_index_list = [image_index]*no_images
    pattern_num_list = [pattern_no]*no_images
    result = run_proj_single_camera(savedir=savedir,
                                    preview_option='Never',
                                    number_scan=1,
                                    acquisition_index=acquisition_index,
                                    image_index_list=image_index_list,
                                    pattern_num_list=pattern_num_list,
                                    cam_gain=0,
                                    cam_bufferCount=15,
                                    cam_capt_timeout=10,
                                    cam_black_level=0,
                                    cam_ExposureCompensation=0,
                                    proj_exposure_period=proj_exposure_period,
                                    proj_frame_period=proj_frame_period,
                                    do_insert_black=True,
                                    led_select=4,
                                    preview_image_index=16,
                                    focus_image_index=None,
                                    image_section_size=None,
                                    pprint_status=True,
                                    save_npy=False,
                                    save_jpeg=True)
    mean_var_pixel = None
    if result:
        path = sorted(glob.glob(os.path.join(savedir,'capt_%03d_*.jpeg'%acquisition_index)),key=lambda x:int(os.path.basename(x)[-11:-5]))
        n_scanned_image_list = np.array([cv2.imread(file,0) for file in path])
        camx = int(cam_width/2)
        camy = int(cam_height/2)
        capt_cropped = n_scanned_image_list[:, camy - half_cross_length: camy + half_cross_length, camx - half_cross_length: camx + half_cross_length]
        var_pixel = np.var(capt_cropped, axis=0)
        mean_var_pixel = np.mean(var_pixel)
        np.save(os.path.join(savedir, 'mean_var_pixel.npy'), mean_var_pixel)
    else:
        print('ERROR: Capture failure')
    return mean_var_pixel, var_pixel

def optimal_frame_rate(image_indices, no_iterations):
    device = usb.core.find(idVendor=0x0451, idProduct=0x6401)  # find the projector usb port
    device.set_configuration()
    lcr = lcpy.dlpc350(device)
    result = lcr.pattern_display('stop')
    max_time_list = []
    for i in range(no_iterations):
        result, time_list_microsec = lcr.image_loading_time(image_indices)
        max_time_list.append(max(time_list_microsec))
    pattern_frame_period = (max(max_time_list) + 500)/3
    pattern_exposure_period = pattern_frame_period - 6250
    print('Approx. 8 bit pattern frame period = %6.3f' % pattern_frame_period)
    print('Approx. 8 bit pattern exposure period = %6.3f' % pattern_exposure_period)
    device.reset()
    del lcr
    del device
    return result

def main():
    """
    Example main function.
    """
    option = input("Please choose:\n1: test\n2: Approx.frame period and exposure time\n3: gamma curve\n4: Camera noise\n5: calibration capture\n6: Reconstruction ")
    result = True
    if option == '1':
        image_index_list = np.repeat(np.array([17,19,21,23,24,25]),3).tolist()
        pattern_num_list = [0, 1, 2] * len(set(image_index_list))
        savedir = r'C:\Users\kl001\Documents\grasshopper3_python\images'
        result &= run_proj_single_camera(savedir=savedir,
                                         preview_option='Always',
                                         number_scan=1,
                                         acquisition_index=0,
                                         image_index_list=image_index_list,
                                         pattern_num_list=pattern_num_list,
                                         cam_gain=0,
                                         cam_bufferCount=15,
                                         cam_capt_timeout=10,
                                         cam_black_level=0,
                                         cam_ExposureCompensation=0,
                                         proj_exposure_period=27000,#27084,
                                         proj_frame_period=34000,#33334,
                                         do_insert_black=True,
                                         led_select=4,
                                         preview_image_index=16,
                                         focus_image_index=29,
                                         image_section_size=None,
                                         pprint_status=True,
                                         save_npy=False,
                                         save_jpeg=True)
    elif option == '2':
        starting_index = int(input("\nEnter starting image index of the sequence:"))
        no_images = int(input("\nEnter number of images in the sequence:"))
        no_iterations = int(input("\nNo. of iterations:"))
        image_indices = np.arange(starting_index, starting_index+no_images).tolist()
        result &= optimal_frame_rate(image_indices, no_iterations)
    elif option == '3':
        gamma_image_index_list = np.repeat(np.arange(0, 17), 3).tolist()
        gamma_pattern_num_list = [0, 1, 2] * len(set(gamma_image_index_list))
        savedir = r'C:\Users\kl001\Documents\pyfringe_test\gamma_images'
        result &= gamma_curve(gamma_image_index_list,
                              gamma_pattern_num_list,
                              savedir,
                              cam_width=1920,
                              cam_height=1200,
                              half_cross_length=100)
    elif option == '4':
        image_index = int(input("\nImage index to be used:")) #image 18; pattern:200, 205, 210
        pattern_no = int(input("\nPattern number:"))
        no_images = int(input("\nNo. of iterations:"))
        acquisition_index=int(input("\nAcquisation index"))
        savedir = r'C:\Users\kl001\Documents\pyfringe_test\mean_pixel_std'
        meanpixel_var(savedir,
                      image_index,
                      pattern_no,
                      no_images,
                      cam_width=1920,
                      cam_height=1200,
                      half_cross_length=100,
                      acquisition_index=acquisition_index)
        
    elif option == '5':
        image_index_list = np.repeat(np.arange(17, 29), 3).tolist()
        pattern_num_list = [0, 1, 2] * len(set(image_index_list))
        savedir = r'C:\Users\kl001\Documents\pyfringe_test\multifreq_calib_images'
        number_scan = int(input("\nEnter number of scans"))
        acquisition_index = int(input("\nEnter acquisition index"))
        result &= calib_capture(image_index_list=image_index_list,
                                pattern_num_list=pattern_num_list,
                                savedir=savedir,
                                number_scan=number_scan,
                                acquisition_index=acquisition_index)
    elif option == '6':
        no_of_levels =input("\nNo. of levels 2,3,4:")
        number_scan = int(input("\nEnter number of scans"))
        if no_of_levels == "2":
            image_index_list = np.repeat([30,35], 3).tolist()
            pattern_num_list = [0, 1, 2] * len(set(image_index_list))
        elif no_of_levels == "3":
            image_index_list = np.repeat(np.arange(30, 35), 3).tolist()
            pattern_num_list = [0, 1, 2] * len(set(image_index_list))
        elif no_of_levels == "4" :
            image_index_list = np.repeat(np.array([17,19,21,23,24,25]),3).tolist()
            pattern_num_list = [0, 1, 2] * len(set(image_index_list))
        savedir = r'C:\Users\kl001\Documents\grasshopper3_python\images'
        #savedir = r"E:\white board4\data20\test_data_30"
        result &= run_proj_single_camera(savedir=savedir,
                                         preview_option='Once',
                                         number_scan=number_scan,
                                         acquisition_index=0,
                                         image_index_list=image_index_list,
                                         pattern_num_list=pattern_num_list,
                                         cam_gain=0,
                                         cam_bufferCount=15,
                                         cam_capt_timeout=10,
                                         cam_black_level=0,
                                         cam_ExposureCompensation=0,
                                         proj_exposure_period=20000,#27084,Check option 2 for recomended value
                                         proj_frame_period=30000,#34000,#33334,
                                         do_insert_black=True,
                                         led_select=4,
                                         preview_image_index=16,
                                         focus_image_index=29,
                                         image_section_size=None,
                                         pprint_status=True,
                                         save_npy=False,
                                         save_jpeg=True)
    
    return result 


if __name__ == '__main__':
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
