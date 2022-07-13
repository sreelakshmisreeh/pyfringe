import numpy as np
import cv2
from time import sleep, perf_counter_ns
import os
import FringeAcquisition as fa
import nstep_fringe as nstep

def capture_calibration_images(num_pose, starting_poseID, savedir):
    """    
    This program is used to capture calibration images. The images are saved into savedir.
    

    Parameters
    ----------
    num_pose = type:int
        number of poses to be taken.
    savedir = type: string. Directory path to save captured images

    Returns
    -------
    None.

    """
    SLEEP_TIME = 0.07
    SLEEP_TIME_INIT = 0.5
    
    # set up the camera
    result, system, cam_list, num_cameras = fa.sysScan()
    cam = cam_list[0]
    cam.Init()
    
    nodemap = cam.GetNodeMap()
    nodemap_tldevice = cam.GetTLDeviceNodeMap()
    s_node_map = cam.GetTLStreamNodeMap()

    print('=================== Camera status before configuration ==========================')
    
    fa.print_device_info(nodemap_tldevice)    
    fa.get_IEnumeration_node_current_entry_name(nodemap, 'AcquisitionMode')    
    fa.get_IEnumeration_node_current_entry_name(s_node_map, 'StreamBufferHandlingMode')    
    fa.get_IEnumeration_node_current_entry_name(s_node_map, 'StreamBufferCountMode')
    fa.get_IInteger_node_current_val(s_node_map, 'StreamBufferCountManual')
    fa.get_IEnumeration_node_current_entry_name(nodemap, 'TriggerMode')
    fa.get_IEnumeration_node_current_entry_name(nodemap, 'TriggerSelector')
    fa.get_IEnumeration_node_current_entry_name(nodemap, 'TriggerActivation')
    fa.get_IEnumeration_node_current_entry_name(nodemap, 'TriggerSource')
    fa.get_IEnumeration_node_current_entry_name(nodemap, 'ExposureAuto')
    fa.get_IEnumeration_node_current_entry_name(nodemap, 'ExposureMode')
    fa.get_IFloat_node_current_val(nodemap, 'ExposureTime')
    fa.get_IEnumeration_node_current_entry_name(nodemap, 'GainAuto')
    
    fa.configure_trigger(cam, trigerType='off')
    fa.enableFrameRateSetting(nodemap)
    fa.setFrameRate(nodemap, frameRate=30)
    fa.disableGainAuto(nodemap)
    fa.setGain(nodemap, gain_val=10)
    fa.setStreamBufferHandlingMode(s_node_map, StreamBufferHandlingModeName='NewestOnly')
    
    print('=================== Camera status after configuration ==========================')
    
    fa.get_IFloat_node_current_val(nodemap, 'AcquisitionFrameRate')
    fa.get_IFloat_node_current_val(nodemap, 'ExposureTime')
    fa.get_IFloat_node_current_val(nodemap, 'Gain')
    fa.get_IEnumeration_node_current_entry_name(s_node_map, 'StreamBufferHandlingMode')
    fa.get_IInteger_node_current_val(s_node_map, 'StreamBufferCountManual')    
    
    # read calib_fringes.npy. This file contains both vertical and horizontal fringe patterns based on type of unwrapping method
    #fringe_path = os.path.join(savedir,'calib_fringes.npy')
    fringes=np.load(savedir)
    
    for x in range(0, num_pose):        
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
        # Initialting projector window
        fa.configure_trigger(cam, trigerType='software')        
        cv2.startWindowThread()
        cv2.namedWindow('proj',cv2.WINDOW_NORMAL)
        cv2.moveWindow('proj',1920,0)
        cv2.setWindowProperty('proj',cv2.WND_PROP_FULLSCREEN,1)
        # Start pattern projection and capture for a pose
        cam.BeginAcquisition()        
        start = perf_counter_ns()
        for i,img in enumerate(fringes):
            cv2.imshow('proj',img)
            cv2.waitKey(1)
            if i == 0:
                sleep(SLEEP_TIME_INIT)
            else:
                sleep(SLEEP_TIME)
            cam.TriggerSoftware.Execute()            
            ret, image_array = fa.capture_image(cam)            
            poseID = x + starting_poseID
            save_path = os.path.join(savedir, "capt%d_%d.jpg"%(poseID, i))            
            if ret:                    
                cv2.imwrite(save_path, image_array)
                print('Image saved at %s' % save_path)
            else:
                print('Capture fail')        
        end = perf_counter_ns()
        t = (end - start)/1e9
        print('time spent: %2.3f s' % t)
        cam.EndAcquisition()
        cv2.destroyAllWindows()
        fa.configure_trigger(cam, trigerType='off')
    cam.DeInit()
    del cam
    cam_list.Clear()
    system.ReleaseInstance()

def main():    
    savedir = r'C:\Users\kl001\Documents\grasshopper3_python\calib_images'    
    proj_width = 800; proj_height = 1280
    inte_rang = [50,255]    
    pitch_list = [1375, 275, 55, 11]# [1875, 375, 75, 15]
    N_list = [3, 3, 3, 9]
    phi0 = 0
    num_pose = 10
    starting_poseID = 10
    type_unwrap =  'multifreq'
    
    if starting_poseID == 0:        
        fa.clearDir(savedir)
        phase_lst, freq_delta_deck = nstep.calib_generate(proj_width, proj_height, type_unwrap, N_list, pitch_list, inte_rang, phi0, savedir)
        up_savedir = os.path.join(savedir, '{}_fringes.npy'.format(type_unwrap))
    capture_calibration_images(num_pose, starting_poseID,up_savedir )
    
if __name__ == '__main__':
    main()
    

    