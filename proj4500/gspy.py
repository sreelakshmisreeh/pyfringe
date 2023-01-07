# coding=utf-8

import os
import PySpin
import sys
import cv2
from time import perf_counter_ns

def capture_image(cam):
    """
    Once the camera engine has been activated, this function is used to Extract 
    one image from the buffering memery and save it into a numpy array.
    Note that the camera mode must be set, e.g., "Continuous", and the 
    cam.BeginAcquisition() must be called before calling this function.
    
    Parameters
    ----------
    cam : camera object from PySpin
        camera object to be used
        
    Returns
    -------
    result : bool
        True: success
        False: failed
    image_averaged: uint8
        output image (numpy ndarray).

    """    
    image_result = cam.GetNextImage(1000)  
        
    # Ensure image completion
    if image_result.IsIncomplete():
        print('Image incomplete with image status %d ...' % image_result.GetImageStatus(), end="\r")
        return False, None
    else:        
        image_array = image_result.GetNDArray()          
        # Release image
        image_result.Release()       
        return True, image_array

def cam_configuration(cam, 
                      triggerType,
                      frameRate=30,
                      exposureTime=27084,
                      gain=0,
                      bufferCount=15):
    """
    Configurate the camera. Note that the camera must be initialized before calling
    this function, i.e., cam.Init() must be called before calling this function.

    Parameters
    ----------
    cam : camera object
        The camera to be configureated.
    triggerType : str
        Must be one of {"software", "hardware"}.

    Returns
    -------
    result : bool
        operation result.

    """
    nodemap = cam.GetNodeMap()
    nodemap_tldevice = cam.GetTLDeviceNodeMap()
    s_node_map = cam.GetTLStreamNodeMap()
    
    print('\n=================== Camera status before configuration ==========================\n')    
    print_device_info(nodemap_tldevice)    
    get_IEnumeration_node_current_entry_name(nodemap, 'AcquisitionMode')    
    get_IEnumeration_node_current_entry_name(s_node_map, 'StreamBufferHandlingMode')
    get_IEnumeration_node_current_entry_name(s_node_map, 'StreamBufferCountMode')
    get_IInteger_node_current_val(s_node_map, 'StreamBufferCountManual')
    get_IEnumeration_node_current_entry_name(nodemap, 'TriggerMode')
    get_IEnumeration_node_current_entry_name(nodemap, 'TriggerSelector')
    get_IEnumeration_node_current_entry_name(nodemap, 'TriggerActivation')
    get_IEnumeration_node_current_entry_name(nodemap, 'TriggerSource')
    get_IEnumeration_node_current_entry_name(nodemap, 'AcquisitionFrameRateAuto')
    get_IBoolean_node_current_val(nodemap, 'AcquisitionFrameRateEnabled')
    get_IFloat_node_current_val(nodemap, 'AcquisitionFrameRate')
    get_IEnumeration_node_current_entry_name(nodemap, 'ExposureAuto')
    get_IEnumeration_node_current_entry_name(nodemap, 'ExposureMode')    
    get_IFloat_node_current_val(nodemap, 'ExposureTime')
    get_IEnumeration_node_current_entry_name(nodemap, 'GainAuto')
    get_IFloat_node_current_val(nodemap, 'Gain')
    get_IBoolean_node_current_val(nodemap, 'TriggerDelayEnabled')
    get_IFloat_node_current_val(nodemap, 'TriggerDelay')
    
    print('\n=================== Config camera ==============================================\n')
    result = True    
    result &= setAcqusitionMode(nodemap, AcqusitionModeName='Continuous')    
    result &= setFrameRate(nodemap, frameRate=frameRate)
    result &= setExposureTime(nodemap, exposureTime=exposureTime)
    result &= setGain(nodemap, gain=gain)
    result &= configure_trigger(nodemap, triggerType=triggerType)
    
    if triggerType == 'software':
        result &= setStreamBufferHandlingMode(s_node_map, StreamBufferHandlingModeName='NewestOnly')        
    
    if triggerType == 'hardware':
        result &= setStreamBufferHandlingMode(s_node_map, StreamBufferHandlingModeName='OldestFirst')
        result &= setBufferCount(s_node_map, bufferCount=bufferCount)
    
    print('\n=================== Camera status after configuration ==========================\n')    
    get_IEnumeration_node_current_entry_name(nodemap, 'AcquisitionMode')    
    get_IEnumeration_node_current_entry_name(s_node_map, 'StreamBufferHandlingMode')
    get_IEnumeration_node_current_entry_name(s_node_map, 'StreamBufferCountMode')
    get_IInteger_node_current_val(s_node_map, 'StreamBufferCountManual')
    get_IEnumeration_node_current_entry_name(nodemap, 'TriggerMode')
    get_IEnumeration_node_current_entry_name(nodemap, 'TriggerSelector')
    get_IEnumeration_node_current_entry_name(nodemap, 'TriggerActivation')
    get_IEnumeration_node_current_entry_name(nodemap, 'TriggerSource')
    get_IEnumeration_node_current_entry_name(nodemap, 'AcquisitionFrameRateAuto')
    get_IBoolean_node_current_val(nodemap, 'AcquisitionFrameRateEnabled')
    get_IFloat_node_current_val(nodemap, 'AcquisitionFrameRate')
    get_IEnumeration_node_current_entry_name(nodemap, 'ExposureAuto')
    get_IEnumeration_node_current_entry_name(nodemap, 'ExposureMode')    
    get_IFloat_node_current_val(nodemap, 'ExposureTime')
    get_IEnumeration_node_current_entry_name(nodemap, 'GainAuto')
    get_IFloat_node_current_val(nodemap, 'Gain')
    get_IBoolean_node_current_val(nodemap, 'TriggerDelayEnabled')
    get_IFloat_node_current_val(nodemap, 'TriggerDelay')
    return result

def acquire_images(cam, acquisition_index, savedir, triggerType):
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
        ret, frame = capture_image(cam)       
        img_show = cv2.resize(frame, None, fx=0.5, fy=0.5)
        cv2.imshow("press q to quit", img_show)    
        key = cv2.waitKey(1)        
        if key == ord("q"):
            break
    cam.EndAcquisition()
    cv2.destroyAllWindows()
    
    # Retrieve, convert, and save image
    activate_trigger(cam)
    cam.BeginAcquisition()        
    
    if triggerType == "software":
        start = perf_counter_ns()            
        cam.TriggerSoftware.Execute()               
        ret, image_array = capture_image(cam=cam)                
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
        while count < 15:
            try:
                ret, image_array = capture_image(cam=cam)
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
    deactivate_trigger(cam)        

    return result

def print_device_info(nodemap_tldevice):
    """
    This function prints the device information of the camera from the transport
    layer.

    :param nodemap_tldevice: Transport layer device nodemap.
    :type nodemap: INodeMap
    :returns: True if successful, False otherwise.
    :rtype: bool
    """

    print('\n*** DEVICE INFORMATION ***\n')
    try:
        result = True
        node_device_information = PySpin.CCategoryPtr(nodemap_tldevice.GetNode('DeviceInformation'))
        display_name_node_device_information = node_device_information.GetDisplayName()
        print(display_name_node_device_information)
        if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                print('%s: %s' % (node_feature.GetName(),
                                  node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))
            print('\n')
        else:
            print('Device control information not available.')
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False
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
        result &= cam_configuration(cam, triggerType)        
        # Acquire images        
        result &= acquire_images(cam, acquisition_index, savedir, triggerType)
        # Deinitialize camera        
        cam.DeInit()
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False
    return result

def sysScan():
    """
    Scan the system and find all available cameras

    Returns
    -------
    result : bool
        Operation result, True or False.
    system : system object
        Camera system object
    cam_list : list
        Camera list.
    num_cameras : TYPE
        Number of cameras.

    """
    result = True

    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()

    # Get current library version
    version = system.GetLibraryVersion()
    print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()
    
    # Get the total number of cameras
    num_cameras = cam_list.GetSize()
    print('Number of cameras detected: %d' % num_cameras)
    
    return result, system, cam_list, num_cameras

def clearDir(targetDir):
    """
    Clear the directory
    
    Parameters
    ----------
    targetDir : str
        targetDir to be cleared.

    Returns
    -------
    None.

    """
    if len(os.listdir(targetDir)) != 0:
        for f in os.listdir(targetDir):
            os.remove(os.path.join(targetDir, f))
        print('Directory is cleared!')
    else:
        print('The target directory is empty! No image file needs to be removed')    

def get_IEnumeration_node_current_entry_name(nodemap, nodename):
    node = PySpin.CEnumerationPtr(nodemap.GetNode(nodename))
    node_int_val = node.GetIntValue()
    node_entry = node.GetEntry(node_int_val)
    node_entry_name = node_entry.GetSymbolic()
    node_description = node.GetDescription()
    node_entries = node.GetEntries() # node_entries is a list of INode instances    
    print('%s: %s' % (nodename, node_entry_name))    
    print(node_description)    
    print('All entries are listed below:')
    for i, entry in enumerate(node_entries):        
        entry_name = PySpin.CEnumEntryPtr(entry).GetSymbolic()        
        print('%d: %s' % (i, entry_name))    
    print('\n')
    return node_entry_name

def get_IInteger_node_current_val(nodemap, nodename):
    node = PySpin.CIntegerPtr(nodemap.GetNode(nodename))
    node_val = node.GetValue()
    node_val_max = node.GetMax()
    node_val_min = node.GetMin()
    node_description = node.GetDescription()    
    print('%s: %d' % (nodename, node_val))
    print(node_description)
    print('Max = %d' % node_val_max)
    print('Min = %d' % node_val_min)
    print('\n')    
    return node_val

def get_IFloat_node_current_val(nodemap, nodename):
    node = PySpin.CFloatPtr(nodemap.GetNode(nodename))
    node_val = node.GetValue()
    node_val_max = node.GetMax()
    node_val_min = node.GetMin()
    node_unit = node.GetUnit()
    print('%s: %f' % (nodename, node_val))    
    print('Max = %f' % node_val_max)
    print('Min = %f' % node_val_min)
    print('Unit: ', node_unit)
    print('\n')
    return node_val

def get_IString_node_current_str(nodemap, nodename):
    node = PySpin.CStringPtr(nodemap.GetNode(nodename))
    node_str = node.GetValue()
    node_description = node.GetDescription()
    print('%s: %s' % (nodename, node_str))
    print(node_description, '\n')
    return node_str

def get_IBoolean_node_current_val(nodemap, nodename):
    node = PySpin.CBooleanPtr(nodemap.GetNode(nodename))
    node_val = node.GetValue()
    node_description = node.GetDescription()
    print('%s: %s' % (nodename, node_val))
    print(node_description, '\n')
    return node_val

def enableFrameRateSetting(nodemap):
    # Turn off "AcquisitionFrameRateAuto"    
    acqFrameRateAuto = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionFrameRateAuto"))
    if (not PySpin.IsAvailable(acqFrameRateAuto)) or (not PySpin.IsWritable(acqFrameRateAuto)): 
        print('Unable to retrieve AcquisitionFrameRateAuto. Aborting...')
        return False
    acqFrameRateAutoOff = acqFrameRateAuto.GetEntryByName('Off')
    if not PySpin.IsAvailable(acqFrameRateAutoOff) or not PySpin.IsReadable(acqFrameRateAutoOff):
        print('Unable to set Buffer Handling mode (Value retrieval). Aborting...')
        return False    
    acqFrameRateAuto.SetIntValue(acqFrameRateAutoOff.GetValue()) # setting to Off
    print('Set AcquisitionFrameRateAuto to off')    
    # Turn on "AcquisitionFrameRateEnabled"
    acqframeRateEnable = PySpin.CBooleanPtr(nodemap.GetNode("AcquisitionFrameRateEnabled"))    
    if (not PySpin.IsAvailable(acqframeRateEnable)) or (not PySpin.IsWritable(acqframeRateEnable)): 
        print('Unable to retrieve AcqFrameRateEnable. Aborting...')
        return False
    acqframeRateEnable.SetValue(True)
    print('Set AcquisitionFrameRateEnabled to True')
    return True

def setFrameRate(nodemap, frameRate):
    # First enable framerate setting
    enableFrameRateSetting(nodemap)
    # frame rate should be a float number. Get the node and check availability   
    ptrAcquisitionFramerate = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
    if not PySpin.IsAvailable(ptrAcquisitionFramerate) and not PySpin.IsReadable(ptrAcquisitionFramerate):
        print('Unable to retrieve AcquisitionFrameRate. Aborting...')
        return False
    # Set framerate value
    ptrAcquisitionFramerate.SetValue(frameRate)
    print('AcquisitionFrameRate set to %3.3f Hz' % frameRate)      
    return True

def enableExposureAuto(nodemap):
    # Get the node "ExposureAuto" and convert it to Enumeration class
    ptrExposureAuto = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureAuto"))
    if (not PySpin.IsAvailable(ptrExposureAuto)) or (not PySpin.IsWritable(ptrExposureAuto)): 
        print('Unable to retrieve ExposureAuto. Aborting...')
        return False
    # Get the "Continuous" entry
    ExposureAuto_on = ptrExposureAuto.GetEntryByName("Continuous")
    if not PySpin.IsAvailable(ExposureAuto_on) or not PySpin.IsReadable(ExposureAuto_on):
        print('Unable to set ExposureAuto mode to Continuous. Aborting...')
        return False
    # set the "Continuous" entry to ExposureAuto
    ptrExposureAuto.SetIntValue(ExposureAuto_on.GetValue())
    print('ExposureAuto mode is set to "Continuous"')
    return True

def disableExposureAuto(nodemap):
    # Get the node "ExposureAuto" and convert it to Enumeration class
    ptrExposureAuto = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureAuto"))
    if (not PySpin.IsAvailable(ptrExposureAuto)) or (not PySpin.IsWritable(ptrExposureAuto)): 
        print('Unable to retrieve ExposureAuto. Aborting...')
        return False
    # Get the "Off" entry
    ExposureAuto_off = ptrExposureAuto.GetEntryByName("Off")
    if not PySpin.IsAvailable(ExposureAuto_off) or not PySpin.IsReadable(ExposureAuto_off):
        print('Unable to set ExposureAuto mode to Off. Aborting...')
        return False
    # set the "Off" entry to ExposureAuto
    ptrExposureAuto.SetIntValue(ExposureAuto_off.GetValue())
    print('ExposureAuto mode is set to "off"')
    return True

def setExposureMode(nodemap, exposureModeToSet):
    """
    Sets the operation mode of the exposure (shutter). Toggles the Trigger
    Selector. Timed = Exposure Start; Trigger Width = Exposure Active

    Parameters
    ----------
    nodemap : INodeMap
        Camara nodemap.
    exposureModeToSet : Str
        ExposureModeEnums, must be one of {"Timed", "TriggerWidth"}

    Returns
    -------
    bool
        Operation result.

    """
    # Get the node "ExposureMode" and check if it is available and writable
    ptrExposureMode = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureMode"))
    if (not PySpin.IsAvailable(ptrExposureMode)) or (not PySpin.IsWritable(ptrExposureMode)): 
        print('Unable to retrieve ExposureMode. Aborting...')
        return False
    # Get the Entry to be set and check if it is available and writable
    ExposureMode_selected = ptrExposureMode.GetEntryByName(exposureModeToSet)
    if not PySpin.IsAvailable(ExposureMode_selected) or not PySpin.IsReadable(ExposureMode_selected):
        print('Unable to set ExposureMode to %s. Aborting...'%exposureModeToSet)
        return False    
    # Set the entry to the node
    ptrExposureMode.SetIntValue(ExposureMode_selected.GetValue())
    print('ExposureMode is set to %s...'%exposureModeToSet)
    return True

def setTriggerMode(nodemap, TriggerModeToSet):
    """
    Controls whether or not the selected trigger is active

    Parameters
    ----------
    nodemap : INodeMap
        Camara nodemap.
    TriggerModeToSet : str
        TriggerModeEnums, must be one of {"Off", "On"}

    Returns
    -------
    bool
        Operation result.

    """
    # Get the node "TriggerMode" and check if it is available and writable
    ptrTriggerMode = PySpin.CEnumerationPtr(nodemap.GetNode("TriggerMode"))
    if (not PySpin.IsAvailable(ptrTriggerMode)) or (not PySpin.IsWritable(ptrTriggerMode)): 
        print('Unable to retrieve TriggerMode. Aborting...')
        return False
    # Get the Entry to be set and check if it is available and writable
    TriggerMode_selected = ptrTriggerMode.GetEntryByName(TriggerModeToSet)
    if not PySpin.IsAvailable(TriggerMode_selected) or not PySpin.IsReadable(TriggerMode_selected):
        print('Unable to set TriggerMode to %s. Aborting...'%TriggerModeToSet)
        return False 
    ptrTriggerMode.SetIntValue(TriggerMode_selected.GetValue())
    print('TriggerMode is set to %s...'%TriggerModeToSet)
    return True
    
def setTriggerActivation(nodemap, TriggerActivationToSet):
    """
    Specifies the activation mode of the trigger

    Parameters
    ----------
    nodemap : INodeMap
        Camara nodemap.
    TriggerActivationToSet : str
       TriggerActivationEnums, must be one of {"FallingEdge", "RisingEdge"}

    Returns
    -------
    bool
        Operation result.

    """
    # Get the node "TriggerActivation" and check if it is available and writable
    ptrTriggerActivation = PySpin.CEnumerationPtr(nodemap.GetNode("TriggerActivation"))
    if (not PySpin.IsAvailable(ptrTriggerActivation)) or (not PySpin.IsWritable(ptrTriggerActivation)): 
        print('Unable to retrieve TriggerActivation. Aborting...')
        return False
    # Get the Entry to be set and check if it is available and writable
    TriggerActivation_selected = ptrTriggerActivation.GetEntryByName(TriggerActivationToSet)
    if not PySpin.IsAvailable(TriggerActivation_selected) or not PySpin.IsReadable(TriggerActivation_selected):
        print('Unable to set TriggerActivation to %s. Aborting...'%TriggerActivationToSet)
        return False
    ptrTriggerActivation.SetIntValue(TriggerActivation_selected.GetValue())
    print('TriggerActivation is set to %s...'%TriggerActivationToSet)   
    return True

def setTriggerOverlap(nodemap, TriggerOverlapToSet):
    """
    Overlapped Exposure Readout Trigger

    Parameters
    ----------
    nodemap : INodeMap
        Camara nodemap.
    TriggerOverlapToSet : str
        TriggerOverlapEnums, must be one of {"Off", "ReadOut"}

    Returns
    -------
    bool
        Operation result.

    """
    # Get the node "TriggerOverlap" and check if it is available and writable
    ptrTriggerOverlap = PySpin.CEnumerationPtr(nodemap.GetNode("TriggerOverlap"))
    if (not PySpin.IsAvailable(ptrTriggerOverlap)) or (not PySpin.IsWritable(ptrTriggerOverlap)): 
        print('Unable to retrieve TriggerOverlap. Aborting...')
        return False
    # Get the Entry to be set and check if it is available and writable
    TriggerOverlap_selected = ptrTriggerOverlap.GetEntryByName(TriggerOverlapToSet)
    if not PySpin.IsAvailable(TriggerOverlap_selected) or not PySpin.IsReadable(TriggerOverlap_selected):
        print('Unable to set TriggerOverlap to %s. Aborting...'%TriggerOverlapToSet)
        return False
    ptrTriggerOverlap.SetIntValue(TriggerOverlap_selected.GetValue())
    print('TriggerOverlap is set to %s..'%TriggerOverlapToSet)   
    return True

def setTriggerSelector(nodemap, TriggerSelectorToSet):
    """
    Selects the type of trigger to configure. Derived from Exposure Mode.

    Parameters
    ----------
    nodemap : INodeMap
        Camara nodemap.
    TriggerSelectorToSet : str
        TriggerSelectorEnums, must be one of {"FrameStart", "ExposureActive"}.

    Returns
    -------
    bool
        Operation result.

    """
    # Get the node "TriggerOverlap" and check if it is available and writable
    ptrTriggerSelector = PySpin.CEnumerationPtr(nodemap.GetNode("TriggerSelector"))
    if (not PySpin.IsAvailable(ptrTriggerSelector)) or (not PySpin.IsWritable(ptrTriggerSelector)): 
        print('Unable to retrieve TriggerSelector. Aborting...')
        return False
    # Get the Entry to be set and check if it is available and writable
    TriggerSelector_selected = ptrTriggerSelector.GetEntryByName(TriggerSelectorToSet)
    if not PySpin.IsAvailable(TriggerSelector_selected) or not PySpin.IsReadable(TriggerSelector_selected):
        print('Unable to set TriggerSelector to %s. Aborting...'%TriggerSelectorToSet)
        return False
    ptrTriggerSelector.SetIntValue(TriggerSelector_selected.GetValue())
    print('TriggerSelector is set to %s...'%TriggerSelectorToSet)   
    return True

def setTriggerSource(nodemap, TriggerSourceToSet):
    """
    Specifies the internal signal or physical input line to use as the trigger source.

    Parameters
    ----------
    nodemap : INodeMap
        Camara nodemap.
    TriggerSourceToSet : str
        TriggerSourceEnums, must be one of {"Software", "Line0", "Line1", "Line2", "Line3"}.

    Returns
    -------
    bool
        Operation result.

    """
    # Get the node "TriggerSource" and check if it is available and writable
    ptrTriggerSource = PySpin.CEnumerationPtr(nodemap.GetNode("TriggerSource"))
    if (not PySpin.IsAvailable(ptrTriggerSource)) or (not PySpin.IsWritable(ptrTriggerSource)): 
        print('Unable to retrieve TriggerSource. Aborting...')
        return False
    # Get the Entry to be set and check if it is available and writable
    TriggerSource_selected = ptrTriggerSource.GetEntryByName(TriggerSourceToSet)
    if not PySpin.IsAvailable(TriggerSource_selected) or not PySpin.IsReadable(TriggerSource_selected):
        print('Unable to set TriggerSource to %s. Aborting...'%TriggerSourceToSet)
        return False
    ptrTriggerSource.SetIntValue(TriggerSource_selected.GetValue())
    print('TriggerSource is set to %s...'%TriggerSourceToSet)   
    return True

def setExposureTime(nodemap, exposureTime=None):
    # First disable the ExposureAuto
    disableExposureAuto(nodemap)
    # Get the node "ExposureTime" and check if it is available and writable
    ptrExposureTime = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
    if not PySpin.IsAvailable(ptrExposureTime) and not PySpin.IsReadable(ptrExposureTime):
        print('Unable to retrieve Exposure Time. Aborting...')
        return False
    # Ensure desired exposure time does not exceed the maximum
    exposureTimeMax = ptrExposureTime.GetMax()
    if exposureTime is None:
        exposureTime = exposureTimeMax
    else:
        if exposureTime > exposureTimeMax:
            exposureTime = exposureTimeMax
    # Set the exposure time
    ptrExposureTime.SetValue(exposureTime)
    print('Exposure Time set to %5.6f microseconds'%exposureTime)      
    return True

def setAcqusitionMode(nodemap, AcqusitionModeName):
    """
    Explicitely set AcqusitionMode

    Parameters
    ----------
    nodemap : camera nodemap
        camera nodemap.
    AcqusitionModeName : str
        must be one from the three: Continuous, SingleFrame, MultiFrame.

    Returns
    -------
    bool
        result.

    """
    #  Retrieve enumeration node from nodemap

    # # In order to access the node entries, they have to be casted to a pointer type (CEnumerationPtr here)
    node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
    if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
        print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
        return False
    # Retrieve entry node from enumeration node
    node_acquisition_mode_selected = node_acquisition_mode.GetEntryByName(AcqusitionModeName)
    if not PySpin.IsAvailable(node_acquisition_mode_selected) or not PySpin.IsReadable(node_acquisition_mode_selected):
        print('Unable to set acquisition mode to %s. Aborting...' % node_acquisition_mode_selected)
        return False
    # Set integer value from entry node as new value of enumeration node
    node_acquisition_mode.SetIntValue(node_acquisition_mode_selected.GetValue())
    print('Acquisition mode set to %s...'%AcqusitionModeName)  
    return True

def setStreamBufferHandlingMode(s_node_map, StreamBufferHandlingModeName):
    """
    Explicitely set StreamBufferHandlingModeName

    Parameters
    ----------
    s_node_map : camera node
        camera node.
    StreamBufferHandlingModeName : String
        must be one from the four: OldestFirst, OldestFirstOverwrite, NewestOnly, NewestFirst.

    Returns
    -------
    bool
        result

    """
    handlingMode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferHandlingMode'))
    if not PySpin.IsAvailable(handlingMode) or not PySpin.IsWritable(handlingMode):
        print('Unable to set Buffer Handling mode (node retrieval). Aborting...')
        return False    
    handlingModeSelected = handlingMode.GetEntryByName(StreamBufferHandlingModeName)
    if not PySpin.IsAvailable(handlingModeSelected) or not PySpin.IsReadable(handlingModeSelected):
        print('Unable to set Buffer Handling mode (Value retrieval). Aborting...')
        return False
    handlingMode.SetIntValue(handlingModeSelected.GetValue())
    print('Buffer Handling Mode set to %s...'%StreamBufferHandlingModeName)
    return True

def setBufferCount(s_node_map, bufferCount):
    # Retrieve and modify Stream Buffer Count
    buffer_count = PySpin.CIntegerPtr(s_node_map.GetNode('StreamBufferCountManual'))
    if not PySpin.IsAvailable(buffer_count) or not PySpin.IsWritable(buffer_count):
        print('Unable to set Buffer Count (Integer node retrieval). Aborting...')
        return False
    buffer_count.SetValue(bufferCount)
    print('Buffer count now set to: %d'%buffer_count.GetValue())
    return True
    
def disableGainAuto(nodemap):    
    gainAuto = PySpin.CEnumerationPtr(nodemap.GetNode("GainAuto"))
    if (not PySpin.IsAvailable(gainAuto)) or (not PySpin.IsWritable(gainAuto)): 
        print('Unable to retrieve GainAuto. Aborting...')
        return False
    gainAutoOff = gainAuto.GetEntryByName('Off')
    if not PySpin.IsAvailable(gainAutoOff) or not PySpin.IsReadable(gainAutoOff):
        print('Unable to set GainAuto to off (Value retrieval). Aborting...')
        return False
    # setting "Off" for the Gain auto
    gainAuto.SetIntValue(gainAutoOff.GetValue()) # setting to Off
    print('Set GainAuto to off')
    return True

def setGain(nodemap, gain):
    # First disable gainAuto
    disableGainAuto(nodemap)
    # Get the node "Gain" and check the availability
    gainValue = PySpin.CFloatPtr(nodemap.GetNode("Gain"))
    if (not PySpin.IsAvailable(gainValue)) or (not PySpin.IsWritable(gainValue)): 
        print('Unable to retrieve Gain. Aborting...')
        return False
    # Set the gain value
    gainValue.SetValue(gain)
    print('Set Gain to %2.3f'%gain)
    return True

def configure_trigger(nodemap, triggerType):
    """
    This function configures the camera to use a trigger. First, trigger mode is
    ensured to be off in order to select the trigger source.

     :param cam: Camera to configure trigger for.
     :type cam: CameraPtr
     :param triggerType: Trigger type, 'software' or 'hardware' or 'off'
     :return: True if successful, False otherwise.
     :rtype: bool
    """

    print('\n*** CONFIGURING TRIGGER ***\n')
    print('Note that if the application / user software triggers faster than frame time, the trigger may be dropped / skipped by the camera.\n')
    print('If several frames are needed per trigger, a more reliable alternative for such case, is to use the multi-frame mode.\n\n')

    if triggerType == 'software':
        print('Software trigger is chosen...')
    elif triggerType == 'hardware':
        print('Hardware trigger is chosen...')
    elif triggerType == 'off':
        print('Disable trigger mode...') 

    try:
        result = True

        # Ensure trigger mode off
        # The trigger must be disabled in order to configure whether the source
        # is software or hardware.
        setTriggerMode(nodemap, "Off")
                
        if  triggerType == 'off':            
            return True
        
        if triggerType == 'software':
            setTriggerSource(nodemap, "Software")            
            setExposureMode(nodemap, "Timed")
            setTriggerSelector(nodemap, "FrameStart")
            setTriggerActivation(nodemap, "FallingEdge")           
            
        if triggerType == 'hardware':
            setTriggerSource(nodemap, "Line0")                       
            setExposureMode(nodemap, "TriggerWidth")
            setTriggerSelector(nodemap, "ExposureActive")
            setTriggerActivation(nodemap, "FallingEdge")
            
    except PySpin.SpinnakerException as ex:
        print('Error: %s'%ex)
        return False
    return result

def activate_trigger(cam):
    nodemap = cam.GetNodeMap()
    setTriggerMode(nodemap, "On")    
    # setTriggerOverlap(nodemap, "ReadOut")    
    return True

def deactivate_trigger(cam):
    nodemap = cam.GetNodeMap()
    setTriggerMode(nodemap, "Off")
    return True    

def main():
    triggerType = "hardware"
    result, system, cam_list, num_cameras = sysScan()
    
    if result:
        # Run example on each camera
        savedir = r'C:\Users\kl001\Documents\grasshopper3_python\images'
        clearDir(savedir)
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
