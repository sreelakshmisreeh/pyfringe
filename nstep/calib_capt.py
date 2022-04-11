#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:21:34 2022

@author: Sreelakshmi
"""

import io
import time
import picamera
import cv2
import pickle

def outputs(N,fringes):
    stream = io.BytesIO()
    cv2.startWindowThread()
    cv2.namedWindow('proj', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('proj',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    
    for i in range(4*N):  # N each of cos_v,cos_h,step_v,step_h
        cv2.imshow('proj',fringes[i])
        # This returns the stream for the camera to capture to    
        yield stream
        # Once the capture is complete,
        # Reset the stream for the next capture
        stream.seek(0)
        stream.truncate()
    cv2.destroyAllWindows()
        
def calib_capture(no_pose,N,fringes):  #no of poses
    #fringes=np.load('fringes.npy')   #check correct path name
    tot_img_lst=[]
    for i in range(0,no_pose):    
        with picamera.PiCamera() as camera:
            camera.resolution = (4056, 3040)
            camera.framerate = 80
            time.sleep(2)
            start = time.time()
            img_stream_list=outputs(N,fringes)
            camera.capture_sequence(img_stream_list, 'jpeg', use_video_port=True)
            finish = time.time()
            print('Captured 4*N images at %.2ffps' % (4*N / (finish - start)))
            tot_img_lst.append(img_stream_list)
            filename='pose%d'%i
            outfile=open(filename,'wb')
            pickle.dump(img_stream_list,outfile)
            outfile.close()
        
    return()
    