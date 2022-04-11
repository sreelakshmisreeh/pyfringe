#!/usr/bin/env python
# coding: utf-8



import numpy as np
import nstep_fringe as nstep
import cv2




def projcam_calib_img(no_pose,limit,N,pitch,width,height,total_step_height,kernel_v,kernel_h):
    calib_images=np.load('calib_images.npy') #check path
    unwrap_v_lst=[]
    unwrap_h_lst=[]
    white_lst=[]
    for x in range (0,no_pose):    
    
        cos_v_int8,mod1=nstep.mask_img(calib_images[x][0:N],limit,N)
        cos_h_int8,mod2=nstep.mask_img(calib_images[x][N:2*N],limit,N)
        step_v_int8,mod3=nstep.mask_img(calib_images[x][2*N:3*N],limit,N)
        step_h_int8,mod4=nstep.mask_img(calib_images[x][3*N:4*N],limit,N)

        orig_img=(np.sum(cos_h_int8,axis=0)/N)+mod2
        #filtered unwrapped phase maps
        fil_unwrap_v,fil_unwrap_h=nstep.temp_unwrap(cos_v_int8,cos_h_int8,step_v_int8,step_h_int8,
                                                    N,pitch,width,height,
                                                    total_step_height,kernel_v,kernel_h)
        unwrap_v_lst.append(fil_unwrap_v)
        unwrap_h_lst.append(fil_unwrap_h)
        white_lst.append(orig_img)
    return(unwrap_v_lst,unwrap_h_lst,white_lst) 


def camera_calib(objp,white_lst,bobmin,bobmax):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    blobParams = cv2.SimpleBlobDetector_Params()

    # Filter by Area.
    blobParams.filterByArea = True
    blobParams.minArea = bobmin#2000
    blobParams.maxArea = bobmax#100000
    
    blobDetector = cv2.SimpleBlobDetector_create(blobParams)
    objpoints = [] # 3d point in real world space
    cam_imgpoints = [] # 2d points in image plane.
    found = 0
    
    cv2.startWindowThread()
    for x in (white_lst):
        white_gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        keypoints = blobDetector.detect(white_gray) # Detect blobs.
      
        # Draw detected blobs as green circles. This helps cv2.findCirclesGrid() .
        im_with_keypoints = cv2.drawKeypoints(x, keypoints, np.array([]), (0,255,0), 
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                             )
        im_with_keypoints_gray = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findCirclesGrid(im_with_keypoints_gray, (4,11), None, 
                                           flags = cv2.CALIB_CB_ASYMMETRIC_GRID+cv2.CALIB_CB_CLUSTERING,
                                          blobDetector=blobDetector)# Find the circle grid
        if ret == True:
            objpoints.append(objp)  # Certainly, every loop objp is the same, in 3D.
            
            center = cv2.cornerSubPix(im_with_keypoints_gray, corners, (11,11), (-1,-1), criteria)    # Refines the corner locations.
            cam_imgpoints.append(center)
            # Draw and display the corners.
            im_with_keypoints = cv2.drawChessboardCorners(x, (4,11), center, ret)#4x11 circles
            found += 1
        cv2.imshow("img", im_with_keypoints) # display
        cv2.waitKey(1000)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cam_ret, cam_mtx, cam_dist, cam_rvecs, cam_tvecs = cv2.calibrateCamera(objpoints, 
                                                                           cam_imgpoints, white_gray.shape[::-1], 
                                                                           None, None)
    tot_error=0
    for i in range(len(objpoints)):
        cam_img2,_=cv2.projectPoints(objpoints[i],cam_rvecs[i],cam_tvecs[i],cam_mtx,cam_dist )
        error=cv2.norm(cam_imgpoints[i],cam_img2,cv2.NORM_L2)/len(cam_img2)
        tot_error+=error
    r_error=tot_error/len(objpoints)
    print("Camera reprojection error:",r_error)
    return(r_error,objpoints,cam_imgpoints,cam_mtx, cam_dist, cam_rvecs, cam_tvecs)



def proj_calib(objp,white_lst,unwrap_v_lst,unwrap_h_lst,centers,pitch,width,height):
    objpoints = [] # 3d point in real world space
    proj_imgpoints = []
    count_lst=[]
    count=0
    for x in range(0,len(unwrap_v_lst)):
        unwrap_v=unwrap_v_lst[x]
        unwrap_h=unwrap_h_lst[x]
        orig_gray=white_lst[x]
        center=centers[x]
        u=[unwrap_v[i[1],i[0]]*pitch/(2*np.pi) for i in center]
        v=[unwrap_h[i[1],i[0]]*pitch/(2*np.pi) for i in center]
        coordi=np.column_stack((u,v)).reshape(44,1,2).astype(np.float32)
        p_img=nstep.proj_img(unwrap_v,unwrap_h,orig_gray,pitch,width,height)
        
        if(~np.isnan(coordi).any()):
            objpoints.append(objp)
            proj_imgpoints.append(coordi)
            proj_keypoints = cv2.drawChessboardCorners(p_img.astype('uint8'), (4,11), coordi, True)
            cv2.imshow("proj", proj_keypoints) # display
            cv2.waitKey(1000)
            count_lst.append(count)
        count+=1

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
    proj_ret, proj_mtx, proj_dist, proj_rvecs, proj_tvecs = cv2.calibrateCamera(objpoints, 
                                                                           proj_imgpoints,(height,width), 
                                                                           None, None)
    tot_error=0
    for i in range(len(objpoints)):
        proj_img2,_=cv2.projectPoints(objpoints[i],proj_rvecs[i],proj_tvecs[i],proj_mtx,proj_dist )
        error=cv2.norm(proj_imgpoints[i],proj_img2,cv2.NORM_L2)/len(proj_img2)
        tot_error+=error
    r_error=tot_error/len(objpoints)
    print("Projector reprojection error:",r_error)
    tot_count=np.arange(0,len(unwrap_v_lst))
    count_lst=np.array(count_lst)
    missing=np.setdiff1d(np.union1d(tot_count, count_lst), np.intersect1d(tot_count, count_lst))
    return(r_error,objpoints,proj_imgpoints,proj_mtx, proj_dist, proj_rvecs, proj_tvecs,missing)


def asymmetric_world_points():
        objp = []
        for i in range(11):
            for j in range(4):
                x = i/2*72
                if i%2 == 0:
                    y = j*72
                else:
                    y = (j + 0.5)*72
                objp.append((x,y,0))
        #objp = np.hstack((objp,np.zeros((44*3,1)))).astype(np.float32)
        return(np.array(objp).astype('float32'))
