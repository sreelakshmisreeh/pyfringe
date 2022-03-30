import cv2
import os
import numpy as np
import scipy.ndimage

#3-step phase shift algorithm 

#p=pitch=no.of pixels per period,
#i0=dc component,i1=half of the peak-to-valley intensity modulation,
#width,height=projector
#inte_rang= projector intensity range
#direc=direction of fringes

def cos_func(width,height,inte_rang,pitch,direc):
    i1=(inte_rang[1]-inte_rang[0])/2
    i0=i1+inte_rang[0]
    if(direc=='v'):
        ar=np.arange(0,width)
        ar_array=np.ones((height,1))*ar
    elif(direc=='h'):
        ar=np.arange(0,height)
        ar_array=np.ones((width,1))*ar
        ar_array=np.rot90(ar_array,3)
    phi=(-np.pi)+(2*np.pi*(ar_array/pitch))
    inte1=i0+i1*np.cos(phi-(2*np.pi/3))
    inte2=i0+i1*np.cos(phi) 
    inte3=i0+i1*np.cos(phi+(2*np.pi)/3) 
    return(inte1,inte2,inte3)



def step_func(width,height,inte_rang,pitch,direc,total_step_hieght):
    i1=(inte_rang[1]-inte_rang[0])/2
    i0=i1+inte_rang[0]
    if(direc=='v'):
        n_fring=width/pitch
        ar=np.arange(0,width)
        ar_array=np.ones((height,1))*ar
    elif(direc=='h'):
        n_fring=height/pitch
        ar=np.arange(0,height)
        ar_array=np.ones((width,1))*ar
        ar_array=np.rot90(ar_array,3)
    phi_s=(-(total_step_hieght/2)*np.pi)+(np.floor(ar_array/pitch)*(total_step_hieght*np.pi/(n_fring-1)))
    inte1=i0+i1*np.cos(phi_s-(2*np.pi/3))
    inte2=i0+i1*np.cos(phi_s) 
    inte3=i0+i1*np.cos(phi_s+(2*np.pi)/3) 
    return(inte1,inte2,inte3)


#Wrap phase calculation
def phase_cal(images):
    i1_org=images[0].astype('float')    
    i2_org=images[1].astype('float')
    i3_org=images[2].astype('float')
    #wrapped phase
    y=(i1_org-i3_org)*np.sqrt(3)
    x=(2*i2_org-i1_org-i3_org)
    ph=np.arctan2(y,x)# wraped phase;  
    return(ph)

#Temporal unwrapping
def unwrap_cal(step_wrap,cos_wrap,width,height,pitch,total_step_hieght):
    if(direc=='v'):
        n_fring=width/pitch
    elif(direc=='h'):
        n_fring=height/pitch
    k=np.round((n_fring-1)*(step_wrap+(np.pi*total_step_hieght/2))/(total_step_hieght*np.pi))
    cos_unwrap=(2*np.pi*k)+cos_wrap 
    return(cos_unwrap)

#Median filter
def filt(image,kernel,direc):
    dup_img=image.copy()
    if (direc=='v'):
        k=(1,kernel)
    elif(direc=='h'):
        k=(kernel,1)
    med_fil=scipy.ndimage.median_filter(dup_img,k)
    k_array=(dup_img-med_fil)/(2*np.pi)
    fil_img=dup_img-(k_array*2*np.pi)
    return(fil_img)

#Removing trend

#Calculation of coefficients
def fit_trend(filter_img,x_grid,y_grid):
    filter_img_flat=filter_img.flatten()[:,np.newaxis]
    x_row=x_grid.flatten()
    y_row=y_grid.flatten()
    one_row=np.ones(len(x_row))
    xy_array=np.array([one_row,x_row,y_row]).T
    coeff=np.linalg.inv(xy_array.T@xy_array)@(xy_array).T@filter_img_flat

    return(coeff)

def trend(x_grid,y_grid,coeff):
    x_row=x_grid.flatten()
    y_row=y_grid.flatten()
    one_row=np.ones(len(x_row))
    xy_arr=np.array([one_row,x_row,y_row]).T
    phi_col=xy_arr@coeff
    return(phi_col.reshape((x_grid.shape)))


