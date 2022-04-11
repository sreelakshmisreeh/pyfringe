#!/usr/bin/env python
# coding: utf-8


import numpy as np
import scipy.ndimage
import cv2

def cos_func(width,height,inte_rang,pitch,direc,N):
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
    delta=2*np.pi*np.arange(1,N+1)/N 
    one_block=np.ones((N,height,width))
    delta_deck=np.einsum('ijk,i->ijk',one_block,delta)
    inte=i0+i1*np.cos(phi+delta_deck) 
    return(inte)


def step_func(width,height,inte_rang,pitch,direc,total_step_hieght,N):
    i1=(inte_rang[1]-inte_rang[0])/2
    i0=i1+inte_rang[0]
    if(direc=='v'):
        n_fring=np.ceil(width/pitch)
        ar=np.arange(0,width)
        ar_array=np.ones((height,1))*ar
    elif(direc=='h'):
        n_fring=np.ceil(height/pitch)
        ar=np.arange(0,height)
        ar_array=np.ones((width,1))*ar
        ar_array=np.rot90(ar_array,3)
    phi_s=(-(total_step_hieght/2 )*np.pi)+(np.floor(ar_array/pitch)*(total_step_hieght*np.pi/(n_fring-1)))
    delta=2*np.pi*np.arange(1,N+1)/N 
    one_block=np.ones((N,height,width))
    delta_deck=np.einsum('ijk,i->ijk',one_block,delta)
    inte=i0+i1*np.cos(phi_s+delta_deck) 
    return(inte)

def mask_img(images,limit,N):
    images=images.astype(np.float32)
    delta=2*np.pi*np.arange(1,N+1)/N
    one_block=np.ones(images.shape)
    delta_deck=np.einsum('ijk,i->ijk',one_block,delta)
    sin_lst=(np.sum(images*np.sin(delta_deck), axis=0))**2
    cos_lst=(np.sum(images*np.cos(delta_deck), axis=0))**2
    modulation=2*np.sqrt(sin_lst+cos_lst)/N
    #mask=np.full(modulation.shape,False)
    #mask[modulation<limit]=True
    #mask=np.repeat(mask[np.newaxis,:,:],N,axis=0)
    #images[mask]=np.nan
    mask=np.full(modulation.shape,True)
    mask[modulation>limit]=False
    masked_img=np.ma.masked_array(images,np.repeat(mask[np.newaxis,:,:],N,axis=0))
    return(masked_img,modulation)

#Wrap phase calculation
def phase_cal(images,N):
    delta=2*np.pi*np.arange(1,N+1)/N 
    one_block=np.ones(images.shape)
    delta_deck=np.einsum('ijk,i->ijk',one_block,delta)
    sin_lst=(np.sum(images*np.sin(delta_deck),axis=0))
    cos_lst=(np.sum(images*np.cos(delta_deck),axis=0))
    #wrapped phase
    ph=-np.arctan2(sin_lst,cos_lst)# wraped phase;  
    return(ph)

#temporal phase unwrapping
def unwrap_cal(step_wrap,cos_wrap,pitch,width,height,total_step_hieght,direc):
    if(direc=='v'):
        n_fring=np.ceil(width/pitch)
    elif(direc=='h'):
        n_fring=np.ceil(height/pitch)
    k=np.round((n_fring-1)*(step_wrap+(np.pi*total_step_hieght/2))/(total_step_hieght*np.pi))
    cos_unwrap=(2*np.pi*k)+cos_wrap 
    return(cos_unwrap)

#median filter 
def filt(image,kernel,direc):
    dup_img=image.copy()
    if (direc=='v'):
        k=(1,kernel) #kernel size
    elif(direc=='h'):
        k=(kernel,1)
    med_fil=scipy.ndimage.median_filter(dup_img,k)
    k_array=(dup_img-med_fil)/(2*np.pi)
    fil_img=dup_img-(k_array*2*np.pi)
    return(fil_img)


#Single function for all
def temp_unwrap(mask_cos_v,mask_cos_h,mask_step_v,mask_step_h,limit,N,pitch,width,height,total_step_hieght,kernel_v,kernel_h):
    
    #Wrapped phases
    cos_wrap_v=phase_cal(mask_cos_v,N)
    cos_wrap_h=phase_cal(mask_cos_h,N)
    step_wrap_v=phase_cal(mask_step_v,N)
    step_wrap_h=phase_cal(mask_step_h,N)
    #Unwrapped
    unwrap_v=unwrap_cal(step_wrap_v,cos_wrap_v,pitch,width,height,total_step_hieght,'v')
    unwrap_h=unwrap_cal(step_wrap_h,cos_wrap_h,pitch,width,height,total_step_hieght,'h')
    
    #Apply filter
    fil_unwrap_v=filt(unwrap_v,kernel_v,'v')
    fil_unwrap_h=filt(unwrap_h,kernel_h,'h')
    
    return(fil_unwrap_v,fil_unwrap_h)

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

def proj_img(fil_unwrap_v,fil_unwrap_h,orig_img,pitch,width,height):
    proj_im=np.zeros((height,width))
    proj_u=unwrap_v*pitch/(2*np.pi)
    proj_v=unwrap_h*pitch/(2*np.pi)
    proj_u=proj_u.astype('int')
    proj_v=proj_v.astype('int')
    proj_im[proj_v,proj_u]=orig
    return(proj_im)




