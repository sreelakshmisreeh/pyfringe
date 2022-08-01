#!/usr/bin/env python
# coding: utf-8


import numpy as np
import scipy.ndimage
import os
import matplotlib.pyplot as plt

def delta_deck_gen(N, height, width):
    ''' 
    Function computes phase shift δ  values used in N-step phase shifting algorithm for each image pixel of given height and width. 
    δ_k  =  (2kπ)/N, where k = 1,2,3,... N and N is the number of steps.
    
    Parameters
    ----------
    N = type: int. The number of steps in phase shifting algorithm.
    height = type: float. Height of the pattern image.
    width = type: float. Width of pattern image.
    
    Returns
    -------
    delta_deck = type:float. N delta images.
    
    Ref: J. H. Brunning, D. R. Herriott, J. E. Gallagher, D. P. Rosenfeld, A. D. White, and D. J. Brangaccio, Digital wavefront measuring interferometer for testing optical surfaces, lenses, Appl. Opt. 13(11), 2693–2703, 1974. 
    '''
    delta = 2 * np.pi * np.arange(1, N + 1) / N 
    one_block = np.ones((N, height, width))
    delta_deck = np.einsum('ijk,i->ijk', one_block, delta)
    return delta_deck

def cos_func(inte_rang, pitch, direc, phase_st, delta_deck): #phase_st=datatype:float    
    ''' 
    Function creates cosine fringe pattern according to N step phase shifting algorithm.
    The intensity of the kth images with a phase shift of δ_k can be represented as
    I_k(x, y) = I′(x, y) + I′′(x, y) cos(φ(x, y) + δ_k) where φ(x, y) is the phase at pixel (x,y).
    
    Parameters
    ----------
    inte_rang = type: float. Operating intensity range or projector's linear operation region.
    pitch = type:float. Number of pixels per fringe period.
    direc = type: string. Visually vertical (v) or horizontal(h) pattern.
    phase_st = type: float. Starting phase. To apply multifrequency and multiwavelength temporal unwraping starting phase should be zero. Whereas for phase coding trmporal unwrapping starting phase should be -π.
    delta_deck =  type: float. Delta values at each pixel for each N step pattern.
    
    Returns
    -------
    inte = type:float array.  N intensity patterns.
    absolute_phi = absolute phase at each pixel. 
    '''
    height = delta_deck.shape[1]
    width = delta_deck.shape[2]
    #I′′(x, y) average intensity
    i1 = (inte_rang[1] - inte_rang[0])/2
    # I′(x, y) intensity modulation
    i0 = i1 + inte_rang[0]
    if direc == 'v': #vertical fringe pattern
        array = np.ones((height,1)) * np.arange(0,width)
    elif direc =='h': #horizontal fringe pattern
        array = np.ones((width,1)) * np.arange(0,height)
        array = np.rot90(array,3)
    
    absolute_phi = array / pitch * 2*np.pi + phase_st
    
    inte = i0 + i1 * np.cos(absolute_phi + delta_deck) 
    return inte, absolute_phi

def step_func(inte_rang, pitch, direc, delta_deck):
    '''
    Function generates stair phase coded images used for temporal phase unwrapping using phase coding.
    The stair phase function, φs(x, y)= −π + [x∕P] × 2π/n , where x∕Pk is the truncated integer representing fringe order.
    P the number of pixels per period; and n the total number of fringe periods.
    The intensity of the kth stair image,
    I_k(x, y) =  I'(x, y)+ I''(x, y) cos(φs + δ_k)

    Parameters
    ----------
    inte_rang = type: float. Operating intensity range or projector's linear operation region.
    pitch = type:float. number of pixels per fringe period.
    direc = type: string. vertical (v) or horizontal(h) patterns.
    delta_deck =  type: float. Delta values at each pixel for each N step pattern.

    Returns
    -------
    inte =  type:float array.  N intensity patterns
    
    Ref: Y. Wang and S.Zhang, Novel phase-coding method for absolute phase retrieval,Opt. Lett. 37(11), 2067–2069, 2012.
    '''
    height = delta_deck.shape[1]
    width = delta_deck.shape[2]
    i1 = (inte_rang[1] - inte_rang[0]) / 2
    i0 = i1 + inte_rang[0]
    if direc == 'v':
        n_fring = np.ceil(width / pitch) # number of fringes
        ar = np.arange(0, width)
        ar_array = np.ones((height, 1)) * ar
    elif direc == 'h':
        n_fring = np.ceil(height / pitch) # number of fringes
        ar = np.arange(0, height)
        ar_array = np.ones((width, 1)) * ar
        ar_array = np.rot90(ar_array, 3)
    phi_s = -np.pi + (np.floor(ar_array / pitch) * (2 * np.pi / (n_fring-1)))
    inte = i0 + i1 * np.cos(phi_s + delta_deck) 
    return inte

def calib_generate(width, height, type_unwrap, N_list, pitch_list, phase_st, inte_rang, path):
    '''
    Function to generate fringe patterns based on type of unwrapping. 
    This function generates both vertically and horizontally varying fringe patterns which is usually required for system calibration.

    Parameters
    ----------
    width = type: float. Width of pattern image.
    height = type: float. Height of the pattern image.
    type_unwrap = type: string. Type of temporal unwrapping to be applied. 
                  'phase' = phase coded unwrapping method, 
                  'multifreq' = multifrequency unwrapping method
                  'multiwave' = multiwavelength unwrapping method.
    N_list = type: float array. The number of steps in phase shifting algorithm. If phase coded unwrapping method is used this is a single element array. 
                                For other methods corresponding to each pitch one element in the list.
    pitch_list = type: float. Number of pixels per fringe period.
    phase_st = type: float. Starting phase. To apply multifrequency and multiwavelength temporal unwraping starting phase should be zero. 
                            Whereas for phase coding trmporal unwrapping starting phase should be -π.
    inte_rang = type: float. Operating intensity range or projector's linear operation region.
    path = type: string. Path to which the generated pattern is to be saved.

    Returns
    -------
    fringe_arr = type: Array of uint8. Array of generated fringe patterns in both directions.
    delta_deck_list = type:List of float. List of N delta images.

    '''
    
    fringe_lst = []; delta_deck_list = []
    if type_unwrap == 'phase':
        delta_deck_list = delta_deck_gen(N_list[0], height, width)
        step_v = step_func(inte_rang, pitch_list[0], 'v', delta_deck_list)
        step_h = step_func(inte_rang, pitch_list[0], 'h', delta_deck_list)
        cos_v, absolute_phi_v = cos_func(inte_rang, pitch_list[0], 'v', phase_st, delta_deck_list)
        cos_h, absolute_phi_h = cos_func(inte_rang, pitch_list[0], 'h', phase_st, delta_deck_list)
        fringe_lst = np.concatenate((cos_v, cos_h, step_v, step_h),axis=0)
        fringe_arr = np.ceil (fringe_lst).astype('uint8')  # for rounding to the next int number to avoid phase ambiguity
        
    elif (type_unwrap == 'multifreq' or type_unwrap == 'multiwave'):
        for p, n in zip(pitch_list, N_list): 
            delta_deck = delta_deck_gen(n, height, width)
            cos_v, absolute_phi_v = cos_func(inte_rang, p,'v', phase_st, delta_deck)
            cos_h, absolute_phi_h = cos_func(inte_rang, p,'h', phase_st, delta_deck)
            fringe_lst.append(cos_v)
            fringe_lst.append(cos_h)
            delta_deck_list.append(delta_deck)
        fringe_arr = np.ceil(np.vstack(fringe_lst)).astype('uint8')
    np.save(os.path.join(path, '{}_fringes.npy'.format(type_unwrap)), fringe_arr) 
    
    return fringe_arr, delta_deck_list 

def recon_generate(width, height, type_unwrap, N_list, pitch_list, phase_st, inte_rang, direc, path):
    '''
    Function is used to generate fringe pattern in a specified direction.

    width = type: float. Width of pattern image.
    height = type: float. Height of the pattern image.
    type_unwrap = type: string. Type of temporal unwrapping to be applied. 
                  'phase' = phase coded unwrapping method, 
                  'multifreq' = multifrequency unwrapping method
                  'multiwave' = multiwavelength unwrapping method.
    N_list = type: float array. The number of steps in phase shifting algorithm. If phase coded unwrapping method is used this is a single element array. 
                                For other methods corresponding to each pitch one element in the list.
    pitch_list = type: float. Number of pixels per fringe period.
    phase_st = type: float. Starting phase. To apply multifrequency and multiwavelength temporal unwraping starting phase should be zero. 
                            Whereas for phase coding trmporal unwrapping starting phase should be -π.
    inte_rang = type: float. Operating intensity range or projector's linear operation region.
    direc = type: string. Visually vertical (v) or horizontal(h) pattern.
    path = type: string. Path to which the generated pattern is to be saved.

    Returns
    -------
    fringe_arr = type: Array of uint8. Array of generated fringe patterns in single direction.
    delta_deck_list = type:List of float. List of N delta images.

    '''
    fringe_lst = []; delta_deck_list = []
    if type_unwrap == 'phase':
        delta_deck_list = delta_deck_gen(N_list[0], height, width)
        if direc =='v':
            step = step_func(inte_rang, pitch_list[0], 'v', delta_deck_list)
            cos, absolute_phi_v = cos_func(inte_rang, pitch_list[0], 'v', phase_st, delta_deck_list)
        elif direc == 'h':    
            step = step_func(inte_rang, pitch_list[0], 'h', delta_deck_list)
            cos, absolute_phi_h = cos_func(inte_rang, pitch_list[0], 'h', phase_st, delta_deck_list)
        fringe_lst = np.concatenate((cos, step),axis = 0)
        fringe_arr = np.ceil (fringe_lst).astype('uint8') 
    elif (type_unwrap == 'multifreq' or type_unwrap == 'multiwave'):
        for p, n in zip(pitch_list, N_list): 
            delta_deck = delta_deck_gen(n, height, width)
            if direc =='v':
                cos, absolute_phi = cos_func(inte_rang, p,'v', phase_st, delta_deck)
            elif direc == 'h':
                cos, absolute_phi = cos_func(inte_rang, p,'h', phase_st, delta_deck)
            fringe_lst.append(cos)
            delta_deck_list.append(delta_deck)
        fringe_arr=np.ceil(np.vstack(fringe_lst)).astype('uint8') 
    np.save(os.path.join(path, '{}_fringes.npy'.format(type_unwrap)), fringe_arr) 
    return fringe_arr, delta_deck_list

def mask_img(images, limit ):
    '''
    Function computes and applies mask to captured image based on data modulation (relative modulation) of each pixel.
    data modulation = I''(x,y)/I'(x,y). 

    Parameters
    ----------
    images = type:uint. Captured fringe images. 
    limit = type: float. Data modulation limit. Regions with data modulation lower than limit will be masked out.

    Returns
    -------
    masked_img = type: numpy masked array of floats. Images after applying mask
    modulation : type:float. Intensity modulation array(image) for each captured image 
    avg = type: float. Average intensity array(image) for each captured image.
    gamma = type: float. Data modulation (relative modulation) array for each captured image
    delta_deck = type: float. Delta values at each pixel for each captured image

    '''
    delta_deck = delta_deck_gen(images.shape[0], images.shape[1], images.shape[2])
    N = delta_deck.shape[0]
    images = images.astype(np.float64)
    sin_delta = np.sin(delta_deck)
    sin_delta[np.abs(sin_delta) < 1e-15] = 0 
    sin_lst = (np.sum(images * sin_delta, axis = 0)) ** 2
    cos_delta = np.cos(delta_deck)
    cos_delta[np.abs(cos_delta)<1e-15] = 0
    cos_lst = (np.sum(images * cos_delta, axis = 0)) ** 2
    modulation = 2 * np.sqrt(sin_lst + cos_lst) / N
    avg = np.sum(images, axis = 0) / N
    gamma = modulation / avg
    mask = np.full(modulation.shape,True)
    mask[ modulation > limit] = False
    masked_img = np.ma.masked_array(images, np.repeat(mask[np.newaxis,:,:], N, axis = 0),fill_value=np.nan)
   
    return masked_img, modulation, avg, gamma , delta_deck


#Wrap phase calculation
def phase_cal(images, N, delta_deck):
    '''
    Function computes the wrapped phase map from captured N step fringe pattern images.

    Parameters
    ----------
    images = type: float. Captured fringe pattern images
    N = type:mint.  The number of steps in phase shifting algorithm
    delta_deck = type: float. Delta values at each pixel for each captured image

    Returns
    -------
    ph = type: float. Wrapped phase map

    '''
    sin_delta = np.sin(delta_deck)
    sin_delta[np.abs(sin_delta) < 1e-15] = 0 
    sin_lst = (np.sum(images * sin_delta, axis = 0))   
    cos_delta = np.cos(delta_deck)
    cos_delta[np.abs(cos_delta) < 1e-15] = 0
    cos_lst = (np.sum(images * cos_delta, axis = 0))
    #wrapped phase
    ph = -np.arctan2(sin_lst,cos_lst)# wraped phase;  
    
    return ph 

def step_rectification(step_ph,direc):
    '''
    This function rectify abnormal phase jumps at the ends of stair coded phase maps caused by unstable region of arctangent function (−π,π).

    Parameters
    ----------
    step_ph = type: float. Wrapped phase map from stair coded pattern images.
    direc = type: string. vertical (v) or horizontal(h) pattern.

    Returns
    -------
    step_ph = type: float. Rectified stair coded wrapped phase map.

    '''
    img_width = step_ph.shape[1] # number of col
    img_height = step_ph.shape[0] # number of row
    if direc == 'v':
        step_ph[:,0:int(img_width/2)][step_ph[:,0:int(img_width/2)] >( 0.9 * np.pi)] = step_ph[:,0:int(img_width/2)][step_ph[:,0:int(img_width/2)] >( 0.9 * np.pi)] - 2 * np.pi
        step_ph[:,int(img_width/2):img_width][step_ph[:,int(img_width/2):img_width] < (-0.9 * np.pi)] = step_ph[:,int(img_width/2):img_width][step_ph[:,int(img_width/2):img_width] < (-0.9 * np.pi)] + 2 * np.pi
    elif direc == 'h':
        step_ph[0:int(img_height/2),:][step_ph[0:int(img_height/2),:] > (0.9 * np.pi)] = step_ph[0:int(img_height/2),:][step_ph[0:int(img_height/2),:] > (0.9 * np.pi)] - 2 * np.pi
        step_ph[int(img_height/2):img_height, :][step_ph[int(img_height/2):img_height, :] < (-0.9 * np.pi)] = step_ph[int(img_height/2):img_height, :][step_ph[int(img_height/2):img_height, :] < (-0.9 * np.pi)] + 2 * np.pi

    return step_ph


def unwrap_cal(step_wrap,cos_wrap,pitch,width,height,direc):
    '''
    Function applies phase coded temporal phase unwrapping to obtain absolute phase map.

    Parameters
    ----------
    step_wrap = type: float. Wrapped stair coded phase map.
    cos_wrap = type: float. Wrapped phase map of cosine intensity fringe patterns.
    pitch = type:float. number of pixels per fringe period.
    width = type:float. Width of projector image.
    height = type: float. Height of projector image.
    direc = type: string. vertical (v) or horizontal(h) pattern.

    Returns
    -------
    cos_unwrap = type:float. Unwrapped absolute phase map
    k = type: int. Fringe order from stair phase
        
    Ref: Y. Wang and S.Zhang, Novel phase-coding method for absolute phase retrieval,Opt. Lett. 37(11), 2067–2069, 2012.

    '''
    if direc == 'v':
        n_fring = np.ceil(width/pitch)
    elif direc == 'h':
        n_fring = np.ceil(height/pitch)
    k = np.round((n_fring - 1) * (step_wrap + (np.pi)) / (2 * np.pi))    
    cos_unwrap = (2 * np.pi * k) + cos_wrap
    return cos_unwrap, k

#median filter 
def filt(unwrap, kernel ,direc):
    '''
    Function is used to remove artifacts generated in the temporal unwrapped phase map. 
    A median filter is applied to locate incorrectly unwrapped points, and those point phase is corrected by adding or subtracting a integer number of 2π.

    Parameters
    ----------
    unwrap = type: float. Unwrapped phase map with spike noise.
    kernel = type: int. Kernel size for median filter.
    direc = type: string. vertical (v) or horizontal(h) pattern.

    Returns
    -------
    correct_unwrap = type: float. Corrected unwrapped phase map.
    k_array = type: int. Spiking point fringe order.

    '''
    dup_img = unwrap.copy()
    if direc == 'v':
        k = (1,kernel) #kernel size
    elif direc == 'h':
        k = (kernel,1)
    med_fil = scipy.ndimage.median_filter(dup_img, k)
    k_array = np.round((dup_img - med_fil) / (2 * np.pi))
    correct_unwrap = dup_img - (k_array * 2 * np.pi)
    return correct_unwrap, k_array

def ph_temp_unwrap(mask_cos_v, mask_cos_h, mask_step_v, mask_step_h, pitch, height,width, capt_delta_deck, kernel_v, kernel_h):
    '''
    Wrapper function for phase coded temporal unwrapping. This function takes masked cosine and stair phase shifted images as input and computes wrapped phase maps. 
    Wrapped phase maps are used to unwrap and obtain absolute phase map which is then further processed to remove spike noise using median filter rectification.

    Parameters
    ----------
    mask_cos_v = type: float. Masked numpy array of cosine fringe pattern variation in the horizontal direction
    mask_cos_h = type: float. Masked numpy array of cosine fringe pattern variation in the vertical direction
    mask_step_v = type: float. Masked numpy array of stair fringe pattern variation in the horizontal direction
    mask_step_h = type: float. Masked numpy array of stair fringe pattern variation in the vertical direction
    pitch = type:float. number of pixels per fringe period.
    height = type: float. Height of projector image.
    width = type: float. Width of projector image.
    capt_delta_deck = type: float. Delta values at each pixel for each captured image
    kernel_v = type: int. Kernel size for median filter to be applied in the horizontal direction
    kernel_h = type: int. Kernel size for median filter to be applied in the vertical direction

    Returns
    -------
    fil_unwrap_v = type: float. Unwrapped median filter rectified absolute phase map varying in the horizontal direction
    fil_unwrap_h = type: float. Unwrapped median filter rectified absolute phase map varying in the vertical direction
    k0_v = type: int. Spiking point fringe order in the horizontal direction.
    k0_h = type: int. Spiking point fringe order in the vertical direction.
    cos_wrap_v = type: float. Wrapped phase map of cosine intensity pattern varying in the horizontal direction.
    cos_wrap_h = type: float. Wrapped phase map of cosine intensity pattern vvarying in the vertical direction.
    step_wrap_v = type: float.  Wrapped phase map of stair intensity pattern varying in the horizontal direction.
    step_wrap_h = type: float.  Wrapped phase map of stair intensity pattern varying in the vertical direction.

    '''
    N = capt_delta_deck.shape[0]
    
    #Wrapped phases
    cos_wrap_v = phase_cal(mask_cos_v, N, capt_delta_deck)
    cos_wrap_h = phase_cal(mask_cos_h, N, capt_delta_deck)
    step_wrap_v = phase_cal(mask_step_v, N, capt_delta_deck)
    step_wrap_h = phase_cal(mask_step_h, N, capt_delta_deck)
    
    #step rectification for border jumps
    step_wrap_v = step_rectification(step_wrap_v, 'v')
    step_wrap_h = step_rectification(step_wrap_h, 'h')
    #Unwrapped
    unwrap_v,k_v = unwrap_cal(step_wrap_v, cos_wrap_v, pitch, width, height, 'v')
    unwrap_h,k_h = unwrap_cal(step_wrap_h, cos_wrap_h, pitch, width, height, 'h')
    
    #Apply median rectification
    fil_unwrap_v,k0_v = filt(unwrap_v, kernel_v, 'v')
    fil_unwrap_h,k0_h = filt(unwrap_h, kernel_h, 'h')    
    return fil_unwrap_v, fil_unwrap_h, k0_v, k0_h, cos_wrap_v, cos_wrap_h, step_wrap_v, step_wrap_h # Filtered unwraped phase maps and k values

def multi_kunwrap(wavelength, ph):
    '''
    Function performs temporal phase unwrapping using the low and high wavelngth wrapped phase maps.
    

    Parameters
    ----------
    wavelength = type: float array of wavelengths with decreasing wavelengths (increasing frequencies)
    ph = type: float array of wrapped phase maps corresponding to decreasing wavelngths (increasing frequencies).

    Returns
    -------
    unwrap = type. float. Unwrapped phase map
    k = type: int. Fringe order of lowest wavelength (highest frequency)

    '''
    k = np.round(((wavelength[0] / wavelength[1]) * ph[0] - ph[1])/ (2 * np.pi))
    unwrap = ph[1] + 2 * np.pi * k
    return unwrap, k

def multifreq_unwrap(wavelength_arr, phase_arr, kernel, direc):
    '''
    Function performs sequential temporal multifrequency phase unwrapping from high wavelength (low frequency) wrapped phase map to low wavelength (high frequency) wrapped phase map.

    Parameters
    ----------
    wavelength_arr = type: array of float wavelengths from high wavelngth to low wavelength.
    phase_arr = type: array of float wrapped phase maps from high wavelngth to low wavelength.

    Returns
    -------
    absolute_ph4 = type float. The final unwrapped phase map of low wavelength (high frequency) wrapped phase map.
    k4 = type: int. The fringe order of low wavelength (high frequency) phase map.

    '''
    absolute_ph,k = multi_kunwrap(wavelength_arr[0:2], phase_arr[0:2])   
    for i in range(1,len(wavelength_arr)-1):
        absolute_ph,k = multi_kunwrap(wavelength_arr[i:i+2], [absolute_ph, phase_arr[i+1]])    
    absolute_ph, k0 = filt(absolute_ph, kernel, direc)    
    return absolute_ph, k

def multiwave_unwrap(wavelength_arr, phase_arr, kernel, direc):
    '''
    Function performs sequential temporal multiwavelength phase unwrapping from high wavelength and applies median filter rectification to remove artifacts.

    Parameters
    ----------
    wavelength_arr = type:array of float wavelengths from high wavelngth to low wavelength.
    phase_arr = type: array of float wrapped phase maps from high wavelngth to low wavelength.
    kernel type: int. Kernel size for median filter to be applied.
    direc = type: string. vertical (v) or horizontal(h) pattern.

    Returns
    -------
    absolute_ph1 = type:float. Absolute unwrapped phase map.
    k1 = type: int. The fringe order of low wavelength (high frequency) phase map. 

    '''
    
    absolute_ph,k = multi_kunwrap(wavelength_arr[0:2], phase_arr[0:2])
    absolute_ph, k0 = filt(absolute_ph, kernel, direc)
    for i in range(1,len(wavelength_arr)-1):
        absolute_ph,k = multi_kunwrap(wavelength_arr[i:i+2], [absolute_ph, phase_arr[i+1]])    
    absolute_ph, k0 = filt(absolute_ph, kernel, direc)    
    return absolute_ph, k

def edge_rectification(multi_phase_123,direc):
    '''
    Function to rectify abnormal phase jumps at the edge of multiwavelength high wavelength wrapped phase map.

    Parameters
    ----------
    multi_phase_123 = type: float.  Highest wavelength wrapped phase map in multiwavelength temporal unwrapping algorithm.
    direc = type: string. vertical (v) or horizontal(h) pattern.

    Returns
    -------
    multi_phase_123 = type: float. Rectified phase map.

    '''
    img_height=multi_phase_123.shape[1]
    img_width=multi_phase_123.shape[0]
    if direc == 'v':
        multi_phase_123[:,0:int(img_width/2)][multi_phase_123[:,0:int(img_width/2)] > 1.5 * np.pi] = multi_phase_123[:,0:int(img_width/2)][multi_phase_123[:,0:int(img_width/2)] > 1.5 * np.pi] - 2 * np.pi
        multi_phase_123[:,int(img_width/2):][multi_phase_123[:,int(img_width/2):] < -1.5 * np.pi] = multi_phase_123[:,int(img_width/2):][multi_phase_123[:,int(img_width/2):] < -1.5 * np.pi] + 2 * np.pi
    elif direc == 'h':
        multi_phase_123[0:int(img_height/2)][multi_phase_123[0:int(img_height/2)] > 1.5 * np.pi] = multi_phase_123[0:int(img_height/2)][multi_phase_123[0:int(img_height/2)] > 1.5 * np.pi] - 2 * np.pi
        multi_phase_123[int(img_height/2):][multi_phase_123[int(img_height/2):] < -1.5 * np.pi] = multi_phase_123[int(img_height/2):][multi_phase_123[int(img_height/2):] < -1.5 * np.pi] + 2 * np.pi
    return multi_phase_123

def bilinear_interpolate(unwrap, center):
    '''
    Function to perform bilinear interpolation to obtain subpixel circle center phase values.

    Parameters
    ----------
    unwrap = type:float. Absolute phase map
    center = type:float. Subpixel coordinate from OpenCV circle center detection.

    Returns
    -------
   Subpixel mapped absolute phase value corresponding to given circle center. 

    '''
    x = np.asarray(center[0])
    y = np.asarray(center[1])
    #neighbours
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1
    unwrap_a = unwrap[y0, x0 ]
    unwrap_b = unwrap[y1, x0 ]
    unwrap_c = unwrap[y0, x1 ]
    unwrap_d = unwrap[y1, x1 ]
    #weights
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*unwrap_a + wb*unwrap_b + wc*unwrap_c + wd*unwrap_d
#=====================================================
# For diagnosis
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








