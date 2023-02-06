#!/usr/bin/env python
# coding: utf-8


import numpy as np
import scipy.ndimage
import os
from typing import Tuple
import pickle

def delta_deck_gen(N: int,
                   height: int,
                   width: int) -> np.ndarray:
    """
    Function computes phase shift δ  values used in N-step phase shifting algorithm for each image pixel of given height and width. 
    δ_k  =  (2kπ)/N, where k = 1,2,3,... N and N is the number of steps.
    
    Parameters
    ----------
    N : int.
        The number of steps in phase shifting algorithm.
    height : int.
            Height of the pattern image.
    width : int.
            Width of pattern image.
    
    Returns
    -------
    delta_deck :numpy.ndarray:float.
                N delta images.
    
    Ref: J. H. Brunning, D. R. Herriott, J. E. Gallagher, D. P. Rosenfeld, A. D. White, and D. J. Brangaccio, Digital wavefront measuring interferometer for testing optical surfaces, lenses,
    Appl. Opt. 13(11), 2693–2703, 1974.
    """
    delta = 2 * np.pi * np.arange(1, N + 1) / N
    one_block = np.ones((N, height, width))
    delta_deck = np.einsum('ijk,i->ijk', one_block, delta)
    return delta_deck

def cos_func(inte_rang: list,
             pitch: int,
             direc: str,
             phase_st: float,
             delta_deck: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function creates cosine fringe pattern according to N step phase shifting algorithm.
    The intensity of the kth images with a phase shift of δ_k can be represented as
    I_k(x, y) = I′(x, y) + I′′(x, y) cos(φ(x, y) + δ_k) where φ(x, y) is the phase at pixel (x,y).
    
    Parameters
    ----------
    inte_rang = type: list. Operating intensity range or projector's linear operation region.
    pitch = type: int. Number of pixels per fringe period.
    direc = type: str. Visually vertical (v) or horizontal(h) pattern.
    phase_st = type: float. Starting phase. To apply multi frequency and multi wavelength temporal un wrapping starting
                            phase should be zero. Whereas for phase coding temporal unwrapping starting phase
                            should be -π.
    delta_deck =  type: np.ndarray. Delta values at each pixel for each N step pattern.
    
    Returns
    -------
    inte = type:numpy.ndarray.  N intensity patterns.
    absolute_phi = type:numpy.ndarray absolute phase at each pixel.
    """
    height = delta_deck.shape[1]
    width = delta_deck.shape[2]

    # I′(x, y) intensity modulation
    i1 = (inte_rang[1] - inte_rang[0])/2

    # I′′(x, y) average intensity
    i0 = i1 + inte_rang[0]

    if direc == 'v':  # vertical fringe pattern
        array = np.ones((height, 1)) * np.arange(0, width)
    elif direc == 'h':  # horizontal fringe pattern
        array = np.ones((width, 1)) * np.arange(0, height)
        array = np.rot90(array, 3)
    else:
        array = None
        print("ERROR: direction parameter is invalid, must be one of {'v', 'h'}.")
    absolute_phi = array / pitch * 2*np.pi + phase_st
    inte = i0 + i1 * np.cos(absolute_phi + delta_deck) 
    return inte, absolute_phi

def step_func(inte_rang: list,
              pitch: int,
              direc: str,
              delta_deck: np.ndarray) -> np.ndarray:
    """
    Function generates stair phase coded images used for temporal phase unwrapping using phase coding.
    The stair phase function, φs(x, y)= −π + [x∕P] × 2π/n , where x∕Pk is the truncated integer representing fringe order.
    P the number of pixels per period; and n the total number of fringe periods.
    The intensity of the kth stair image,
    I_k(x, y) =  I'(x, y)+ I''(x, y) cos(φs + δ_k)
    Parameters
    ----------
    inte_rang = type:list. Operating intensity range or projector's linear operation region.
    pitch = type:float. number of pixels per fringe period.
    direc = type: string. vertical (v) or horizontal(h) patterns.
    delta_deck =  type:numpy.ndarray:float. Delta values at each pixel for each N step pattern.
    Returns
    -------
    inte =  type:float array.  N intensity patterns
    Ref: Y. Wang and S.Zhang, Novel phase-coding method for absolute phase retrieval,Opt. Lett. 37(11), 2067–2069, 2012.
    """
    height = delta_deck.shape[1]
    width = delta_deck.shape[2]
    i1 = (inte_rang[1] - inte_rang[0]) / 2
    i0 = i1 + inte_rang[0]
    if direc == 'v':
        n_fringe = np.ceil(width / pitch)  # number of fringes
        ar = np.arange(0, width)
        ar_array = np.ones((height, 1)) * ar
    elif direc == 'h':
        n_fringe = np.ceil(height / pitch)  # number of fringes
        ar = np.arange(0, height)
        ar_array = np.ones((width, 1)) * ar
        ar_array = np.rot90(ar_array, 3)
    else:
        n_fringe = None
        ar_array = None
        print("ERROR: direction parameter is invalid, must be one of {'v', 'h'}.")
    phi_s = -np.pi + (np.floor(ar_array / pitch) * (2 * np.pi / (n_fringe-1)))
    inte = i0 + i1 * np.cos(phi_s + delta_deck) 
    return inte

def calib_generate(width: int,
                   height: int,
                   type_unwrap: str,
                   N_list: list,
                   pitch_list: list,
                   phase_st: float,
                   inte_rang: list,
                   path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function to generate fringe patterns based on type of unwrapping. 
    This function generates both vertically and horizontally varying fringe patterns which is usually required for system calibration.

    Parameters
    ----------
    width = type: float. Width of pattern image.
    height = type: float. Height of the pattern image.
    type_unwrap = type: string. Type of temporal unwrapping to be applied. 
                  'phase' = phase coded unwrapping method, 
                  'multifreq' = multi frequency unwrapping method
                  'multiwave' = multi wavelength unwrapping method.
    N_list = type: float array. The number of steps in phase shifting algorithm. If phase coded unwrapping method is used this is a single element array. 
                                For other methods corresponding to each pitch one element in the list.
    pitch_list = type: float. Number of pixels per fringe period.
    phase_st = type: float. Starting phase. To apply multi frequency and multi wavelength temporal un wrapping starting phase should be zero.
                            Whereas for phase coding temporal unwrapping starting phase should be -π.
    inte_rang = type:list. Operating intensity range or projector's linear operation region.
    path = type: string. Path to which the generated pattern is to be saved.

    Returns
    -------
    fringe_arr = type: Array of uint8. Array of generated fringe patterns in both directions.
    delta_deck_list = type:List of float. List of N delta images.

    """

    if type_unwrap == 'phase':
        delta_deck_list = delta_deck_gen(N_list[0], height, width)
        step_v = step_func(inte_rang, pitch_list[0], 'v', delta_deck_list)
        step_h = step_func(inte_rang, pitch_list[0], 'h', delta_deck_list)
        cos_v, absolute_phi_v = cos_func(inte_rang, pitch_list[0], 'v', phase_st, delta_deck_list)
        cos_h, absolute_phi_h = cos_func(inte_rang, pitch_list[0], 'h', phase_st, delta_deck_list)
        fringe_lst = np.concatenate((cos_v, cos_h, step_v, step_h), axis=0)
        fringe_arr = np.ceil(fringe_lst).astype('uint8')  # for rounding to the next int number to avoid phase ambiguity
        
    elif type_unwrap == 'multifreq' or type_unwrap == 'multiwave':
        fringe_lst = []
        delta_deck_list = []
        for p, n in zip(pitch_list, N_list): 
            delta_deck = delta_deck_gen(n, height, width)
            cos_v, absolute_phi_v = cos_func(inte_rang, p, 'v', phase_st, delta_deck)
            cos_h, absolute_phi_h = cos_func(inte_rang, p, 'h', phase_st, delta_deck)
            fringe_lst.append(cos_v)
            fringe_lst.append(cos_h)
            delta_deck_list.append(delta_deck)
        fringe_arr = np.ceil(np.vstack(fringe_lst)).astype('uint8')
    else:
        print('ERROR:Invalid unwrapping type')
        fringe_arr = None
        delta_deck_list = None
    np.save(os.path.join(path, '{}_fringes.npy'.format(type_unwrap)), fringe_arr) 
    
    return fringe_arr, delta_deck_list 

def recon_generate(width: int,
                   height: int,
                   type_unwrap: str,
                   N_list: list,
                   pitch_list: list,
                   phase_st: float,
                   inte_rang: list,
                   direc: str,
                   path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function is used to generate fringe pattern in a specified direction.

    width = type: float. Width of pattern image.
    height = type: float. Height of the pattern image.
    type_unwrap = type: string. Type of temporal unwrapping to be applied. 
                  'phase' = phase coded unwrapping method, 
                  'multifreq' = multi frequency unwrapping method
                  'multiwave' = multi wavelength unwrapping method.
    N_list = type: float array. The number of steps in phase shifting algorithm. If phase coded unwrapping method is used this is a single element array. 
                                For other methods corresponding to each pitch one element in the list.
    pitch_list = type: float. Number of pixels per fringe period.
    phase_st = type: float. Starting phase. To apply multi frequency and multi wavelength temporal un wrapping starting phase should be zero.
                            Whereas for phase coding temporal unwrapping starting phase should be -π.
    inte_rang = type: list. Operating intensity range or projector's linear operation region.
    direc = type: string. Visually vertical (v) or horizontal(h) pattern.
    path = type: string. Path to which the generated pattern is to be saved.

    Returns
    -------
    fringe_arr = type: Array of uint8. Array of generated fringe patterns in single direction.
    delta_deck_list = type:List of float. List of N delta images.

    """
    fringe_lst = []
    delta_deck_list = []
    if type_unwrap == 'phase':
        delta_deck_list = delta_deck_gen(N_list[0], height, width)
        if direc == 'v':
            step = step_func(inte_rang, pitch_list[0], 'v', delta_deck_list)
            cos, absolute_phi_v = cos_func(inte_rang, pitch_list[0], 'v', phase_st, delta_deck_list)
        elif direc == 'h':    
            step = step_func(inte_rang, pitch_list[0], 'h', delta_deck_list)
            cos, absolute_phi_h = cos_func(inte_rang, pitch_list[0], 'h', phase_st, delta_deck_list)
        else:
            print('ERROR:Invalid direction. Directions should be \'v\'for vertical fringes and \'h\'for horizontal fringes')
            cos = None
            step = None
        fringe_lst = np.concatenate((cos, step), axis=0)
        fringe_arr = np.ceil(fringe_lst).astype('uint8')
    elif type_unwrap == 'multifreq' or type_unwrap == 'multiwave':
        for p, n in zip(pitch_list, N_list): 
            delta_deck = delta_deck_gen(n, height, width)
            if direc == 'v':
                cos, absolute_phi = cos_func(inte_rang, p, 'v', phase_st, delta_deck)
            elif direc == 'h':
                cos, absolute_phi = cos_func(inte_rang, p, 'h', phase_st, delta_deck)
            else:
                print('ERROR:Invalid direction. Directions should be \'v\'for vertical fringes and \'h\'for horizontal fringes')
                cos = None
            fringe_lst.append(cos)
            delta_deck_list.append(delta_deck)
        fringe_arr = np.ceil(np.vstack(fringe_lst)).astype('uint8')
    else:
        print('ERROR:Invalid unwrapping type')
        fringe_arr = None
        delta_deck_list = None
    np.save(os.path.join(path, '{}_fringes.npy'.format(type_unwrap)), fringe_arr) 
    return fringe_arr, delta_deck_list

def B_cutoff_limit(sigma_path: str,
                   quantile_limit: float,
                   N_list: list,
                   pitch_list: list) -> float:
    """
    Function to calculate modulation minimum based on success rate.
    :param sigma_path:  Path to read variance of noise model (sigma)
    :param quantile_limit:  Sigma level upto which all pixels can be successfully unwrapped.
    :param N_list:  Number of images taken for each level.
    :param pitch_list: Number of pixels per fringe period in each level
    :type sigma_path:str
    :type quantile_limit:float
    :type N_list:list
    :type pitch_list:list
    :return Lower limit of modulation. Pixels above this value is used for reconstruction.
    :rtype:float

    """
    sigma = np.load(sigma_path)
    sigma_sq_delta_phi = (np.pi / quantile_limit)**2
    modulation_limit_sq = ((pitch_list[-1]**2 / pitch_list[-2]**2) + 1) * (2 * sigma**2) / (N_list[-1] * sigma_sq_delta_phi)
    return np.sqrt(modulation_limit_sq)

def level_process(image_stack: np.ndarray,
                  n: int)->Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Helper function for phase_cal to perform intermediate calculation in each level.
    Parameters
    ----------
    image_stack: np.ndarray:np.float64.
                 Image stack of levels with same number of patterns (N).
    n: int.
        Number of patterns.
    """
    delta = 2 * np.pi * np.arange(1, n + 1) / n
    sin_delta = np.sin(delta)
    sin_delta[np.abs(sin_delta) < 1e-15] = 0
    cos_delta = np.cos(delta)
    cos_delta[np.abs(cos_delta) < 1e-15] = 0
    sin_deck = np.einsum('ijkl,j->ikl', image_stack, sin_delta)
    cos_deck = np.einsum('ijkl,j->ikl', image_stack, cos_delta)
    modulation_deck = 2 * np.sqrt(sin_deck ** 2 + cos_deck ** 2) / n
    average_deck = np.sum(image_stack, axis=1) / n
    return sin_deck, cos_deck, modulation_deck, average_deck


def mask_application(mask: np.ndarray,
                     mod_stack: np.ndarray,
                     sin_stack: np.ndarray,
                     cos_stack: np.ndarray)->Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Helper function to apply mask get relevant pixels for phase calculations.
    """
    num_total_levels = mod_stack.shape[0]

    mask = mask.astype('float')
    mask[mask == 0] = np.nan
    # apply mask to all levels
    mod_stack = np.einsum("ijk, jk->ijk", mod_stack, mask)
    flag = ~np.isnan(mod_stack)
    sin_stack = sin_stack[flag].reshape((num_total_levels, -1))
    cos_stack = cos_stack[flag].reshape((num_total_levels, -1))
    mod_stack = mod_stack[flag].reshape((num_total_levels, -1))
    return sin_stack, cos_stack, mod_stack

def phase_cal(images: np.ndarray,
              limit: float, 
              N: list,
              calibration: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Function computes phase map for all levels given in list N.
    Parameters
    ----------
    images: np.ndarray:np.float64.
            Captured fringe images.
    limit: float.
           Data modulation limit. Regions with data modulation lower than limit will be masked out.
    N: list.
        List of number of patterns in each level.
    calibration: bool.
                 If calibration is set the double of N is taken assuming horizontal and vertical fringes.

    Returns
    -------
    mod_stack :np.ndarray:float.
                Intensity modulation array(image) for each captured image
    white_stack: np.ndarray:float.
                 White image from fringe patterns.
    phase_map: np.ndarray:float.
               Wrapped phase map of each level stacked together after applying mask.
    mask: bool
          Mask applied to image.

    """
    if calibration:
        repeat = 2
    else:
        repeat = 1
    if len(set(N))==2:
        image_set1 = images[0:(repeat*len(N)-repeat)*N[0]].reshape((repeat*len(N)-repeat), N[0], images.shape[-2], images.shape[-1])
        image_set2 = images[(repeat*len(N)-repeat)*N[0]:].reshape(repeat, N[-1], images.shape[-2], images.shape[-1])
        image_set = [image_set1, image_set2]

        mask = np.full((images.shape[-2], images.shape[-1]), True)
        sin_stack = None
        cos_stack = None
        mod_stack = None
        white_stack = None
        for i, n in enumerate(sorted(set(N))):
            sin_deck, cos_deck, modulation, average_int = level_process(image_set[i], n)
            if i == 0:
                sin_stack = sin_deck
                cos_stack = cos_deck
                mod_stack = modulation
                white_stack = modulation + average_int
            else:
                sin_stack = np.vstack((sin_stack, sin_deck))
                cos_stack = np.vstack((cos_stack, cos_deck))
                mod_stack = np.vstack((mod_stack, modulation))
                white_stack = np.vstack((white_stack, (modulation + average_int)))
            mask_temp = modulation > limit
            mask &= np.prod(mask_temp, axis=0, dtype=bool)
    else:
        image_set = images.reshape(int(images.shape[0]/N[0]), N[0], images.shape[-2], images.shape[-1])
        sin_stack, cos_stack, mod_stack, average_stack = level_process(image_set, N[0])
        white_stack = mod_stack + average_stack
        mask = mod_stack > limit
        mask = np.prod(mask, axis=0, dtype=bool)
    sin_stack, cos_stack, mod_stack = mask_application(mask, mod_stack, sin_stack, cos_stack)
    phase_map = -np.arctan2(sin_stack, cos_stack)  # wrapped phase;
    return mod_stack, white_stack, phase_map, mask

def recover_image(vector_array: np.ndarray, 
                  flag: np.ndarray, 
                  cam_height: int, 
                  cam_width: int)-> np.ndarray:
    """
    Function to convert vector to image array using flag.
    vector_array: np.ndarray
                  Vector to be converted
    flag: np.ndarray
          Indexes of the array cordinates with data.
    cam_width: int.
               Width of image.
    cam_height: int.
                Height of image.
    """
    image = np.full((cam_height, cam_width), np.nan)
    image[flag] = vector_array
    return image

def step_rectification(step_ph: np.ndarray,
                       direc: str) -> np.ndarray:
    """
    This function rectify abnormal phase jumps at the ends of stair coded phase maps caused by unstable region of
    arc-tangent function (−π,π).
    Parameters
    ----------
    step_ph = type: float. Wrapped phase map from stair coded pattern images.
    direc = type: string. vertical (v) or horizontal(h) pattern.
    Returns
    -------
    step_ph = type: float. Rectified stair coded wrapped phase map.
    """
    img_width = step_ph.shape[1]  # number of col
    img_height = step_ph.shape[0]  # number of row
    if direc == 'v':
        step_ph[:, 0:int(img_width/2)][step_ph[:, 0:int(img_width/2)] > (0.9 * np.pi)] = step_ph[:, 0:int(img_width/2)][step_ph[:, 0:int(img_width/2)] > (0.9 * np.pi)] - 2 * np.pi
        step_ph[:, int(img_width/2):img_width][step_ph[:, int(img_width/2):img_width] < (-0.9 * np.pi)] = step_ph[:, int(img_width/2):img_width][step_ph[:, int(img_width/2):img_width] < (-0.9 * np.pi)] + 2 * np.pi
    elif direc == 'h':
        step_ph[0:int(img_height/2), :][step_ph[0:int(img_height/2), :] > (0.9 * np.pi)] = step_ph[0:int(img_height/2), :][step_ph[0:int(img_height/2), :] > (0.9 * np.pi)] - 2 * np.pi
        step_ph[int(img_height/2):img_height, :][step_ph[int(img_height/2):img_height, :] < (-0.9 * np.pi)] = step_ph[int(img_height/2):img_height, :][step_ph[int(img_height/2):img_height, :] < (-0.9 * np.pi)] + 2 * np.pi

    return step_ph


def unwrap_cal(step_wrap: np.ndarray,
               cos_wrap: np.ndarray,
               pitch: int,
               width: int,
               height: int,
               direc: str) -> Tuple[np.ndarray, np.ndarray]:
    """
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

    """
    if direc == 'v':
        n_fring = np.ceil(width/pitch)
    elif direc == 'h':
        n_fring = np.ceil(height/pitch)
    else:
        print("ERROR:Invalid directions.Directions should be \'v\'for vertical fringes and \'h\'for horizontal fringes")
        n_fring = None
    k = np.round((n_fring - 1) * (step_wrap + np.pi) / (2 * np.pi))
    cos_unwrap = (2 * np.pi * k) + cos_wrap
    return cos_unwrap, k

# median filter
def filt(unwrap: np.ndarray,
         kernel: int,
         direc: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function is used to remove artifacts generated in the temporal unwrapped phase map. 
    A median filter is applied to locate incorrectly unwrapped points, and those point phase is corrected by adding or
    subtracting an integer number of 2π.

    Parameters
    ----------
    unwrap: np.ndarray:float.
            Unwrapped phase map with spike noise.
    kernel: int.
            Kernel size for median filter.
    direc: str.
           Vertical (v) or horizontal(h) pattern.
    Returns
    -------
    correct_unwrap: np.ndarray:float.
                    Corrected unwrapped phase map.
    k_array: int.
             Spiking point fringe order.

    """
    if direc == 'v':
        k = (1, kernel)  # kernel size
    elif direc == 'h':
        k = (kernel, 1)
    else:
        print("ERROR:Invalid directions.Directions should be \'v\'for vertical fringes and \'h\'for horizontal fringes")
        k = None
    med_fil = scipy.ndimage.median_filter(unwrap, k)
    k_array = np.round((unwrap - med_fil) / (2 * np.pi))
    correct_unwrap = unwrap - (k_array * 2 * np.pi)
    return correct_unwrap, k_array

def ph_temp_unwrap(cos_wrap_v: np.ndarray,
                   cos_wrap_h: np.ndarray,
                   step_wrap_v: np.ndarray,
                   step_wrap_h: np.ndarray,
                   pitch: int,
                   height: int,
                   width: int,
                   kernel_v: int,
                   kernel_h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Wrapper function for phase coded temporal unwrapping. This function takes masked cosine and stair phase shifted images as input and computes wrapped phase maps. 
    Wrapped phase maps are used to unwrap and obtain absolute phase map which is then further processed to remove spike noise using median filter rectification.

    Parameters
    ----------
    cos_wrap_v = type: float. Cosine fringe wrapped phase map in the horizontal direction
    cos_wrap_h = type: float. Cosine fringe wrapped phase map in the vertical direction
    step_wrap_v = type: float. Stair fringe wrapped phase map in the horizontal direction
    step_wrap_h = type: float. Stair fringe wrapped phase map in the vertical direction
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
    cos_wrap_h = type: float. Wrapped phase map of cosine intensity pattern varying in the vertical direction.
    step_wrap_v = type: float.  Wrapped phase map of stair intensity pattern varying in the horizontal direction.
    step_wrap_h = type: float.  Wrapped phase map of stair intensity pattern varying in the vertical direction.
    """
    # step rectification for border jumps
    step_wrap_v = step_rectification(step_wrap_v, 'v')
    step_wrap_h = step_rectification(step_wrap_h, 'h')
    # Unwrapped
    unwrap_v, k_v = unwrap_cal(step_wrap_v, cos_wrap_v, pitch, width, height, 'v')
    unwrap_h, k_h = unwrap_cal(step_wrap_h, cos_wrap_h, pitch, width, height, 'h')
    
    # Apply median rectification
    fil_unwrap_v, k0_v = filt(unwrap_v, kernel_v, 'v')
    fil_unwrap_h, k0_h = filt(unwrap_h, kernel_h, 'h')
    return fil_unwrap_v, fil_unwrap_h, k0_v, k0_h, cos_wrap_v, cos_wrap_h, step_wrap_v, step_wrap_h  # Filtered unwrapped phase maps and k values

def multi_kunwrap(wavelength: np.array,
                  ph: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function performs temporal phase unwrapping using the low and high wavelength wrapped phase maps.
    Parameters
    ----------
    wavelength: np.ndarray:float.
                Array of wavelengths with decreasing wavelengths (increasing frequencies)
    ph: list:float.
        Array of wrapped phase maps corresponding to decreasing wavelengths (increasing frequencies).

    Returns
    -------
    unwrap: np.ndarray:float.
            Unwrapped phase map
    k: np.ndarray: int.
       Fringe order of the lowest wavelength (the highest frequency)

    """
    k = np.round(((wavelength[0] / wavelength[1]) * ph[0] - ph[1]) / (2 * np.pi))
    unwrap = ph[1] + 2 * np.pi * k
    return unwrap, k

def multifreq_unwrap(wavelength_arr: np.array,
                     phase_arr: np.ndarray,
                     kernel_size: int, 
                     direc: str,
                     mask: np.ndarray,
                     cam_width: int,
                     cam_height: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function performs sequential temporal multi-frequency phase unwrapping from high wavelength (low frequency)
    wrapped phase map to low wavelength (high frequency) wrapped phase map.
    Parameters
    ----------
    wavelength_arr: np.array:float.
                    Wavelengths from high wavelength to low wavelength.
    phase_arr: np.ndarray.
               Wrapped phase maps from high wavelength to low wavelength.
    kernel_size: int
            Filter kernel.
    direc: str
           'v' for vertical or 'h' for horizontal filter
    mask: np.ndarray
            Mask for image recovery.
    cam_width: int
                Camera width
    cam_height: int
                Camera height
    Returns
    -------
    absolute_ph4: np.ndarray:float.
                  The final unwrapped phase map of low wavelength (high frequency) wrapped phase map.
    k4: np.ndarray:int.
        The fringe order of low wavelength (high frequency) phase map.

    """
    absolute_ph, k = multi_kunwrap(wavelength_arr[0:2], phase_arr[0:2])
    for i in range(1, len(wavelength_arr)-1):
        absolute_ph, k = multi_kunwrap(wavelength_arr[i:i+2], [absolute_ph, phase_arr[i+1]]) 
    absolute_ph = recover_image(absolute_ph, mask, cam_height, cam_width)
    absolute_ph, k0 = filt(absolute_ph, kernel_size, direc)
    #mask &= ~np.isnan(absolute_ph) # correction of unwrapped phase map image with nan using median filter creates nan values.
    absolute_ph = absolute_ph[mask]
    return absolute_ph, k

def multiwave_unwrap(wavelength_arr: np.ndarray,
                     phase_arr: np.array,
                     kernel: int,
                     direc: str,
                     mask: np.ndarray, 
                     cam_width: int, 
                     cam_height: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function performs sequential temporal multi wavelength phase unwrapping from high wavelength and applies median filter rectification to remove artifacts.

    Parameters
    ----------
    wavelength_arr = type:array of float wavelengths from high wavelength to low wavelength.
    phase_arr = type: array of float wrapped phase maps from high wavelength to low wavelength.
    kernel type: int. Kernel size for median filter to be applied.
    direc = type: string. vertical (v) or horizontal(h) pattern.

    Returns
    -------
    absolute_ph1 = type:float. Absolute unwrapped phase map.
    k1 = type: int. The fringe order of low wavelength (high frequency) phase map. 

    """
    
    absolute_ph, k = multi_kunwrap(wavelength_arr[0:2], phase_arr[0:2])
    absolute_ph = recover_image(absolute_ph, mask, cam_height, cam_width)
    absolute_ph, k0 = filt(absolute_ph, kernel, direc)
    absolute_ph = absolute_ph[mask]
    for i in range(1, len(wavelength_arr)-1):
        absolute_ph, k = multi_kunwrap(wavelength_arr[i:i+2], [absolute_ph, phase_arr[i+1]])
    absolute_ph = recover_image(absolute_ph, mask, cam_height, cam_width)
    absolute_ph, k0 = filt(absolute_ph, kernel, direc)
    absolute_ph = absolute_ph[mask]    
    return absolute_ph, k

def edge_rectification(multi_phase_123: np.ndarray,
                       direc: str) -> np.ndarray:
    """
    Function to rectify abnormal phase jumps at the edge of multi wavelength high wavelength wrapped phase map.

    Parameters
    ----------
    multi_phase_123 = type: float.  Highest wavelength wrapped phase map in multi wavelength temporal unwrapping algorithm.
    direc = type: string. vertical (v) or horizontal(h) pattern.

    Returns
    -------
    multi_phase_123 = type: float. Rectified phase map.

    """
    img_height = multi_phase_123.shape[1]
    img_width = multi_phase_123.shape[0]
    if direc == 'v':
        multi_phase_123[:, 0:int(img_width/2)][multi_phase_123[:, 0:int(img_width/2)] > 1.5 * np.pi] = multi_phase_123[:, 0:int(img_width/2)][multi_phase_123[:, 0:int(img_width/2)] > 1.5 * np.pi] - 2 * np.pi
        multi_phase_123[:, int(img_width/2):][multi_phase_123[:, int(img_width/2):] < -1.5 * np.pi] = multi_phase_123[:, int(img_width/2):][multi_phase_123[:, int(img_width/2):] < -1.5 * np.pi] + 2 * np.pi
    elif direc == 'h':
        multi_phase_123[0:int(img_height/2)][multi_phase_123[0:int(img_height/2)] > 1.5 * np.pi] = multi_phase_123[0:int(img_height/2)][multi_phase_123[0:int(img_height/2)] > 1.5 * np.pi] - 2 * np.pi
        multi_phase_123[int(img_height/2):][multi_phase_123[int(img_height/2):] < -1.5 * np.pi] = multi_phase_123[int(img_height/2):][multi_phase_123[int(img_height/2):] < -1.5 * np.pi] + 2 * np.pi
    return multi_phase_123

def bilinear_interpolate(unwrap, x, y):
    """
    Function to perform bi-linear interpolation to obtain subpixel circle center phase values.

    Parameters
    ----------
    unwrap = type:float. Absolute phase map
    center = type:float. Subpixel coordinate from OpenCV circle center detection.

    Returns
    -------
   Subpixel mapped absolute phase value corresponding to given circle center. 

    """
    # neighbours
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1
    unwrap_a = unwrap[y0, x0]
    unwrap_b = unwrap[y1, x0]
    unwrap_c = unwrap[y0, x1]
    unwrap_d = unwrap[y1, x1]
    # weights
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*unwrap_a + wb*unwrap_b + wc*unwrap_c + wd*unwrap_d

def undistort(image, camera_mtx, camera_dist): # image with nan values after undistorting and applying interpolation creates nan values
    u = np.arange(0, image.shape[1])
    v = np.arange(0, image.shape[0])
    uc, vc = np.meshgrid(u, v)
    x = (uc - camera_mtx[0, 2])/camera_mtx[0, 0]
    y = (vc - camera_mtx[1, 2])/camera_mtx[1, 1]
    r_sq = x**2 + y**2
    x_double_dash = x*(1 + camera_dist[0, 0] * r_sq + camera_dist[0, 1] * r_sq**2)
    y_double_dash = y*(1 + camera_dist[0, 0] * r_sq + camera_dist[0, 1] * r_sq**2)
    map_x = x_double_dash * camera_mtx[0, 0] + camera_mtx[0, 2]
    map_y = y_double_dash * camera_mtx[1, 1] + camera_mtx[1, 2]
    undist_image = bilinear_interpolate(image, map_x, map_y)
    return undist_image
# =====================================================
# For diagnosis
# Removing trend
# Calculation of coefficients
def fit_trend(filter_img, x_grid, y_grid):
    filter_img_flat = filter_img.flatten()[:, np.newaxis]
    x_row = x_grid.flatten()
    y_row = y_grid.flatten()
    one_row = np.ones(len(x_row))
    xy_array = np.array([one_row, x_row, y_row]).T
    coeff = np.linalg.inv(xy_array.T@xy_array) @ xy_array.T @ filter_img_flat
    return coeff

def trend(x_grid, y_grid, coeff):
    x_row = x_grid.flatten()
    y_row = y_grid.flatten()
    one_row = np.ones(len(x_row))
    xy_arr = np.array([one_row, x_row, y_row]).T
    phi_col = xy_arr@coeff
    return phi_col.reshape(x_grid.shape)

def main():
    test_limit = 0.9
    pitch_list = [50, 20]
    N_list = [3, 3]
    cam_width = 128
    cam_height = 128
    fringe_arr_np = np.load("test_data/toy_data.npy")
    with open(r'test_data\vertical_fringes_np.pickle', 'rb') as f:
        vertical_fringes = pickle.load(f)
    with open(r'test_data\horizontal_fringes_np.pickle', 'rb') as f:
        horizontal_fringes = pickle.load(f)
    # testing #1:
    mod_stack, white_stack, phase_map, mask = phase_cal(fringe_arr_np, test_limit, N_list, True)
    phase_np_v = phase_map[::2]
    phase_np_h = phase_map[1::2]
    if phase_np_v.all() == vertical_fringes['phase_map_np_v'].all():
        print('\n All vertical phase maps match')
        multifreq_unwrap_np_v, k_arr_np_v = multifreq_unwrap(pitch_list, phase_np_v, 1, 'v', mask, cam_width, cam_height)
        if multifreq_unwrap_np_v.all() == vertical_fringes['multifreq_unwrap_np_v'].all():
            print('\n Vertical unwrapped phase maps match')
        else:
            print('\n Vertical unwrapped phase map mismatch ')  
    else:
        print('\n Vertical phase map mismatch')
    if phase_np_h.all() == horizontal_fringes['phase_map_np_h'].all():
        print('\n All horizontal phase maps match')
        multifreq_unwrap_np_h, k_arr_np_h = multifreq_unwrap(pitch_list, phase_np_h, 1, 'h', mask, cam_width, cam_height)
        if multifreq_unwrap_np_h.all() == horizontal_fringes['multifreq_unwrap_np_h'].all():
            print('\n Horizontal unwrapped phase maps match')
        else:
            print('\n Horizontal unwrapped phase map mismatch ')  
    else:
        print('\n Horizontal phase map mismatch')

    return 


if __name__ == '__main__':
    main()
