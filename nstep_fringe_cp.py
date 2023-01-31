# coding: utf-8

import cupy as cp
from time import perf_counter_ns
from typing import Tuple
from cupyx.scipy import ndimage
import pickle


def phase_cal_cp(images_cp: cp.ndarray,
                 limit: float) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Function that computes and applies mask to captured image based on data modulation_cp (relative modulation_cp) of each pixel
    and computes phase map.
    data modulation_cp = I''(x,y)/I'(x,y).
    Parameters
    ----------
    images_cp: cp.ndarray:cp.float64.
            Captured fringe images. Numpy images must be converted to cupy images first using cupy.asarray()
    limit: float.
           Data modulation_cp limit. Regions with data modulation_cp lower than limit will be masked out.
    Returns
    -------
    masked_img: cupy.ndarray:float.
                Images after applying mask
    modulation_cp :cupy.ndarray:float.
                Intensity modulation_cp array(image) for each captured image
    average_int_cp: cupy.ndarray:float.
                 Average intensity array(image) for each captured image
    phase_map_cp: cupy.ndarray:float.
           Delta values at each pixel for each captured image
    """
    N = images_cp.shape[0]
    delta_cp = 2 * cp.pi * cp.arange(1, N + 1) / N
    sin_delta = cp.sin(delta_cp)
    sin_delta[cp.abs(sin_delta) < 1e-15] = 0
    cos_delta = cp.cos(delta_cp)
    cos_delta[cp.abs(cos_delta) < 1e-15] = 0    
    sin_lst = cp.einsum('kij,k->ij', images_cp, sin_delta)    
    cos_lst = cp.einsum('kij,k->ij', images_cp, cos_delta)
    modulation_cp = 2 * cp.sqrt(sin_lst**2 + cos_lst**2) / N
    average_int_cp = cp.sum(images_cp, axis=0) / N
    mask = modulation_cp > limit    
    white_img = modulation_cp + average_int_cp    
    # wrapped phase
    # sin_lst[mask] = cp.nan
    # cos_lst[mask] = cp.nan
    # phase_map_cp = -cp.arctan2(sin_lst, cos_lst)  # wrapped phase;
    return modulation_cp, white_img, sin_lst, cos_lst, mask

def recover_image_cp(vector_array: cp.ndarray, 
                     flag: cp.ndarray, 
                     cam_height:int,
                     cam_width:int)->Tuple[cp.ndarray]:
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
    image = cp.full((cam_height,cam_width), cp.nan)
    image[flag] = vector_array
    return image

def filt_cp(unwrap_cp: cp.ndarray,
            kernel_size: int,
            direc: str) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Function is used to remove artifacts generated in the temporal unwrapped phase map.
    A median filter is applied to locate incorrectly unwrapped points, and those point phase is corrected by adding or
    subtracting an integer number of 2π.
    Parameters
    ----------
    unwrap_cp: cupy.ndarray:float.
            Unwrapped phase map with spike noise.
    kernel_size: int.
            Kernel size for median filter.
    direc: str.
           Vertical (v) or horizontal(h) pattern.
    Returns
    -------
    correct_unwrap_cp: cupy.ndarray:float.
                    Corrected unwrapped phase map.
    k0_array_cp: int.
             Spiking point fringe order.
    """
    # if direc == 'v':
    #     k = (1, kernel_size)  # kernel size
    # elif direc == 'h':
    #     k = (kernel_size, 1)
    # else:
    #     k = None
    #     print("ERROR: direction str must be one of {'v', 'h'}")
    med_fil_cp = ndimage.median_filter(unwrap_cp, kernel_size)  # not need to do copy, unwrap will not be modified
    k0_array_cp = cp.round((unwrap_cp - med_fil_cp) / (2 * cp.pi))
    correct_unwrap_cp = unwrap_cp - (k0_array_cp * 2 * cp.pi)
    return correct_unwrap_cp, k0_array_cp

def multi_kunwrap_cp(wavelength_cp: list,
                     ph: list) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Function performs temporal phase unwrapping using the low and high wavelength wrapped phase maps.
    Parameters
    ----------
    wavelength_cp: list
                Array of wavelengths with decreasing wavelengths (increasing frequencies)
    ph: list:float.
        Array of wrapped phase maps corresponding to decreasing wavelengths (increasing frequencies).
    Returns
    -------
    unwrap_cp: cupy.ndarray:float..
            Unwrapped phase map
    k_array_cp: int.
       Fringe order of the lowest wavelength (the highest frequency)
    """
    k_array_cp = cp.round(((wavelength_cp[0] / wavelength_cp[1]) * ph[0] - ph[1]) / (2 * cp.pi))
    unwrap_cp = ph[1] + 2 * cp.pi * k_array_cp
    return unwrap_cp, k_array_cp

def multifreq_unwrap_cp(wavelength_arr_cp: list,
                        phase_arr_cp: list,
                        kernel_size: int,
                        direc: str) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Function performs sequential temporal multi-frequency phase unwrapping from high wavelength (low frequency)
    wrapped phase map to low wavelength (high frequency) wrapped phase map.
    Parameters
    ----------
    wavelength_arr_cp: list.
                    Wavelengths from high wavelength to low wavelength.
    phase_arr_cp: list.
               Wrapped phase maps from high wavelength to low wavelength.
    kernel_size: int.
            Kernel size for median filter.
    direc: str.
           Vertical (v) or horizontal(h) pattern.
    Returns
    -------
    absolute_ph4: cupy.ndarray:float.
                  The final unwrapped phase map of low wavelength (high frequency) wrapped phase map.
    k4: cupy.ndarray:int.
        The fringe order of low wavelength (high frequency) phase map.
    """
    absolute_ph_cp, k_array_cp = multi_kunwrap_cp(wavelength_arr_cp[0:2], phase_arr_cp[0:2])
    for i in range(1, len(wavelength_arr_cp) - 1):
        absolute_ph_cp, k_array_cp = multi_kunwrap_cp(wavelength_arr_cp[i:i + 2], [absolute_ph_cp, phase_arr_cp[i + 1]])
    absolute_ph_cp, k0 = filt_cp(absolute_ph_cp, kernel_size, direc)
    return absolute_ph_cp, k_array_cp

def bilinear_interpolate_cp(unwrap: cp.ndarray, 
                            center: cp.ndarray)->cp.ndarray:
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
    if len(center.shape) == 1:
        x = cp.asarray(center[0])
        y = cp.asarray(center[1])
    else:
        x = cp.asarray(center[:, 0])
        y = cp.asarray(center[:, 1])
    # neighbours
    x0 = cp.floor(x).astype(int)
    x1 = x0 + 1
    y0 = cp.floor(y).astype(int)
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

# spyder : computing time: 0.046533                         

def main():
    test_limit = 0.9
    pitch_list = [50, 20]
    start = perf_counter_ns()
    fringe_arr_cp = cp.load("test_data/toy_data.npy")
    with open(r'test_data\vertical_fringes_cp.pickle', 'rb') as f:
        vertical_fringes = pickle.load(f)
    with open(r'test_data\horizontal_fringes_cp.pickle', 'rb') as f:
        horizontal_fringes = pickle.load(f)
    end = perf_counter_ns()
    loading_time = (end-start)/1e9
    print('loading time: %2.6f' % loading_time)
    modulation_cp_v1, white_img_v1, sin_lst_v1, cos_lst_v1, mask_v1 = phase_cal_cp(fringe_arr_cp[0:3], test_limit)
    modulation_cp_v2, white_img_v2, sin_lst_v2, cos_lst_v2, mask_v2 = phase_cal_cp(fringe_arr_cp[6:9], test_limit)
    mask_v = mask_v1 & mask_v2
    flag_v = cp.where(mask_v == True)
    sin_lst_v = cp.array([sin_lst_v1[flag_v], sin_lst_v2[flag_v]])
    cos_lst_v = cp.array([cos_lst_v1[flag_v], cos_lst_v2[flag_v]])
    phase_v = -cp.arctan2(sin_lst_v, cos_lst_v)
    if (phase_v.all() == vertical_fringes['phase_map_cp_v'].all()):
        print('\nAll vertical phase maps match')
        multifreq_unwrap_cp_v, k_arr_cp_v = multifreq_unwrap_cp(pitch_list, phase_v, 1, 'v')
        if multifreq_unwrap_cp_v.all() == vertical_fringes['multifreq_unwrap_cp_v'].all():
            print('\nVertical unwrapped phase maps match')
        else:
            print('\nVertical unwrapped phase map mismatch ')
    else:
        print('\nVertical phase map mismatch')
    modulation_cp_h1, white_img_h1, sin_lst_h1, cos_lst_h1, mask_h1 = phase_cal_cp(fringe_arr_cp[3:6], test_limit)
    modulation_cp_h2, white_img_h2, sin_lst_h2, cos_lst_h2, mask_h2 = phase_cal_cp(fringe_arr_cp[9:12], test_limit)
    mask_h = mask_h1 & mask_h2
    flag_h = cp.where(mask_h == True)
    sin_lst_h = cp.array([sin_lst_h1[flag_h], sin_lst_h2[flag_h]])
    cos_lst_h = cp.array([cos_lst_h1[flag_h], cos_lst_h2[flag_h]])
    phase_h = -cp.arctan2(sin_lst_h, cos_lst_h)
    if (phase_h.all() == horizontal_fringes['phase_map_cp_h'].all()):
        print('\nAll horizontal phase maps match')
        multifreq_unwrap_cp_h, k_arr_cp_h = multifreq_unwrap_cp(pitch_list, phase_h, 1, 'h')
        if multifreq_unwrap_cp_h.all() == horizontal_fringes['multifreq_unwrap_cp_h'].all():
            print('\nHorizontal unwrapped phase maps match')
        else:
            print('\nHorizontal unwrapped phase map mismatch ')
    else:
        print('\nHorizontal phase map mismatch')
    
    end = perf_counter_ns()
    computing_time = (end - start) / 1e9
    print('computing time: %2.6f' % computing_time)
    return


if __name__ == '__main__':
    main()
