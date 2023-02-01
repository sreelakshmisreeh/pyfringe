# coding: utf-8

import cupy as cp
from time import perf_counter_ns
from typing import Tuple
from cupyx.scipy import ndimage
import pickle


def level_process_cp(image_stack, n):
    delta = 2 * cp.pi * cp.arange(1, n + 1) / n
    sin_delta = cp.sin(delta)
    sin_delta[cp.abs(sin_delta) < 1e-15] = 0
    cos_delta = cp.cos(delta)
    cos_delta[cp.abs(cos_delta) < 1e-15] = 0
    sin_deck = cp.einsum('ijkl,j->ikl', image_stack, sin_delta)
    cos_deck = cp.einsum('ijkl,j->ikl', image_stack, cos_delta)
    modulation_deck = 2 * cp.sqrt(sin_deck ** 2 + cos_deck ** 2) / n
    average_deck = cp.sum(image_stack, axis=1) / n
    return sin_deck, cos_deck, modulation_deck, average_deck


def mask_application(mask_cp, mod_stack_cp, sin_stack_cp, cos_stack_cp):
    num_total_levels = mod_stack_cp.shape[0]

    mask_cp = mask_cp.astype('float')
    mask_cp[mask_cp == 0] = cp.nan
    # apply mask to all levels
    mod_stack_cp = cp.einsum("ijk, jk->ijk", mod_stack_cp, mask_cp)
    flag = ~cp.isnan(mod_stack_cp)
    sin_stack_cp = sin_stack_cp[flag].reshpae((num_total_levels, -1))
    cos_stack_cp = cos_stack_cp[flag].reshpae((num_total_levels, -1))
    mod_stack_cp = mod_stack_cp[flag].reshpae((num_total_levels, -1))
    return sin_stack_cp, cos_stack_cp, mod_stack_cp


def phase_cal_cp(images_cp: cp.ndarray,
                 limit: float,
                 N: list,
                 calibration: bool) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
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
    N : list.
        List of number of levels in each level.
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
    if calibration:
        repeat = 2
    else:
        repeat = 1
    if len(set(N)) == 2:
        image_set1 = images_cp[0:(repeat*len(N)-repeat)*N[0]].reshape((repeat*len(N)-repeat), N[0], images_cp.shape[-2], images_cp.shape[-1])
        image_set2 = images_cp[repeat*N[-1]:].reshape(repeat, N[-1], images_cp.shape[-2], images_cp.shape[-1])
        image_set = [image_set1, image_set2]

        mask_cp = cp.full((images_cp.shape[-2], images_cp.shape[-1]), True)
        sin_stack_cp = None
        cos_stack_cp = None
        mod_stack_cp = None
        white_stack_cp = None

        for i, n in enumerate(sorted(set(N))):
            sin_deck, cos_deck, modulation, average_int = level_process_cp(image_set[i], n)

            if i == 0:
                sin_stack_cp = sin_deck
                cos_stack_cp = cos_deck
                mod_stack_cp = modulation
                white_stack_cp = modulation + average_int
            else:
                sin_stack_cp = cp.vstack((sin_stack_cp, sin_deck))
                cos_stack_cp = cp.vstack((cos_stack_cp, cos_deck))
                mod_stack_cp = cp.vstack((mod_stack_cp, modulation))
                white_stack_cp = cp.vstack((white_stack_cp, (modulation + average_int)))

            mask_temp = modulation > limit
            mask_cp &= cp.prod(mask_temp, axis=0, dtype=bool)

    else:
        image_set = images_cp.reshape(int(images_cp.shape[0]/N[0]), N[0], images_cp.shape[-2], images_cp.shape[-1])
        sin_stack_cp, cos_stack_cp, mod_stack_cp, average_stack_cp = level_process_cp(image_set, N[0])
        white_stack_cp = mod_stack_cp + average_stack_cp
        mask_cp = mod_stack_cp > limit
        mask_cp = cp.prod(mask_cp, axis=0, dtype=bool)

    sin_stack_cp, cos_stack_cp, mod_stack_cp = mask_application(mask_cp, mod_stack_cp, sin_stack_cp, cos_stack_cp)
    phase_map_cp = -cp.arctan2(sin_stack_cp, cos_stack_cp)  # wrapped phase;
    return mod_stack_cp, white_stack_cp, phase_map_cp, mask_cp

def recover_image_cp(vector_array: cp.ndarray,
                     mask: cp.ndarray,
                     cam_height:int,
                     cam_width:int)-> cp.ndarray:
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
    image[mask] = vector_array
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
                            x: cp.ndarray,
                            y: cp.ndarray)->cp.ndarray:
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

def undistort_cp(image, camera_mtx, camera_dist):
    u = cp.arange(0, image.shape[1])
    v = cp.arange(0, image.shape[0])
    uc, vc = cp.meshgrid(u, v)
    x = (uc - camera_mtx[0,2])/camera_mtx[0,0]
    y = (vc - camera_mtx[1,2])/camera_mtx[1,1]
    r_sq = x**2 + y**2
    x_double_dash = x*(1 + camera_dist[0,0] * r_sq + camera_dist[0,1] * r_sq**2)
    y_double_dash = y*(1 + camera_dist[0,0] * r_sq + camera_dist[0,1] * r_sq**2)
    map_x = x_double_dash * camera_mtx[0,0] + camera_mtx[0,2]
    map_y = y_double_dash * camera_mtx[1,1] + camera_mtx[1,2]
    undist_image = bilinear_interpolate_cp(image, map_x,map_y)
    return undist_image
    
    
# spyder : computing time: 0.026262                         

def main():
    test_limit = 0.9
    pitch_list = [50, 20]
    N_list = [3, 3]
    start = perf_counter_ns()
    fringe_arr_cp = cp.load("test_data/toy_data.npy")
    with open(r'test_data\vertical_fringes_cp.pickle', 'rb') as f:
        vertical_fringes = pickle.load(f)
    with open(r'test_data\horizontal_fringes_cp.pickle', 'rb') as f:
        horizontal_fringes = pickle.load(f)
    end = perf_counter_ns()
    loading_time = (end-start)/1e9
    print('loading time: %2.6f' % loading_time)
    mod_stack_cp, white_stack_cp, phase_map_cp, mask_cp = phase_cal_cp(fringe_arr_cp, test_limit, N_list, True)
    phase_v = phase_map_cp[::2]
    phase_h = phase_map_cp[1::2]
    if (phase_v.all() == vertical_fringes['phase_map_cp_v'].all()):
        print('\nAll vertical phase maps match')
        multifreq_unwrap_cp_v, k_arr_cp_v = multifreq_unwrap_cp(pitch_list, phase_v, 1, 'v')
        if multifreq_unwrap_cp_v.all() == vertical_fringes['multifreq_unwrap_cp_v'].all():
            print('\nVertical unwrapped phase maps match')
        else:
            print('\nVertical unwrapped phase map mismatch ')
    else:
        print('\nVertical phase map mismatch')
        
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
