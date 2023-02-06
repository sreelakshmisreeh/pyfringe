# coding: utf-8

import cupy as cp
#from time import perf_counter_ns
from typing import Tuple
from cupyx.scipy import ndimage
import pickle


def level_process_cp(image_stack_cp: cp.ndarray,
                     n: int)->Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """
        Helper function for phase_cal to perform intermediate calculation in each level.
        Parameters
        ----------
        image_stack_cp: cp.ndarray:np.float64.
                        Image stack of levels with same number of patterns (N).
        n: int.
            Number of patterns.
        """
    delta = 2 * cp.pi * cp.arange(1, n + 1) / n
    sin_delta = cp.sin(delta)
    sin_delta[cp.abs(sin_delta) < 1e-15] = 0
    cos_delta = cp.cos(delta)
    cos_delta[cp.abs(cos_delta) < 1e-15] = 0
    sin_deck = cp.einsum('ijkl,j->ikl', image_stack_cp, sin_delta)
    cos_deck = cp.einsum('ijkl,j->ikl', image_stack_cp, cos_delta)
    modulation_deck = 2 * cp.sqrt(sin_deck ** 2 + cos_deck ** 2) / n
    average_deck = cp.sum(image_stack_cp, axis=1) / n
    return sin_deck, cos_deck, modulation_deck, average_deck


def mask_application_cp(mask_cp: cp.ndarray,
                        mod_stack_cp: cp.ndarray,
                        sin_stack_cp: cp.ndarray,
                        cos_stack_cp: cp.ndarray)->Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Helper function to apply mask get relevant pixels for phase calculations.
    """
    num_total_levels = mod_stack_cp.shape[0]
    flag = cp.tile(mask_cp, (num_total_levels, 1, 1))
    sin_stack_cp = sin_stack_cp[flag].reshape((num_total_levels, -1))
    cos_stack_cp = cos_stack_cp[flag].reshape((num_total_levels, -1))
    mod_stack_cp = mod_stack_cp[flag].reshape((num_total_levels, -1))
    return sin_stack_cp, cos_stack_cp, mod_stack_cp


def phase_cal_cp(images_cp: cp.ndarray,
                 limit: float,
                 N: list,
                 calibration: bool) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Function computes phase map for all levels given in list N.
    Parameters
    ----------
    images_cp: cp.ndarray:np.float64.
            Captured fringe images.
    limit: float.
           Data modulation limit. Regions with data modulation lower than limit will be masked out.
    N: list.
        List of number of patterns in each level.
    calibration: bool.
                 If calibration is set the double of N is taken assuming horizontal and vertical fringes.

    Returns
    -------
    mod_stack_cp: cp.ndarray:float.
                  Intensity modulation array(image) for each captured image
    white_stack_cp: cp.ndarray:float.
                    White image from fringe patterns.
    phase_map: cp.ndarray:float.
               Wrapped phase map of each level stacked together after applying mask.
    mask: bool
          Mask applied to image.
    """
    if calibration:
        repeat = 2
    else:
        repeat = 1
    if len(set(N)) == 2:
        image_set1 = images_cp[0:(repeat*len(N)-repeat)*N[0]].reshape((repeat*len(N)-repeat), N[0], images_cp.shape[-2], images_cp.shape[-1])
        image_set2 = images_cp[(repeat*len(N)-repeat)*N[0]:].reshape(repeat, N[-1], images_cp.shape[-2], images_cp.shape[-1])
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
    sin_stack_cp, cos_stack_cp, mod_stack_cp = mask_application_cp(mask_cp, mod_stack_cp, sin_stack_cp, cos_stack_cp)
    phase_map_cp = -cp.arctan2(sin_stack_cp, cos_stack_cp)  # wrapped phase;
    return mod_stack_cp, white_stack_cp, phase_map_cp, mask_cp

def recover_image_cp(vector_array: cp.ndarray,
                     mask: cp.ndarray,
                     cam_height: int,
                     cam_width: int)-> cp.ndarray:
    """
    Function to convert vector to image array using flag.
    vector_array: np.ndarray
                  Vector to be converted
    flag: np.ndarray
          Indexes of the array coordinates with data.
    cam_width: int.
               Width of image.
    cam_height: int.
                Height of image.
    """
    image = cp.full((cam_height, cam_width), cp.nan)
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
    if direc == 'v':
        k = (1, kernel_size)  # kernel size
    elif direc == 'h':
        k = (kernel_size, 1)
    else:
        k = None
        print("ERROR: direction str must be one of {'v', 'h'}")
    med_fil_cp = ndimage.median_filter(unwrap_cp, k)  # not need to do copy, unwrap will not be modified
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
                        phase_arr_cp: cp.ndarray,
                        kernel_size: int,
                        direc: str,
                        mask: cp.ndarray,
                        cam_width: int,
                        cam_height: int) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Function performs sequential temporal multi-frequency phase unwrapping from high wavelength (low frequency)
    wrapped phase map to low wavelength (high frequency) wrapped phase map.
    Parameters
    ----------
    wavelength_arr_cp: list.
                    Wavelengths from high wavelength to low wavelength.
    phase_arr_cp: cp.ndarray.
               Wrapped phase maps from high wavelength to low wavelength.
    kernel_size: int.
            Kernel size for median filter.
    direc: str.
           Vertical (v) or horizontal(h) pattern.
    mask: cp.ndarray
            Mask for image recovery.
    cam_width: int
                Camera width
    cam_height: int
                Camera height
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
    absolute_ph_cp = recover_image_cp(absolute_ph_cp, mask, cam_height, cam_width)
    absolute_ph_cp, k0 = filt_cp(absolute_ph_cp, kernel_size, direc)
    absolute_ph_cp = absolute_ph_cp[mask]
    return absolute_ph_cp, k_array_cp

def bilinear_interpolate_cp(image: cp.ndarray,
                            x: cp.ndarray,
                            y: cp.ndarray)->cp.ndarray:
    """
    Function to perform bi-linear interpolation to obtain subpixel values.

    Parameters
    ----------
    image = cupy.ndarray.
    x = cupy.ndarray.
        X coordinate
    y = cupy.ndarray.
        Y coordinate
    Returns
    -------
    Subpixel mapped absolute value.
    """
    # neighbours
    x0 = cp.floor(x).astype(int)
    x1 = x0 + 1
    y0 = cp.floor(y).astype(int)
    y1 = y0 + 1
    image_a = image[y0, x0]
    image_b = image[y1, x0]
    image_c = image[y0, x1]
    image_d = image[y1, x1]
    # weights
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*image_a + wb*image_b + wc*image_c + wd*image_d

def undistort_cp(image: cp.ndarray,
                 camera_mtx: cp.ndarray,
                 camera_dist: cp.ndarray)->cp.ndarray:
    """
    Function to un distort  an image.
    Parameters
    ----------
    image: cp.ndarray
           Image to apply un distortion
    camera_mtx: cp.ndarray
                Camera intrinsic matrix.
    camera_dist: cp.ndarray
                 Camera distortion matrix.
    Returns
    -------
    undistort_image: cp.ndarray
                  Undistorted image
    """
    u = cp.arange(0, image.shape[1])
    v = cp.arange(0, image.shape[0])
    uc, vc = cp.meshgrid(u, v)
    x = (uc - camera_mtx[0, 2])/camera_mtx[0, 0]
    y = (vc - camera_mtx[1, 2])/camera_mtx[1, 1]
    r_sq = x**2 + y**2
    x_double_dash = x*(1 + camera_dist[0, 0] * r_sq + camera_dist[0, 1] * r_sq**2)
    y_double_dash = y*(1 + camera_dist[0, 0] * r_sq + camera_dist[0, 1] * r_sq**2)
    map_x = x_double_dash * camera_mtx[0, 0] + camera_mtx[0, 2]
    map_y = y_double_dash * camera_mtx[1, 1] + camera_mtx[1, 2]
    undistort_image = bilinear_interpolate_cp(image, map_x, map_y)
    return undistort_image
    
    
# spyder : computing time: 0.026262                         

def main():
    test_limit = 0.9
    pitch_list = [50, 20]
    N_list = [3, 3]
    start = perf_counter_ns()
    cam_width = 128
    cam_height = 128
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
    if phase_v.all() == vertical_fringes['phase_map_cp_v'].all():
        print('\nAll vertical phase maps match')
        multifreq_unwrap_cp_v, k_arr_cp_v = multifreq_unwrap_cp(pitch_list, phase_v, 1, 'v', mask_cp, cam_width, cam_height)
        if multifreq_unwrap_cp_v.all() == vertical_fringes['multifreq_unwrap_cp_v'].all():
            print('\nVertical unwrapped phase maps match')
        else:
            print('\nVertical unwrapped phase map mismatch ')
    else:
        print('\nVertical phase map mismatch')
        
    if phase_h.all() == horizontal_fringes['phase_map_cp_h'].all():
        print('\nAll horizontal phase maps match')
        multifreq_unwrap_cp_h, k_arr_cp_h = multifreq_unwrap_cp(pitch_list, phase_h, 1, 'h', mask_cp, cam_width, cam_height)
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
