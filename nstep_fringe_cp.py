# coding: utf-8

import numpy as np
import cupy as cp
from time import perf_counter_ns
from typing import Tuple
from cupyx.scipy import ndimage

#TODO: write test main to test all functions
#TODO: Generate 128 x 128 test image with margin nan values as test data. Check with manual calculation(ground truth).
#TODO: Compare the results from cupy and numpy
def delta_deck_gen_cp(N: int, height: int, width: int) -> cp.ndarray:
    """
    Function computes phase shift δ  values used in N-step phase shifting algorithm for each image pixel of
    given height and width.
    δ_k  =  (2kπ)/N, where k = 1,2,3,... N and N is the number of steps.

    Parameters
    ----------
    N: int.
       The number of steps in phase shifting algorithm.
    height: int.
            Height of the pattern image.
    width: int.
           Width of pattern image.

    Returns
    -------
    delta_deck_cp: cupy.ndarray:float.
                   N delta images. Shape is N x height x width

    Ref: J. H. Brunning, D. R. Herriott, J. E. Gallagher, D. P. Rosenfeld, A. D. White, and D. J. Brangaccio,
    Digital wavefront measuring interferometer for testing optical surfaces, lenses, Appl. Opt. 13(11), 2693–2703, 1974.
    """
    delta_cp = 2 * cp.pi * cp.arange(1, N + 1) / N
    one_block_cp = cp.ones((N, height, width))
    delta_deck_cp = cp.einsum('ijk,i->ijk', one_block_cp, delta_cp)
    return delta_deck_cp

def phase_cal_cp(images: cp.ndarray, limit: float ) -> Tuple[cp.ndarray,cp.ndarray,cp.ndarray,cp.ndarray]:
    """
    Function that computes and applies mask to captured image based on data modulation (relative modulation) of each pixel
    and computes phase map.
    data modulation = I''(x,y)/I'(x,y).

    Parameters
    ----------
    images: np.ndarray:uint8.
            Captured fringe images.
    limit: float.
           Data modulation limit. Regions with data modulation lower than limit will be masked out.

    Returns
    -------
    masked_img: cupy.ndarray:float.
                Images after applying mask
    modulation :cupy.ndarray:float.
                Intensity modulation array(image) for each captured image
    average_int: cupy.ndarray:float.
                 Average intensity array(image) for each captured image
    phase_map: cupy.ndarray:float.
           Delta values at each pixel for each captured image


    """
    delta_deck = delta_deck_gen_cp(images.shape[0], images.shape[1], images.shape[2])
    N = delta_deck.shape[0]
    images_numpy = images.astype(np.float64)
    images_cupy = cp.asarray(images_numpy) # convert to cupy array
    sin_delta = cp.sin(delta_deck)
    sin_delta[cp.abs(sin_delta) < 1e-15] = 0
    sin_lst = (cp.sum(images_cupy * sin_delta, axis = 0))
    cos_delta = cp.cos(delta_deck)
    cos_delta[cp.abs(cos_delta)<1e-15] = 0
    cos_lst = (cp.sum(images_cupy * cos_delta, axis = 0))
    modulation = 2 * cp.sqrt(sin_lst**2 + cos_lst**2) / N
    average_int = cp.sum(images_cupy, axis = 0) / N
    mask = cp.full(modulation.shape,True)
    mask[modulation > limit] = False
    mask_deck = cp.repeat(mask[cp.newaxis,:,:],images_cupy.shape[0],axis = 0)
    images_cupy[mask_deck]=cp.nan
    #wrapped phase
    sin_lst[mask] = cp.nan
    cos_lst[mask] = cp.nan
    phase_map = -cp.arctan2(sin_lst,cos_lst)# wraped phase;
    return images_cupy, modulation, average_int , phase_map

def filt_cp(unwrap: cp.ndarray , kernel: int ,direc: str)->Tuple[cp.ndarray,cp.ndarray]:
    """
    Function is used to remove artifacts generated in the temporal unwrapped phase map.
    A median filter is applied to locate incorrectly unwrapped points, and those point phase is corrected by adding or
    subtracting a integer number of 2π.

    Parameters
    ----------
    unwrap: cupy.ndarray:float.
            Unwrapped phase map with spike noise.
    kernel: int.
            Kernel size for median filter.
    direc: str.
           Vertical (v) or horizontal(h) pattern.
    Returns
    -------
    correct_unwrap: cupy.ndarray:float.
                    Corrected unwrapped phase map.
    k_array: int.
             Spiking point fringe order.
    """
    dup_img = unwrap.copy()
    if direc == 'v':
        k = (1,kernel) #kernel size
    elif direc == 'h':
        k = (kernel,1)
    med_fil = ndimage.median_filter(dup_img, k)
    k_array = cp.round((dup_img - med_fil) / (2 * cp.pi))
    correct_unwrap = dup_img - (k_array * 2 * cp.pi)
    return correct_unwrap, k_array

def multi_kunwrap_cp(wavelength: cp.array , ph: list)->Tuple[cp.ndarray,cp.ndarray]:
    """
    Function performs temporal phase unwrapping using the low and high wavelngth wrapped phase maps.
    Parameters
    ----------
    wavelength: cupy.ndarray:float.
                Array of wavelengths with decreasing wavelengths (increasing frequencies)
    ph: list:float.
        Array of wrapped phase maps corresponding to decreasing wavelengths (increasing frequencies).

    Returns
    -------
    unwrap: cupy.ndarray:float..
            Unwrapped phase map
    k: int.
       Fringe order of the lowest wavelength (the highest frequency)

    """
    k = cp.round(((wavelength[0] / wavelength[1]) * ph[0] - ph[1])/ (2 * cp.pi))
    unwrap = ph[1] + 2 * cp.pi * k
    return unwrap, k

def multifreq_unwrap_cp(wavelength_arr: cp.array, phase_arr: cp.ndarray, kernel: int, direc: str)-> Tuple[cp.ndarray,cp.ndarray]:
    """
    Function performs sequential temporal multi-frequency phase unwrapping from high wavelength (low frequency)
    wrapped phase map to low wavelength (high frequency) wrapped phase map.
    Parameters
    ----------
    wavelength_arr: cupy.array:float.
                    Wavelengths from high wavelength to low wavelength.
    phase_arr: cupy.ndarray:float.
               Wrapped phase maps from high wavelength to low wavelength.
    Returns
    -------
    absolute_ph4: cupy.ndarray:float.
                  The final unwrapped phase map of low wavelength (high frequency) wrapped phase map.
    k4: cupy.ndarray:int.
        The fringe order of low wavelength (high frequency) phase map.

    """
    absolute_ph,k = multi_kunwrap_cp(wavelength_arr[0:2], phase_arr[0:2])
    for i in range(1,len(wavelength_arr)-1):
        absolute_ph,k = multi_kunwrap_cp(wavelength_arr[i:i+2], [absolute_ph, phase_arr[i+1]])
    absolute_ph, k0 = filt_cp(absolute_ph, kernel, direc)
    return absolute_ph, k

def main():
    N = 3
    height = 1200
    width = 1920

    # testing #1
    start = perf_counter_ns()
    test_cparray = delta_deck_gen_cp(N, height, width)
    end = perf_counter_ns()
    print(test_cparray)
    t = (end - start) / 1e9
    print('time spent: %1.6f s' % t)
    print("array shape: ", test_cparray.shape)


if __name__ == '__main__':
    main()
