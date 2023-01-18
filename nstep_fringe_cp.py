# coding: utf-8

import cupy as cp
from time import perf_counter_ns
from typing import Tuple
from cupyx.scipy import ndimage

# TODO: write test main to test all functions
# TODO: Generate 128 x 128 test image with margin nan values as test data. Check with manual calculation(ground truth).
# TODO: Compare the results from cupy and numpy
def delta_deck_gen_cp(N: int,
                      height: int,
                      width: int) -> cp.ndarray:
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

def phase_cal_cp(images_cp: cp.ndarray,
                 delta_deck_cp: cp.ndarray,
                 limit: float) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Function that computes and applies mask to captured image based on data modulation_cp (relative modulation_cp) of each pixel
    and computes phase map.
    data modulation_cp = I''(x,y)/I'(x,y).
    Parameters
    ----------
    images_cp: cp.ndarray:cp.float64.
            Captured fringe images. Numpy images must be converted to cupy images first using cupy.asarray()
    delta_deck_cp: cp.ndarray:cp.float64
            phase shift matrix, see function delta_deck_gen_cp() for more details.
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
    N = delta_deck_cp.shape[0]
    sin_delta = cp.sin(delta_deck_cp)
    sin_delta[cp.abs(sin_delta) < 1e-15] = 0
    sin_lst = (cp.sum(images_cp * sin_delta, axis=0))
    cos_delta = cp.cos(delta_deck_cp)
    cos_delta[cp.abs(cos_delta) < 1e-15] = 0
    cos_lst = (cp.sum(images_cp * cos_delta, axis=0))
    modulation_cp = 2 * cp.sqrt(sin_lst**2 + cos_lst**2) / N
    average_int_cp = cp.sum(images_cp, axis=0) / N
    mask = cp.full(modulation_cp.shape, True)
    mask[modulation_cp > limit] = False
    mask_deck = cp.repeat(mask[cp.newaxis, :, :], images_cp.shape[0], axis=0)
    images_cp[mask_deck] = cp.nan
    # wrapped phase
    sin_lst[mask] = cp.nan
    cos_lst[mask] = cp.nan
    phase_map_cp = -cp.arctan2(sin_lst, cos_lst)  # wrapped phase;
    return images_cp, modulation_cp, average_int_cp, phase_map_cp

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

def multi_kunwrap_cp(wavelength_cp: cp.ndarray,
                     ph: list) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Function performs temporal phase unwrapping using the low and high wavelength wrapped phase maps.
    Parameters
    ----------
    wavelength_cp: cupy.ndarray:float.
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

def multifreq_unwrap_cp(wavelength_arr_cp: cp.ndarray,
                        phase_arr_cp: list,
                        kernel_size: int,
                        direc: str) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Function performs sequential temporal multi-frequency phase unwrapping from high wavelength (low frequency)
    wrapped phase map to low wavelength (high frequency) wrapped phase map.
    Parameters
    ----------
    wavelength_arr_cp: cupy.array:float.
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
