# coding: utf-8

import numpy as np
import scipy.ndimage
import os
import cupy as cp
from time import perf_counter_ns
from cupyx.scipy import ndimage


def delta_deck_gen_cp(N: int, height: int, width: int) -> cp.ndarray:
    """
    Function computes phase shift δ  values used in N-step phase shifting algorithm for each image pixel of
    given height and width.
    δ_k  =  (2kπ)/N, where k = 1,2,3,... N and N is the number of steps.

    Parameters
    ----------
    N = type: int. The number of steps in phase shifting algorithm.
    height = type: int. Height of the pattern image.
    width = type: int. Width of pattern image.

    Returns
    -------
    delta_deck_cp = type:cupy.ndarray:float. N delta images. Shape is N x height x width

    Ref: J. H. Brunning, D. R. Herriott, J. E. Gallagher, D. P. Rosenfeld, A. D. White, and D. J. Brangaccio,
    Digital wavefront measuring interferometer for testing optical surfaces, lenses, Appl. Opt. 13(11), 2693–2703, 1974.
    """
    delta_cp = 2 * cp.pi * cp.arange(1, N + 1) / N
    one_block_cp = cp.ones((N, height, width))
    delta_deck_cp = cp.einsum('ijk,i->ijk', one_block_cp, delta_cp)
    return delta_deck_cp

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


if __name__ == '__main__':
    main()
