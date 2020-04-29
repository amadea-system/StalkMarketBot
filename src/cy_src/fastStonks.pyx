"""

"""
# from libc.stdlib cimport malloc, free
import numpy as np
cimport cython
cimport numpy as np
from libc.math cimport trunc, round, floor, ceil

# from cpython cimport array
# import array

DEF RATE_MULTIPLIER = 10000  # Separately defined in stalkMarketPredictions.py

# TODO: Clean this mess up

# region convolve_updf
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
# @cython.infertypes(True)
@cython.initializedcheck(False)
# cpdef double[:] convolve_updf(int a1, int b1, int a2, int b2):
cpdef np.ndarray convolve_updf(int a1, int b1, int a2, int b2):
    """
    Add (convolve) two uniform PDF Functions.
    Where the uniform PDFs take the form:
        U[a1:b1] -> f(y) = 1/(b1-a1) for a1 <= y <= b1
        U[a2:b2] -> f(y) = 1/(b2-a2) for a2 <= y <= b2
        U[a2:b2] -> f(y) = 1/(b2-a2) for a2 <= y <= b2

    In this application a1 & a2 are likely to always be 0, but this code supports other cases.
    """
    #http://docs.cython.org/en/latest/src/userguide/numpy_tutorial.html#numpy-tutorial
    cdef int a1_p_a2 = a1 + a2  # Start
    cdef int b1_p_b2 = b1 + b2  # End
    cdef int length = b1_p_b2 - a1_p_a2  #End - Start
    # cdef array.array convolution = array.array('f', [0 for i in range(length+1)])

    # cdef np.ndarray[double, ndim=1] convolution = np.full(length +1, 1 / (b1 - a1), dtype=float)
    cdef np.ndarray[double, ndim=1] convolution = np.zeros(length +1, dtype=float) #full(length +1, 1 / (b1 - a1), dtype=float)
    # cdef np.ndarray[np.float32_t, ndim=1] convolution = np.zeros(length +1, dtype=np.float32) #full(length +1, 1 / (b1 - a1), dtype=float)

    # cdef double[:] convolution = np.zeros(length + 1, dtype=float)
    cdef float divisor = ((b1-a1)*(b2-a2))

    cdef float temp

    cdef float sum = (1 / (b1 - a1)) * ((a2 + b1) - (a1 + b2))
    # cdef float middle_length = (a2 + b1) - (a1 + b2)

    # cdef double *convolution = <double *> malloc(length + 1 * sizeof(double))
    # if not convolution:
    #     raise MemoryError()

    cdef float middle_section = 1 / (b1 - a1)

    cdef int x  # TODO: Change this to an unsigned int
    if a1 + b2 < a2 + b1:
        # middle_section = 1 / (b1 - a1)
        # convolution = np.full(length+1, middle_section, dtype=float)

        for x in range(0, a1 + b2):
            if a1_p_a2 <= x: # and x <  a1 + b2:
                temp = (x - a1_p_a2)/divisor
                convolution[x] = temp
                sum += temp
            else: #x < a1_p_a2:
                convolution[x] = 0

        # for x in range(length, (a2 + b1-1), -1):
        for x in range((a2 + b1),length+1):

            if x < b1_p_b2:
                temp = ((-x) + b1_p_b2) /divisor
                convolution[x] = temp
                sum += temp
            else:  # b1_p_b2 <= x
                convolution[x] = 0

        # middle_section = 1 / (b1 - a1)
        for x in range(( a1 + b2), (a2 + b1)):
            convolution[x] = middle_section

    # elif a1 + b2 > a2 + b1:
    #     raise MemoryError()
    #     middle_section = 1 / (b2 - a2)
    #     convolution = np.full(length+1, middle_section, dtype=float)
    #
    #     for x in range(length+1):
    #         if x < a1_p_a2:
    #             convolution[x] = 0
    #         elif a1_p_a2 <= x and x < a2 + b1:
    #             convolution[x] = (x - a1_p_a2) / divisor
    #
    #         elif a2 + b1 <= x and x < a1 + b2:
    #             convolution[x] = middle_section
    #
    #         elif a1 + b2 <= x and x < b1_p_b2:
    #             convolution[x] = ((-x) + b1_p_b2) / divisor
    #         else:  # b1_p_b2 <= x
    #             convolution[x] = 0
    # print(f"Fast sum: {fast_sum}, sum: {sum}, np Sum: {np.sum(convolution)}")


    for x in range(0, length+1):
        convolution[x] = convolution[x]/sum

    return convolution#/sum  #np.sum(convolution)
# endregion


@cython.cdivision(True)
cpdef double minimum_rate_from_given_and_base(int given_price, int buy_price):
    return RATE_MULTIPLIER * (given_price - 0.99999) / buy_price


@cython.cdivision(True)
cpdef double maximum_rate_from_given_and_base(int given_price, int buy_price):
    return RATE_MULTIPLIER * (<double>given_price + 0.00001) / <double>buy_price


cpdef (double, double) rate_range_from_given_and_base(int given_price, int buy_price):

    return (
        minimum_rate_from_given_and_base(given_price, buy_price),
        maximum_rate_from_given_and_base(given_price, buy_price)
    )

cpdef int get_price(double rate, int base_price):
    # return intceil(_rate * _base_price / rate_multiplier)
    return <int>trunc(rate * base_price / RATE_MULTIPLIER + 0.99999)


cpdef int c_round(double value):
    return <int>round(value)

cpdef int cjs_round(double value):
    """Replacement for the PY round function that mimics the JS Math.round function"""

    cdef double x
    x = floor(value)
    if (value - x) < .50:
        return <int>x
    else:
        return <int>ceil(value)

# intceil included for reference only. Since get_price is the only thing that calls it, it was implemented directly there.
"""
 def intceil(value) -> int:
     # Function that more closely mimics the ceil function in AC NH
     return math.trunc(value + 0.99999)
"""


# cpdef float[:] arrays(int num):
#     cdef array.array a = array.array('f', [0 for i in range(num)])
#
#     for i in range(num):
#         a[i] = int(i*i)
#
#     return a
