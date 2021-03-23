import numpy as np
from scipy.special import roots_genlaguerre
from math import gamma

def gauss_laguerre_quadrature(N, n=2, sum_axis=2):
    def decorator_func(func):
        def wrap_func(*args, **kwargs):
            r, w = roots_genlaguerre(N, n)
            r = r.reshape(1,-1)
            return np.sum(w * func(r, *args, **kwargs), axis=sum_axis)
        return wrap_func
    return decorator_func
