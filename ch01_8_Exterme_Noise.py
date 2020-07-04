import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import kf_book.book_plots as book_plots
from kf_book.book_plots import plot_errorbars
import kf_book.gh_internal as gh
from numpy.random import randn


def gen_data(x0, dx, count, noise_factor):
    return [x0 + dx*i + randn()*noise_factor for i in range(count)]

def g_h_filter(data, x0, dx, g, h, dt=1.):
    """
    Performs g-h filter on 1 state variable with a fixed g and h.
    'data' contains the data to be filtered.
    'x0' is the initial value for our state variable
    'dx' is the initial change rate for our state variable
    'g' is the g-h's g scale factor
    'h' is the g-h's h scale factor
    'dt' is the length of the time step
    """
    x_est = x0
    results = []
    for z in data:
        # prediction step
        x_pred = x_est +(dx*dt)
        dx = dx
        
        # update step
        residual = z - x_pred
        dx = dx + h * (residual) / dt
        x_est = x_pred + g * residual
        results.append(x_est)
    return np.array(results)


zs = gen_data(x0=5, dx=2, count=100, noise_factor=100)
data =g_h_filter(data=zs, x0=5, dx=2, dt=1, g=0.2, h=0.02)

gh.plot_g_h_results(measurements=zs, filtered_data=data)

