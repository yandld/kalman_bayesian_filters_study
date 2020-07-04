import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import kf_book.book_plots as book_plots
from kf_book.book_plots import plot_errorbars
import kf_book.gh_internal as gh

weights = [158.0, 164.2, 160.3, 159.9, 162.1, 164.6,
           169.6, 167.4, 166.4, 171.0, 171.2, 172.6]

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
    
book_plots.plot_track([0,11], [160,172], label='Actual weight')
data = g_h_filter(data=weights, x0=160., dx=1., g =6./10, h=1./3, dt=1.)
gh.plot_g_h_results(weights, data)
print(weights)
print(data)


