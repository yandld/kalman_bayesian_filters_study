import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import kf_book.book_plots as book_plots
from kf_book.book_plots import plot_errorbars
import kf_book.gh_internal as gh
from numpy.random import randn

zs =np.linspace(0,1,50)

def g_h_filter(data, x0, dx, g, h, dt=1.):
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

data1 = g_h_filter(data=zs, x0=0, dx=0, dt=1, g=0.2, h=0.05)
data2 = g_h_filter(data=zs, x0=0, dx=2, dt=1, g=0.2, h=0.01)
data3 = g_h_filter(data=zs, x0=0, dx=2, dt=1, g=0.2, h=0.5)

book_plots.plot_measurements(zs)
book_plots.plot_filter(data1, label='dx=0, h=0.05', c='C0')
book_plots.plot_filter(data2, label='dx=2, h= 0.05', c='C1')
book_plots.plot_filter(data3, label='dx=2, h= 0.5', c='C2')
plt.legend(loc=1)


