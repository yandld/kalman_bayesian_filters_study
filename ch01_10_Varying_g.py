import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import kf_book.book_plots as book_plots
from kf_book.book_plots import plot_errorbars
import kf_book.gh_internal as gh
from numpy.random import randn

np.random.seed(100)

def gen_data(x0, dx, count, noise_factor, accel=0.):
    zs = []
    for i in range(count):
        zs.append(x0 + accel * (i**2)/2 + dx*i + randn()*noise_factor)
        dx += accel
    return zs


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

zs = gen_data(x0=5, dx=5, count=50, noise_factor=50)
#data1 = g_h_filter(data=zs, x0=0, dx=5, dt=1, g=0.1, h=0.01)
#data2 = g_h_filter(data=zs, x0=0, dx=5, dt=1, g=0.4, h=0.01)
#data3 = g_h_filter(data=zs, x0=0, dx=5, dt=1, g=0.8, h=0.01)
#
#book_plots.plot_measurements(zs, color='k')
#book_plots.plot_filter(data1, label='g=0.1', marker='s', c='C0')
#book_plots.plot_filter(data2, label='g=0.4', marker='v', c='C1')
#book_plots.plot_filter(data3, label='g=0.8', c='C2')
#plt.legend(loc=4)

zs = [5,6,7,8,9,10,11,12,13,14]

for i in range(50):
    zs.append(14)

data1 = g_h_filter(data=zs, x0=4, dx=1, dt=1, g=0.1, h=0.01)
data2 = g_h_filter(data=zs, x0=4, dx=1, dt=1, g=0.4, h=0.01)
data3 = g_h_filter(data=zs, x0=4, dx=1, dt=1, g=0.9, h=0.01)
book_plots.plot_measurements(zs, color='k')
book_plots.plot_filter(data1, label='g=0.1', marker='s', c='C0')
book_plots.plot_filter(data2, label='g=0.5', marker='v', c='C1')
book_plots.plot_filter(data3, label='g=0.9', c='C2')
plt.legend(loc=4)






