import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import kf_book.book_plots as book_plots
from kf_book.book_plots import plot_errorbars


plot_errorbars([(160,8,'A'),(170, 8, 'B')], xlims=(120,200))

# uniform
measurements = np.random.uniform(160, 170, size = 10000)
mean = measurements.mean()
print('Average of measurements is', mean)
plt.figure()
#plt.hist(measurements, 30, density=True)
plt.plot(measurements)

# normal
measurements = np.random.normal(165, 5, size = 10000)
mean = measurements.mean()
print('Average of measurements is', mean)
plt.figure()
#plt.hist(measurements, 30, density=True)
plt.plot(measurements)