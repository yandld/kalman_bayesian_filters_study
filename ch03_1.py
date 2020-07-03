import numpy as np
import kf_book.book_plots as book_plots
from kf_book.gaussian_internal import plot_height_std
from filterpy.stats import gaussian
import matplotlib.pyplot as plt
import sys
import math

print(sys.version)

# =============================================================================
# belief = np.array([1,4,2,0,8,2,2,35,4,3,2])
# belief = belief/ np.sum(belief)
# with book_plots.figsize(y=2):
#     book_plots.bar_plot(belief)
# print('sum = ', np.sum(belief))
# =============================================================================

# x = np.array([1.8, 2.0, 1.7, 1.9, 1.6])
# m = x.mean()
# print(m)


#total = 0
#N = 1000000
#for r in np.random.rand(N):
#    if r <= 0.8: total += 1
#    elif r <0.95: total += 3
#    else: total += 5
#
#print(total/N)

#X = [1.8, 2.0, 1.7, 1.9, 1.6]
#Y = [2.2, 1.5, 2.3, 1.7, 1.3]
#Z = [1.8, 1.8, 1.8, 1.8, 1.8]
#print(np.mean(X), np.mean(Y), np.mean(Z))
#print(np.var(X))
#
#plot_height_std(X)
#

#data = 1.8 + np.random.randn(10000)*0.1414
#mean, std = data.mean(), data.std()
#print('mean = {:.3f}'.format(mean))
#print('std = {:.3f}'.format(std))
#print(np.sum((data > mean -std) & (data < mean+std)) / len(data)*100)


#from filterpy.stats import plot_gaussian_pdf
#plot_gaussian_pdf(mean=1.8, variance=0.1414**2, xlabel='Student Height', ylabel='pdf')


#from kf_book.gaussian_internal import display_stddev_plot
#display_stddev_plot()

x = np.arange(-1, 3, 0.01)
g1 = gaussian(x, mean=0.8, var=.1)
g2 = gaussian(x, mean=1.3, var=.2)
plt.plot(x, g1, x, g2)

g = g1 * g2  # element-wise multiplication
g = g / sum(g)  # normalize
plt.plot(x, g, ls='-.');


