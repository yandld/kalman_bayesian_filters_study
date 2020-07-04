import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import kf_book.book_plots as book_plots
from kf_book.book_plots import plot_errorbars
import kf_book.gh_internal as gh

#plot_errorbars([(160,8,'A'),(170, 8, 'B')], xlims=(120,200))
#
## uniform
#measurements = np.random.uniform(160, 170, size = 10000)
#mean = measurements.mean()
#print('Average of measurements is', mean)
#plt.figure()
##plt.hist(measurements, 30, density=True)
#plt.plot(measurements)
#
## normal
#measurements = np.random.normal(165, 5, size = 10000)
#mean = measurements.mean()
#print('Average of measurements is', mean)
#plt.figure()
##plt.hist(measurements, 30, density=True)
#plt.plot(measurements)
#

#gh.plot_hypothesis5()
#gh.plot_estimate_chart_3()


from kf_book.book_plots import figsize
weights = [158.0, 164.2, 160.3, 159.9, 162.1, 164.6,
           169.6, 167.4, 166.4, 171.0, 171.2, 172.6]

time_step = 1.0 #day
scale_factor = 4.0 / 10

def predict_using_gain_guess(estimated_weight, gain_rate, do_print = False):
    # store for the filterd resuts
    estimates, predictions = [estimated_weight], []
    
    for z in weights:
        # predict new position
        predicted_weight = estimated_weight + gain_rate * time_step
        
        # update filter
        estimated_weight = predicted_weight + scale_factor * (z - predicted_weight)
        
        # save and log
        estimates.append(estimated_weight)
        predictions.append(predicted_weight)
        if do_print:
            gh.print_results(estimates, predicted_weight, estimated_weight)
    return estimates, predictions

initial_estimate = 160.
#book_plots.set_figsize(10)

#e, p = predict_using_gain_guess(initial_estimate, 1, True)
#gh.plot_gh_results(weights, e, p, [160, 172])

#
#e, p = predict_using_gain_guess(initial_estimate, -1, True)
#book_plots.set_figsize(10)
#gh.plot_gh_results(weights, e, p, [160, 172])


weight = 160 # inital gauss 
gain_rate = -1.0 # inital guass

time_step = 1.
weight_scale = 4./10
gain_scale = 1./3
estimates = [weight]
predictions = []

for z in weights:
    # prediction step
    weight = weight + gain_rate*time_step
    gain_rate = gain_rate
    predictions.append(weight)
    
    # update
    residual = z - weight
    
    gain_rate = gain_rate + gain_scale * (residual / time_step)

    weight = weight + weight_scale * residual
    estimates.append(weight)
    
gh.plot_gh_results(weights, estimates, predictions, [160, 172])
    



