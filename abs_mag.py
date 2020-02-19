#!/usr/bin/env python

from utils import Isochrone
import percent
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
plt.ion()

def run():
    def abs_mag(distance_modulus, mag_g, mag_r, mag_i):
        v = mag_g - 0.487*(mag_g - mag_r) - 0.0249 # Don't know where these numbers come from, copied from ugali
        flux = np.sum(10**(-v/2.5))
        abs_mag = -2.5*np.log10(flux) - distance_modulus
        return abs_mag

    exponents = np.arange(3, 6.05, 0.05)
    n_trials = [int(10**6/10**p)+1 for p in exponents]
    abs_mags = []
    for i, exponent in enumerate(exponents):
        iso = Isochrone(700) # Distance shouldn't matter...
        distance_modulus = iso.distance_modulus

        trials = []
        for j in range(n_trials[i]):
            stellar_mass = 10**exponent
            trials.append(abs_mag(distance_modulus, *iso.simulate(stellar_mass)))
            percent.bar(sum(n_trials[:i])+j, sum(n_trials))

        abs_mags.append(np.mean(trials))

    return exponents, abs_mags

#exponents, abs_mags = run()
#np.save('abs_mag', (exponents, abs_mags))

exponents, abs_mags = np.load('abs_mag.npy')
linear = scipy.stats.linregress(exponents, abs_mags)
print linear
m, b = linear[0], linear[1]

def abs_mag(log_stellar_mass):
    return m*log_stellar_mass + b

plt.scatter(exponents, abs_mags, s=2)
plt.plot(exponents, map(abs_mag, exponents))
plt.gca().invert_yaxis()
