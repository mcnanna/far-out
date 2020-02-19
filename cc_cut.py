#!/usr/bin/env python
import load_data
from utils import Isochrone, color_cut
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import plot_utils
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
plt.ion()

patch = load_data.Patch()
# For ease of typing:
stars, galaxies = patch.stars, patch.galaxies
center_ra, center_dec = patch.center_ra, patch.center_dec
mag = patch.mag
magerr = patch.magerr

iso = Isochrone(300)

g = iso.mag('g')+iso.distance_modulus
r = iso.mag('r')+iso.distance_modulus
i = iso.mag('i')+iso.distance_modulus
cut = color_cut(g,r,i)

x = g[cut] - r[cut]
y = r[cut] - i[cut]
color = r[cut]

linear = scipy.stats.linregress(x, y)
m, b = linear[0], linear[1]
print m, b
def f(x):
    return m*x + b

thiccness = 0.2
dy = thiccness/np.cos(np.arctan(m))


plt.figure(figsize=(10,8))
plt.xlabel('g - r')
plt.xlim(-0.3, 1.8)
plt.ylabel('r - i')
plt.ylim(-0.2, 0.8)

plt.scatter(stars[mag('g')]-stars[mag('r')], stars[mag('r')]-stars[mag('i')], s=1, marker='.', color='0.3', label='Excluded stars', zorder=0)
plt.plot(x, f(x), label='fit', color='k', zorder=5)
plt.plot(x, f(x)+dy, label='fit', color='k', linestyle='--', zorder=5)
plt.plot(x, f(x)-dy, label='fit', color='k', linestyle='--', zorder=5)

maglim=24.42
#cmap = plot_utils.shiftedColorMap('seismic_r', min(color), max(color), maglim)
#plt.scatter(x, y, c=color, cmap=cmap, label='Isochrone', s=10, zorder=3)
plt.scatter(x, y, c='red', s=10, zorder=3)
#cbar = plt.colorbar(label='Isochrone r mag')
#cbar.ax.invert_yaxis()


