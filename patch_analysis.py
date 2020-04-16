#!/usr/bin/env python

import argparse
import subprocess
import warnings

import astropy.io.fits as fits
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
import numpy as np
import healpy as hp
import scipy
from scipy.stats import poisson, norm

import ugali.analysis.source
import ugali.analysis.kernel
import ugali.analysis.results
import ugali.utils.healpix
import ugali.utils.projector

import percent
import plot_utils
import load_data
import utils
from isochrone import Isochrone

matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
plt.ion()
NSIDE = 4096


patch = load_data.Patch()
# For ease of typing:
stars, galaxies = patch.stars, patch.galaxies
center_ra, center_dec = patch.center_ra, patch.center_dec
mag = patch.mag
magerr = patch.magerr

def plot_isochrone(distance):
    for band_1, band_2 in ('gr', 'ri'):
        iso = Isochrone(distance)

        # Field
        plt.figure()
        plt.xlabel('{} - {}'.format(band_1.lower(), band_2.lower()))
        if band_1 == 'r':
            plt.xlim(-0.3, 0.6)
        elif band_1 == 'g':
            plt.xlim(-0.4, 1.4)
        plt.ylim(15, 26)
        plt.ylabel(band_1.lower())

        cut_stars = utils.cut(iso, stars[mag('g')], stars[mag('r')], stars[mag('i')], stars[magerr('g')], stars[magerr('r')], stars[magerr('i')])
        plt.scatter(stars[~cut_stars][mag(band_1)] - stars[~cut_stars][mag(band_2)], stars[~cut_stars][mag(band_1)], s=1, color='0.75', label='Excluded stars', zorder=0)
        plt.scatter(stars[cut_stars][mag(band_1)] - stars[cut_stars][mag(band_2)], stars[cut_stars][mag(band_1)], s=1, color='red', label='Included stars', zorder=5)
        plt.legend(markerscale=5.0)

        # Isochrone
        #iso.draw(band_1, band_2, cookie=True) # color='k', alpha=0.5, zorder=3)
        iso.draw(band_1, band_2, cookie=False, zorder=3)

        ax = plt.gca()
        ax.invert_yaxis()

        title = '$D = {}$ kpc'.format(distance)
        plt.title(title)
        outdir = 'isochrone_plots/{}-{}/'.format(band_1,band_2)
        outname = 'iso_cmd_{}kpc'.format(distance)
        subprocess.call('mkdir -p {}'.format(outdir).split())
        plt.savefig(outdir + outname + '.png')
        plt.close()    

def plot_color_color(distance, tol=0.2):
    iso = Isochrone(distance)
    cut_stars = utils.cut(iso, stars[mag('g')], stars[mag('r')], stars[mag('i')], stars[magerr('g')], stars[magerr('r')], stars[magerr('i')], color_tol=tol)

    plt.figure()
    plt.xlabel('g - r')
    plt.xlim(-0.3, 1.8)
    plt.ylabel('r - i')
    plt.ylim(-0.2, 0.8)

    plt.scatter(stars[~cut_stars][mag('g')]-stars[~cut_stars][mag('r')], stars[~cut_stars][mag('r')]-stars[~cut_stars][mag('i')], s=1, marker='.', color='0.75', label='Excluded stars', zorder=0)
    plt.scatter(stars[cut_stars][mag('g')]-stars[cut_stars][mag('r')], stars[cut_stars][mag('r')]-stars[cut_stars][mag('i')], s=1, marker='.', color='red', label='Included stars', zorder=5)
    plt.legend(markerscale=5.0)

    # Isochrone
    color = iso.mag('r')
    maglim = 24.42
    cmap = plot_utils.shiftedColorMap('seismic_r', min(color), max(color), maglim)
    plt.scatter(iso.mag('g')-iso.mag('r'), iso.mag('r')-iso.mag('i'), c=color, cmap=cmap, label='Isochrone', zorder=10, s=10)
    cbar = plt.colorbar(label='Isochrone r mag')
    cbar.ax.invert_yaxis()

    # Plot box
    # Parallels
    x = np.linspace(-0.3,1.8,100)
    m, b = 0.369485, -0.0055077
    dx = tol*m
    dy = tol/(np.cos(np.arctan(m)))
    #plt.plot(x, f(x)
    plt.plot(x, m*x+b+dy, color='k', linestyle='--', linewidth=1, zorder=20)
    plt.plot(x, m*x+b-dy, color='k', linestyle='--', linewidth=1, zorder=20)
    # Perpendiculars
    #for x0 in min(x), max(x):
    #    xvals = np.linspace(x0-dx, x0+dx, 100)
    #    plt.plot(xvals, -1/m*(xvals-x0)+m*x0+b, color='k', zorder=20)
    #xmin, xmax = min(x), max(x)

    title = '$D = {}$ kpc'.format(distance)
    plt.title(title)
    outname = 'color_color_{}kpc'.format(distance)
    outdir = 'color_color_plots/'
    subprocess.call('mkdir -p {}'.format(outdir).split())
    plt.savefig(outdir + outname + '.png')
    plt.close()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # Quick plots
    p.add_argument('--cc', action='store_true')
    p.add_argument('--iso', action='store_true')

    args = p.parse_args()

    distances = np.arange(200, 2100, 100)
    for distance in distances:
        if args.cc:
            plot_color_color(distance)
        elif args.iso:
            plot_isochrone(distance)
        else: # If no argument, assume user wants both
            plot_color_color(distance)
            plot_isochrone(distance)

