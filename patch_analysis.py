#!/usr/bin/env python

import argparse
import subprocess
import astropy.io.fits as fits
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import healpy as hp
import scipy
import ugali.analysis.source
import ugali.analysis.kernel
import ugali.analysis.results
import ugali.utils.healpix
import ugali.utils.projector
import percent
import plot_utils
from scipy.stats import poisson, norm
import load_data
import warnings
from utils import *

matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
plt.ion()


p = argparse.ArgumentParser()
# Quick plots
p.add_argument('--cc', action='store_true')
p.add_argument('--iso', action='store_true')
# Sim and plot a single satellite
p.add_argument('--sim', action='store_true')
p.add_argument('-d', '--distance', type=float, help='kpc')
p.add_argument('-r', '--r_physical', type=float, help='pc')
p.add_argument('-m', '--abs_mag', type=float, help='mag')
# Scan parameters space
p.add_argument('--scan', action='store_true')
p.add_argument('--plots', action='store_true')
# Extra
p.add_argument('--main', action='store_true')
args = p.parse_args()

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

        cut_color = color_cut(stars[mag('g')], stars[mag('r')], stars[mag('i')])
        cut_iso = iso_cut(iso, band_1, stars[mag(band_1)], band_2, stars[mag(band_2)])
        cut_field = cut_color & cut_iso
        plt.scatter(stars[~cut_field][mag(band_1)] - stars[~cut_field][mag(band_2)], stars[~cut_field][mag(band_1)], s=1, color='0.75', label='Excluded stars', zorder=0)
        plt.scatter(stars[cut_field][mag(band_1)] - stars[cut_field][mag(band_2)], stars[cut_field][mag(band_1)], s=1, color='red', label='Included stars', zorder=5)
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

def plot_color_color(distance):
    plt.figure()
    plt.xlabel('g - r')
    plt.xlim(-0.3, 1.8)
    plt.ylabel('r - i')
    plt.ylim(-0.2, 0.8)

    cut_field = color_cut(stars[mag('g')], stars[mag('r')], stars[mag('i')])
    plt.scatter(stars[~cut_field][mag('g')]-stars[~cut_field][mag('r')], stars[~cut_field][mag('r')]-stars[~cut_field][mag('i')], s=1, marker='.', color='0.75', label='Excluded stars', zorder=0)
    plt.scatter(stars[cut_field][mag('g')]-stars[cut_field][mag('r')], stars[cut_field][mag('r')]-stars[cut_field][mag('i')], s=1, marker='.', color='red', label='Included stars', zorder=5)
    plt.legend(markerscale=5.0)

    # Isochrone
    iso = Isochrone(distance)
    color = iso.data['r']+iso.distance_modulus
    maglim = 24.42
    cmap = plot_utils.shiftedColorMap('seismic_r', min(color), max(color), maglim)
    plt.scatter(iso.data['g']-iso.data['r'], iso.data['r']-iso.data['i'], c=color, cmap=cmap, label='Isochrone', zorder=10, s=10)
    cbar = plt.colorbar(label='Isochrone r mag')
    cbar.ax.invert_yaxis()

    title = '$D = {}$ kpc'.format(distance)
    plt.title(title)
    outname = 'color_color_{}kpc'.format(distance)
    outdir = 'color_color_plots/'
    subprocess.call('mkdir -p {}'.format(outdir).split())
    plt.savefig(outdir + outname + '.png')
    plt.close()


def simSatellite(inputs, lon_centroid, lat_centroid, distance, abs_mag, r_physical): 
    # Stolen from ugali/scratch/simulation/simulate_population.py. Look there for a more general functioon,
    # which uses maglims, extinction, stuff like that
    """
    r_physical is azimuthally averaged half-light radius, kpc
    """

    # Probably don't want to parse every time
    s = ugali.analysis.source.Source()

    # Following McConnachie 2012, ellipticity = 1 - (b/a) , where a is semi-major axis and b is semi-minor axis
    r_h = np.degrees(np.arcsin(r_physical / distance)) # Azimuthally averaged half-light radius
    # See http://iopscience.iop.org/article/10.3847/1538-4357/833/2/167/pdf
    # Based loosely on https://arxiv.org/abs/0805.2945
    ellipticity = 0.3 #np.random.uniform(0.1, 0.8)
    position_angle = np.random.uniform(0., 180.) # Random position angle (deg)
    a_h = r_h / np.sqrt(1. - ellipticity) # semi-major axis (deg)
    
        
    # Elliptical kernels take the "extension" as the semi-major axis
    ker = ugali.analysis.kernel.EllipticalPlummer(lon=lon_centroid, lat=lat_centroid, ellipticity=ellipticity, position_angle=position_angle)

    flag_too_extended = False
    if a_h >= 1.0:
        print('Too extended: a_h = %.2f'%(a_h))
        a_h = 1.0
        flag_too_extended = True
        raise Exception('flag_too_extended')
    ker.setp('extension', value=a_h, bounds=[0.0,1.0])
    s.set_kernel(ker)
    
    distance_modulus = ugali.utils.projector.distanceToDistanceModulus(distance)
    iso = Isochrone(distance)
    s.set_isochrone(iso)
    # Simulate takes stellar mass as an argument, NOT richness
    mag_g, mag_r, mag_i = s.isochrone.simulate(abs_mag) 

    lon, lat = s.kernel.sample_lonlat(len(mag_r))

    # Depth maps
    nside = hp.npix2nside(len(inputs.m_maglim_g)) # Assuming that the maglim maps have same resolution
    pix = ugali.utils.healpix.angToPix(nside, lon, lat)
    maglim_g = inputs.m_maglim_g[pix]
    maglim_r = inputs.m_maglim_r[pix]
    maglim_i = inputs.m_maglim_i[pix]

    # Extintion
    # DES Y3 Gold fiducial
    nside = hp.npix2nside(len(inputs.m_ebv))
    pix = ugali.utils.healpix.angToPix(nside, lon,lat)
    ext = {'g':3.186, 'r':2.140, 'i':1.569}
    mag_extinction_g = ext['g'] * inputs.m_ebv[pix]
    mag_extinction_r = ext['r'] * inputs.m_ebv[pix]
    mag_extinction_i = ext['i'] * inputs.m_ebv[pix]

    
    # Photometric uncertainties are larger in the presence of interstellar dust reddening
    mag_g_error = 0.01 + 10**(inputs.log_photo_error((mag_g + mag_extinction_g) - maglim_g))
    mag_r_error = 0.01 + 10**(inputs.log_photo_error((mag_r + mag_extinction_r) - maglim_r))
    mag_i_error = 0.01 + 10**(inputs.log_photo_error((mag_i + mag_extinction_i) - maglim_i))

    flux_g_meas = magToFlux(mag_g) + np.random.normal(scale=getFluxError(mag_g, mag_g_error))
    mag_g_meas = np.where(flux_g_meas > 0., fluxToMag(flux_g_meas), 99.)
    flux_r_meas = magToFlux(mag_r) + np.random.normal(scale=getFluxError(mag_r, mag_r_error))
    mag_r_meas = np.where(flux_r_meas > 0., fluxToMag(flux_r_meas), 99.)
    flux_i_meas = magToFlux(mag_i) + np.random.normal(scale=getFluxError(mag_i, mag_i_error))
    mag_i_meas = np.where(flux_i_meas > 0., fluxToMag(flux_i_meas), 99.)

    # Includes penalty for interstellar extinction and also include variations in depth
    # Use r band:
    # 24.42 is the median magnitude limit from Y6 according to Keith's slack message
    cut_detect = (np.random.uniform(size=len(mag_r)) < inputs.completeness(mag_r + mag_extinction_r + (24.42 - np.clip(maglim_r, 20., 26.))))

    # Absoulte Magnitude
    v = mag_g - 0.487*(mag_g - mag_r) - 0.0249 # Don't know where these numbers come from, copied from ugali
    flux = np.sum(10**(-v/2.5))
    abs_mag_realized = -2.5*np.log10(flux) - distance_modulus

    r_physical = distance * np.tan(np.radians(r_h)) # Azimuthally averaged half-light radius, kpc
    surface_brightness_realized = ugali.analysis.results.surfaceBrightness(abs_mag_realized, r_physical, distance) # Average within azimuthally averaged half-light radius

    return lon[cut_detect], lat[cut_detect], mag_g_meas[cut_detect], mag_r_meas[cut_detect], mag_i_meas[cut_detect], a_h, ellipticity, position_angle, abs_mag_realized, surface_brightness_realized, flag_too_extended


def calc_sigma(inputs, distance, abs_mag, r_physical, plot=False):
    lon, lat, mag_g, mag_r, mag_i, a_h, ellipticity, position_angle, abs_mag_realized, surface_brightness_realized, flag_too_extended = simSatellite(inputs, center_ra, center_dec, distance, abs_mag, r_physical)

    iso = Isochrone(distance)
    cut_sat  = iso_cut(iso, 'g', mag_g, 'r', mag_r)
    cut_sat &= iso_cut(iso ,'r', mag_r, 'i', mag_i)
    cut_sat &= color_cut(mag_g, mag_r, mag_i)

    # Apply isochrone and color cut to field stars
    cut_field  = iso_cut(iso, 'g', stars[mag('g')], 'r', stars[mag('r')])
    cut_field &= iso_cut(iso, 'r', stars[mag('r')], 'i', stars[mag('i')])
    cut_field &= color_cut(stars[mag('g')], stars[mag('r')], stars[mag('i')])
    
    ### Significance
    theta = 90-position_angle
    a = a_h
    b = a*(1-ellipticity)

    sat_in_ellipse = ((lon[cut_sat]-center_ra)*np.cos(theta) + (lat[cut_sat]-center_dec)*np.sin(theta))**2/a**2 + \
                     ((lon[cut_sat]-center_ra)*np.sin(theta) - (lat[cut_sat]-center_dec)*np.cos(theta))**2/b**2 <= 1
    field_ras = stars[cut_field]['RA']
    field_decs = stars[cut_field]['DEC']
    field_in_ellipse = ((field_ras-center_ra)*np.cos(theta) + (field_decs-center_dec)*np.sin(theta))**2/a**2 + \
                       ((field_ras-center_ra)*np.sin(theta) - (field_decs-center_dec)*np.cos(theta))**2/b**2 <= 1
    signal = sum(sat_in_ellipse) + sum(field_in_ellipse)

    rho_field = len(stars[cut_field])/1.0
    background = rho_field * np.pi*a*b

    sigma = min(norm.isf(poisson.sf(signal, background)), 38.0)

    if plot:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
        title = '$D = {}$ kpc, $M_V = {}$ $M_\odot$, $r_{{1/2}} = {}$ pc\n$\sigma = {}$'.format(int(round(distance,0)), round(abs_mag_realized, 1), int(round(r_physical*1000, 0)), round(sigma, 1))
        fig.suptitle(title)

        ### CMD plots
        # g - r
        ax = axes[0][0]
        plt.sca(ax)

        plt.scatter(mag_g[cut_sat] - mag_r[cut_sat], mag_g[cut_sat], s=6, c='black', label='Included satellite stars', zorder=10)
        plt.scatter(mag_g[~cut_sat] - mag_r[~cut_sat], mag_g[~cut_sat], s=2, c='0.4', label='Excluded satellite stars', zorder=9)
        plt.scatter(stars[cut_field][mag('g')]-stars[cut_field][mag('r')], stars[cut_field][mag('g')], s=4, color='red', label='Included field stars', zorder=5)
        plt.scatter(stars[~cut_field][mag('g')]-stars[~cut_field][mag('r')], stars[~cut_field][mag('g')], s=0.5, color='coral', label='Excluded field stars', zorder=0)
        plt.xlim(0, 1.2)
        plt.xlabel('g - r')
        plt.ylim(min(stars[cut_field][mag('g')])-1.0, max(stars[cut_field][mag('g')])+1.0)
        plt.ylabel('g')
        ax.invert_yaxis()
        
        # r - i
        ax = axes[0][1]
        plt.sca(ax)

        plt.scatter(mag_r[cut_sat] - mag_i[cut_sat], mag_r[cut_sat], s=6, c='black', label='Included satellite stars', zorder=10)
        plt.scatter(mag_r[~cut_sat] - mag_i[~cut_sat], mag_r[~cut_sat], s=2, c='0.4', label='Excluded satellite stars', zorder=9)
        plt.scatter(stars[cut_field][mag('r')]-stars[cut_field][mag('i')], stars[cut_field][mag('r')], s=4, color='red', label='Included field stars', zorder=5)
        plt.scatter(stars[~cut_field][mag('r')]-stars[~cut_field][mag('i')], stars[~cut_field][mag('r')], s=0.5, color='coral', label='Excluded field stars', zorder=0)
        plt.xlim(-0.3, 0.7)
        plt.xlabel('r - i')
        plt.ylim(min(stars[cut_field][mag('r')])-1.0, max(stars[cut_field][mag('r')])+1.0)
        plt.ylabel('r')
        ax.invert_yaxis()

        ### Color-color plot
        ax = axes[1][0]
        plt.sca(ax)

        plt.xlabel('g - r')
        plt.xlim(0, 1.2)
        plt.ylabel('r - i')
        plt.ylim(-0.3, 0.7)

        plt.scatter(mag_g[cut_sat] - mag_r[cut_sat], mag_r[cut_sat] - mag_i[cut_sat], s=6, c='black', label='Included satellite stars', zorder=10)
        plt.scatter(mag_g[~cut_sat] - mag_r[~cut_sat], mag_r[~cut_sat] - mag_i[~cut_sat], s=2, c='0.4', label='Excluded satellite stars', zorder=9)
        plt.scatter(stars[cut_field][mag('g')]-stars[cut_field][mag('r')], stars[cut_field][mag('r')]-stars[cut_field][mag('i')], s=4, marker='.', color='red', label='Included stars', zorder=5)
        plt.scatter(stars[~cut_field][mag('g')]-stars[~cut_field][mag('r')], stars[~cut_field][mag('r')]-stars[~cut_field][mag('i')], s=0.5, marker='.', color='coral', label='Excluded stars', zorder=0)

        ### Spatial plot
        ax = axes[1][1]
        plt.sca(ax)

        plt.scatter(lon[cut_sat]-center_ra, lat[cut_sat]-center_dec, s=6, color='black', label='Included satellite stars')
        plt.scatter(lon[~cut_sat]-center_ra, lat[~cut_sat]-center_dec, s=2, color='0.3', label='Excluded satellite stars')
        plt.scatter(stars[cut_field]['RA']-center_ra, stars[cut_field]['DEC']-center_dec, s=4, color='red', label='Included field stars')
        plt.scatter(stars[~cut_field]['RA']-center_ra, stars[~cut_field]['DEC']-center_dec, s=0.5, color='coral', label='Excluded field stars')

        ellipse = Ellipse(xy=(0,0), width=2*a_h, height=2*(1-ellipticity)*a_h, angle=90-position_angle, edgecolor='green', linewidth=1.5, fill=False, label="$a_h = {}'$".format(round(a_h*60, 1)))
        big_ellipse = Ellipse(xy=(0,0), width=2*3*a_h, height=2*3*((1-ellipticity)*a_h), angle=90-position_angle, edgecolor='green', linewidth=1.5, linestyle='--', fill=False, label='$3 a_h$')
        ax.add_patch(ellipse)
        ax.add_patch(big_ellipse)
        plt.legend((ellipse, big_ellipse), ("$a_h = {}'$".format(round(a_h*60, 1)), '$3 a_h$'))

        plt.xlim(-5*a_h, 5*a_h)
        plt.ylim(-5*a_h, 5*a_h)
        plt.xlabel('$\Delta$RA (deg)')
        plt.ylabel('$\Delta$DEC (deg)')

        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, markerscale=2.0)

        outname = 'D={}_M={}_r={}'.format(int(distance), round(abs_mag,1), int(round(r_physical*1000,0)))
        outdir = 'sat_plots/'
        subprocess.call('mkdir -p {}'.format(outdir).split())
        plt.savefig(outdir + outname + '.png')
        plt.close()

    #params = {'abs_mag_realized':abs_mag_realized, 'surface_brightness_realized':surface_brightness_realized} # Could add more later if needed
    return sigma


def create_sigma_matrix(distances, abs_mags, r_physical_kpcs, outname='sigma_matrix'):
    n_d = len(distances)
    n_m = len(abs_mags)
    n_r = len(r_physical_kpcs)
    inputs = load_data.Inputs()

    sigma_matrix = np.zeros((n_d, n_m, n_r))
    sigma_fits = []
    for i in range(n_d):
        for j in range(n_m):
            for k in range(n_r):
                d, m, r = distances[i], abs_mags[j], r_physical_kpcs[k]
                sigma = calc_sigma(inputs, d, m, r, plot=False)

                sigma_matrix[i,j,k] = sigma
                sigma_fits.append((d, m, r, sigma))

                percent.bar(i*n_m*n_r + j*n_r + k + 1, n_d*n_m*n_r)

    np.save(outname+'.npy', sigma_matrix) # Not used but I feel like I might as well make it

    dtype = [('distance',float), ('abs_mag',float), ('r_physical',float), ('sigma',float)]
    sigma_fits = np.array(sigma_fits, dtype=dtype)
    fits.writeto(outname+'.fits', sigma_fits, overwrite=True)



def plot_matrix(fname, *args, **kwargs):
    dic = {'distance': {'label':'$D$', 'conversion': lambda d:int(d), 'unit':'kpc', 'scale':'linear', 'reverse':False}, 
           'abs_mag': {'label':'$M_V$', 'conversion': lambda v:round(v,1), 'unit':'mag', 'scale':'linear', 'reverse':True},
           'r_physical': {'label':'$r_{1/2}$', 'conversion': lambda r:int(round(r*1000,0)), 'unit':'pc', 'scale':'log', 'reverse':False},
           'stellar_mass': {'label':'$M_*$', 'conversion': lambda m: '$10^{{{}}}$'.format(round(np.log10(m),1)), 'unit':'$M_{\odot}$', 'scale':'log', 'reverse':False}
           }
    #dic = {'distance': ('$D$', lambda d: int(d), 'kpc', 'linear'),
    #      'abs_mag': ('$M_V$', lambda v: round(v, 1), 'mag', 'linear'),
    #      'r_physical': ('$r_{1/2}$', lambda r: int(round(r*1000, 0)), 'pc', 'log'),
    #      'stellar_mass': ('$M_*$', lambda m: '$10^{{{}}}$'.format(round(np.log10(m), 1)), '$M_{\odot}$', 'log')}
    def is_near(arr, val, e=0.001):
        return np.array([val-e < a < val+e for a in arr])

    if '.fits' not in fname:
        fname += '.fits'
    if len(args)+len(kwargs) != 3:
        raise ValueError("Error in the number of specified parameters")

    # Cut the table down based on the parameters set in kwargs:
    table = fits.open(fname)[1].data
    cut = np.tile(True, len(table))
    for key in kwargs:
        cut &= is_near(table[key], kwargs[key])
    table = table[cut]
    if len(table) == 0:
        raise Exception('Table is empty after cutting')

    # This block is kind of obsolete, I haven't kept it updated
    if len(args) == 1:
        x, y = args
        plt.plot(table[x], table['sigma'], '-o')
        plt.xlabel('{} ({})'.format(dic[x]['label'], dic[x]['unit']))
        plt.xscale(dic[x]['scale'])
        plt.ylabel('$\sigma$')
        title = ('; '.join(["{} = {} {}".format(dic[key]['label'], dic[key]['conversion'](kwargs[key]), dic[key]['unit']) for key in kwargs]))
        plt.title(title)
        outname = '{}__'.format(x) + '_'.join(['{}={}'.format(key, kwargs[key]) for key in kwargs])
        plt.savefig('mat_plots/1D_' + outname + '.png')
        plt.close()

    elif len(args) == 2:
        x, y = args
        # Turn into matrix
        x_vals = sorted(set(table[x]), reverse=dic[x]['reverse'])
        y_vals = sorted(set(table[y]), reverse=dic[y]['reverse'])
        mat = np.zeros((len(x_vals), len(y_vals)))
        for i, x_val in enumerate(x_vals):
            for j, y_val in enumerate(y_vals):
                line = table[is_near(table[x], x_val) & is_near(table[y], y_val)]
                mat[i,j] = line['sigma']

        plt.figure(figsize=(len(x_vals)/2. + 3,len(y_vals)/2.))
        plt.pcolormesh(mat.T, cmap=plot_utils.shiftedColorMap('seismic_r', np.min(mat), np.max(mat), 6))
        if np.max(mat) < 6:
            warnings.warn("No satellites above detectabiliy threshold of 6 for kwarg{} {} = {}".format('s' if len(kwargs)>1 else '', ', '.join(kwargs.keys()), ', '.join(map(str, kwargs.values()))))
            return
        elif np.min(mat) > 6:
            warnings.warn("All satellites above detectabiliy threshold of 6 for kwarg{} {} = {}".format('s' if len(kwargs)>1 else '', ', '.join(kwargs.keys()), ', '.join(map(str, kwargs.values()))))
            return 

        ax = plt.gca()

        xticks = np.arange(len(x_vals)) + 0.5
        ax.set_xticks(xticks)
        ax.set_xticklabels(map(dic[x]['conversion'], x_vals))
        plt.xlabel('{} ({})'.format(dic[x]['label'], dic[x]['unit']))

        yticks = np.arange(len(y_vals)) + 0.5
        ax.set_yticks(yticks)
        ax.set_yticklabels(map(dic[y]['conversion'], y_vals))
        plt.ylabel('{} ({})'.format(dic[y]['label'], dic[y]['unit']))

        title = ('; '.join(["{} = {} {}".format(dic[key]['label'], dic[key]['conversion'](kwargs[key]), dic[key]['unit']) for key in kwargs]))

        # Insert stellar mass ticks/label in appropriate place
        if 'abs_mag' in kwargs:
            stellar_mass = mag_to_mass(kwargs['abs_mag']) 
            title += "; {} = {} {}".format(dic['stellar_mass']['label'], dic['stellar_mass']['conversion'](stellar_mass), dic['stellar_mass']['unit'])
            plt.title(title)
        elif 'abs_mag' in args:
            abs_mags = np.array( sorted(set(table['abs_mag']), reverse=True) )
            stellar_masses = mag_to_mass(abs_mags)
            if x == 'abs_mag':
                twin_ax = ax.twiny()
                twin_ax.set_xticks(list(xticks) + [xticks[-1]+0.5]) # Have to add an extra on the end to make it scale right
                twin_ax.set_xticklabels(map(dic['stellar_mass']['conversion'], stellar_masses) + [''])
                twin_ax.set_xlabel('{} ({})'.format(dic['stellar_mass']['label'], dic['stellar_mass']['unit']))
                plt.subplots_adjust(top=0.85) # Make more vertical room 
                plt.colorbar(label='$\sigma$')
                plt.title(title, y=1.12) # Push title above upper axis labels
            elif y == 'abs_mag':
                twin_ax = ax.twinx()
                twin_ax.set_yticks(list(yticks) + [yticks[-1]+0.5])
                twin_ax.set_yticklabels(map(dic['stellar_mass']['conversion'], stellar_masses) + [''])
                twin_ax.set_ylabel('{} ({})'.format(dic['stellar_mass']['label'], dic['stellar_mass']['unit']))
                plt.colorbar(label='$\sigma$', pad=0.15) # Move colorbar right to make room for axis labels
                plt.title(title)
        else:
            plt.title(title)
            plt.colorbar(label='$\sigma$')

        """
        # Place known sats on plot:
        translation = {'distance':'distance_kpc', 'abs_mag':'m_v', 'r_physical':'r_physical'}
        sats = load_data.Satellites()
        plt.scatter(sats.dwarfs[translation[x]], sats.dwarfs[translation[y]])
        """

        outname = '{}_vs_{}__'.format(x, y) + '_'.join(['{}={}'.format(key, round(kwargs[key],3)) for key in kwargs])
        plt.savefig('mat_plots/2D_' + outname + '.png')
        plt.close()
        

def main():
    pass


if args.cc or args.iso:
    distances = np.arange(200, 2100, 100)
    for distance in distances:
        if args.cc:
            plot_color_color(distance)
        if args.iso:
            plot_isochrone(distance)
if args.sim:
    inputs = load_data.Inputs()
    calc_sigma(inputs, args.distance, args.abs_mag, args.r_physical/1000., plot=True)
if args.scan or args.plots:
    distances = np.arange(400, 2200, 200)
    abs_mags = np.arange(-2.5, -10.5, -0.5) # -2.5 to -10 inclusive
    log_r_physical_pc = np.arange(1, 3.2, 0.2)
    r_physical_kpcs = 10**log_r_physical_pc / 1000.0

    if args.scan:
        create_sigma_matrix(distances, abs_mags, r_physical_kpcs, outname='sigma_matrix')

    if args.plots:
        subprocess.call('mkdir -p {}'.format('mat_plots').split()) # Don't want to have this call happen for each and every plot

        for d in distances:
            plot_matrix('sigma_matrix', 'abs_mag', 'r_physical', distance=d)

        for r in r_physical_kpcs:
            plot_matrix('sigma_matrix', 'distance', 'abs_mag', r_physical=r)

        for m in abs_mags:
            plot_matrix('sigma_matrix', 'distance', 'r_physical', abs_mag=m)

if args.main:
    main()


"""
def background_analysis(maglim=None, nside=2048):
    pix = set(ugali.utils.healpix.angToPix(nside, data['RA'], data['DEC']))

    star_select = np.tile(True, len(stars))
    if maglim:
        star_select = stars[mag(1)] > maglim

    star_locs = ugali.utils.healpix.angToPix(nside, stars[star_select]['RA'], stars[star_select]['DEC'])
    counts = [sum(star_locs == p) for p in pix]

    #bins = min(20, max(counts))
    bins = max(counts)
    hist, edges = np.histogram(counts, bins=bins, density=True)

    mean = np.mean(counts)
    poisson_f = scipy.stats.poisson(mean).pmf
    poisson = map(poisson_f, edges[:-1])

    chisquare, pvalue = scipy.stats.chisquare(hist, poisson)

    plt.hist(counts, bins=edges-0.5, density=True)
    plt.plot(edges, map(poisson_f, edges), label="$\chi^2/({{\\rm dof}}={}) = {}$\np-value = {}".format(len(hist)-1, round(chisquare, 1), round(pvalue, 2)))
    plt.title('maglim = {}'.format(maglim if maglim else 'None'))
    plt.legend()
    plt.savefig('background_plots/maglim={}_nside={}.png'.format((maglim if maglim else 'None'), nside))
    plt.close()

#background_analysis()
#for m in np.linspace(23.0, 25.0, 0.2):
#    print m
#    background_analysis(maglim=m)
"""
