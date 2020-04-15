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
import scipy.ndimage
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

class SimSatellite:
    def __init__(self, inputs, lon_centroid, lat_centroid, distance, abs_mag, r_physical):
        # Stolen from ugali/scratch/simulation/simulate_population.py. Look there for a more general function,
        # which uses maglims, extinction, stuff like that
        """
        r_physical is azimuthally averaged half-light radius, pc
        """

        # Probably don't want to parse every time
        s = ugali.analysis.source.Source()

        # Following McConnachie 2012, ellipticity = 1 - (b/a) , where a is semi-major axis and b is semi-minor axis
        r_h = np.degrees(np.arcsin(r_physical/1000. / distance)) # Azimuthally averaged half-light radius
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
            # raise Exception('flag_too_extended')
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

        flux_g_meas = utils.magToFlux(mag_g) + np.random.normal(scale=utils.getFluxError(mag_g, mag_g_error))
        mag_g_meas = np.where(flux_g_meas > 0., utils.fluxToMag(flux_g_meas), 99.)
        flux_r_meas = utils.magToFlux(mag_r) + np.random.normal(scale=utils.getFluxError(mag_r, mag_r_error))
        mag_r_meas = np.where(flux_r_meas > 0., utils.fluxToMag(flux_r_meas), 99.)
        flux_i_meas = utils.magToFlux(mag_i) + np.random.normal(scale=utils.getFluxError(mag_i, mag_i_error))
        mag_i_meas = np.where(flux_i_meas > 0., utils.fluxToMag(flux_i_meas), 99.)

        # Includes penalty for interstellar extinction and also include variations in depth
        # Use r band:
        # 24.42 is the median magnitude limit from Y6 according to Keith's slack message
        cut_detect = (np.random.uniform(size=len(mag_r)) < inputs.completeness(mag_r + mag_extinction_r + (24.42 - np.clip(maglim_r, 20., 26.))))

        # Absoulte Magnitude
        v = mag_g - 0.487*(mag_g - mag_r) - 0.0249 # Don't know where these numbers come from, copied from ugali
        flux = np.sum(10**(-v/2.5))
        abs_mag_realized = -2.5*np.log10(flux) - distance_modulus

        r_physical = distance * np.tan(np.radians(r_h)) * 1000. # Azimuthally averaged half-light radius, pc
        surface_brightness_realized = ugali.analysis.results.surfaceBrightness(abs_mag_realized, r_physical, distance) # Average within azimuthally averaged half-light radius

        # Turn star info into an array
        sat_stars = np.array([lon, lat, mag_g_meas, mag_r_meas, mag_i_meas, mag_g_error, mag_r_error, mag_i_error]).T
        sat_stars = sat_stars[cut_detect]
        sat_stars = list(map(tuple, sat_stars)) # This makes setting the dtype work
        dtype = [('lon',float),('lat',float),('mag_g',float),('mag_r',float),('mag_i',float),('mag_g_err',float),('mag_r_err',float),('mag_i_err',float)]
        sat_stars = np.array(sat_stars, dtype=dtype)

        self.stars = sat_stars
        self.a_h = a_h
        self.ellipcticity = ellipticity
        self.position_angle = position_angle
        self.abs_mag_realized = abs_mag_realized
        self.surface_brightness_realized = surface_brightness_realized
        self.flag_too_extended = flag_too_extended
        self.iso = iso


class Dataset:
    # Stolen from simple, simple_utils.py, find_peaks and compute_char_density, also search_algorithm.py search_by_distance, etc
    def __init__(self, patch, *sats):
        self.patch = patch
        stars = patch.stars
        mag = patch.mag
        magerr = patch.magerr

        # Make one combined catalog
        self.proj = ugali.utils.projector.Projector(patch.center_ra, patch.center_dec)
        ra = stars['RA']
        dec = stars['DEC']
        x, y = self.proj.sphereToImage(ra, dec)
        g = stars[mag('g')]
        r = stars[mag('r')]
        i = stars[mag('i')]
        gerr = stars[magerr('g')]
        rerr = stars[magerr('r')]
        ierr = stars[magerr('i')]

        for sat in sats:
            x_sat, y_sat = self.proj.sphereToImage(sat.stars['lon'], sat.stars['lat'])
            ra = np.concatenate((ra, sat.stars['lon']))
            dec = np.concatenate((dec, sat.stars['lat']))
            x = np.concatenate((x, x_sat))
            y = np.concatenate((y, y_sat))
            g = np.concatenate((g, sat.stars['mag_g']))
            r = np.concatenate((r, sat.stars['mag_r']))
            i = np.concatenate((i, sat.stars['mag_i']))
            gerr = np.concatenate((gerr, sat.stars['mag_g_err']))
            rerr = np.concatenate((rerr, sat.stars['mag_r_err']))
            ierr = np.concatenate((ierr, sat.stars['mag_i_err']))

        dtype = [('ra',float),('dec',float),('x',float),('y',float),('mag_g',float),('mag_r',float),('mag_i',float),('mag_g_err',float),('mag_r_err',float),('mag_i_err',float)]
        cat = np.array([ra, dec, x, y, g, r, i, gerr, rerr, ierr])
        self.catalog = np.core.records.fromarrays(cat, dtype)

    def __len__(self):
        return(len(self.catalog))
    def __getitem__(self, key):
        return self.catalog[key]
    def __setitem__(self, key, val):
        self.catalog[key] = val


    def reduced_catalog(self, iso):
        cat = self.catalog
        cut_cat = utils.cut(iso, cat['mag_g'], cat['mag_r'], cat['mag_i'], cat['mag_g_err'], cat['mag_r_err'], cat['mag_i_err'])
        return self.catalog[cut_cat]


    def compute_characteristic_density(self, iso=None, region_size=1.0): # TODO: For now, region size is set. Could be increased for larger dataset
        if iso is not None:
            cat = self.reduced_catalog(iso)
        else:
            cat = self.catalog

        delta_x_background = 6./60.
        area_background = delta_x_background**2 # Chunk size in square degrees
        bins_background = np.arange(-region_size/2., region_size/2.+1e-10, delta_x_background)
        h_background = np.histogram2d(cat['x'], cat['y'], bins=[bins_background,bins_background])[0]
        n_background = h_background[h_background > 0].flatten()

        self.characteristic_density = np.median(n_background)/area_background # per square degree
        #return characteristic_density


    def find_peaks(self, iso=None, region_size=1.0): # TODO: Adjust region size for larger dataset. Not necessarily the same as region_size in characteristic_density
        if iso is not None:
            cat = self.reduced_catalog(iso)
        else:
            cat = self.catalog
        self.compute_characteristic_density(iso)

        # Find peaks
        delta_x = 0.01 # degrees
        area = delta_x**2
        bins = np.arange(-region_size/2., region_size/2.+1e-10, delta_x)
        centers = 0.5*(bins[:-1] + bins[1:])

        h = np.histogram2d(cat['x'], cat['y'], bins=[bins, bins])[0]
        smoothing = 2./60. # degrees
        h_g = scipy.ndimage.filters.gaussian_filter(h, smoothing / delta_x)

        factor_array = np.arange(1., 5., 0.05)
        # Create spatial grid and convert to ra/dec values
        yy, xx = np.meshgrid(centers, centers)
        # restrict to singel healpixel? I think here we're assuming a scan over healpixels
        nside = NSIDE
        rara, decdec = self.proj.imageToSphere(xx.flatten(), yy.flatten())
        cutcut = (ugali.utils.healpix.angToPix(nside, rara, decdec) == ugali.utils.healpix.angToPix(nside, self.patch.center_ra, self.patch.center_dec)).reshape(xx.shape)
        #cutcut = 1
        threshold_density = 5 * self.characteristic_density * area
        for factor in factor_array:
            # loops through factors until number of peaks is < 10. 
            h_region, n_region = scipy.ndimage.measurements.label((h_g * cutcut) > (factor * self.characteristic_density * area))
            #print 'factor', factor, n_region, n_region < 10
            if n_region < 10:
                threshold_density = factor * self.characteristic_density * area
                break

        h_region, n_region = scipy.ndimage.measurements.label((h_g * cutcut) > threshold_density)
        h_region = np.ma.array(h_region, mask=(h_region < 1))

        x_peak_array = []
        y_peak_array = []
        angsep_peak_array = []

        # Loop over number of found peaks to build arrays
        for idx in range(1, n_region+1): # loop over peaksa go 
            index_peak = np.argmax(h_g * (h_region == idx))
            x_peak, y_peak = xx.flatten()[index_peak], yy.flatten()[index_peak]
            angsep_peak = np.sqrt((cat['x'] - x_peak)**2 + (cat['y'] - y_peak)**2) # Each element in this array is a list of the angseps of each star from the peak location

            x_peak_array.append(x_peak)
            y_peak_array.append(y_peak)
            angsep_peak_array.append(angsep_peak)

        self.x_peak_array = x_peak_array
        self.y_peak_array = y_peak_array
        self.angsep_peak_array = angsep_peak_array
        self.n_peaks = n_region

        #return x_peak_array, y_peak_array, angsep_peak_array, h_g


    def fit_peaks(self, iso=None):
        #x_peak_array, y_peak_array, angsep_peak_array, h_g = find_peaks(iso)
        self.find_peaks(iso)

        ra_peak_array = np.tile(0., self.n_peaks)
        dec_peak_array = np.tile(0., self.n_peaks)
        aperture_peak_array = np.tile(0., self.n_peaks)
        sig_peak_array = np.tile(0., self.n_peaks)
        n_obs_peak_array = np.tile(0., self.n_peaks)
        n_obs_half_peak_array = np.tile(0., self.n_peaks)
        n_model_peak_array = np.tile(0., self.n_peaks)

        # Loop through peaks, fit aperture for each one
        for j in range(self.n_peaks):
            angsep_peak = self.angsep_peak_array[j]
            # Compute local characteristic density TODO: annulus for background defined here
            inner = 0.3 # degrees
            outer = 0.5 # degrees
            area_field = np.pi*(outer**2 - inner**2) 
            n_field = np.sum((angsep_peak > inner) & (angsep_peak < outer))
            characteristic_density_local = n_field/area_field

            # see simple_utils.py, fit_aperture
            aperture_array = np.arange(0.01, 0.3, 0.01) # degrees TODO: aperture size is scanned over here
            sig_array = np.tile(0., len(aperture_array))

            n_obs_array = np.tile(0, len(aperture_array))
            n_model_array = np.tile(0, len(aperture_array))
            for i in range(len(aperture_array)): # Loop through aperture sizes
                aperture = aperture_array[i]
                n_obs = np.sum(angsep_peak < aperture)
                n_model = characteristic_density_local * (np.pi * aperture**2)
                sig_array[i] = np.clip(norm.isf(poisson.sf(n_obs, n_model)), 0., 37.5)
                n_obs_array[i] = n_obs
                n_model_array[i] = n_model

            index_peak = np.argmax(sig_array)
            sig_peak = sig_array[index_peak]
            aperture_peak = aperture_array[index_peak]
            n_obs_peak = n_obs_array[index_peak]
            n_model_peak = n_model_array[index_peak]
            n_obs_half_peak = np.sum(angsep_peak < 0.5*aperture_peak)
            ra_peak, dec_peak = self.proj.imageToSphere(self.x_peak_array[j], self.y_peak_array[j])

            ra_peak_array[j] = ra_peak
            dec_peak_array[j] = dec_peak
            aperture_peak_array[j] = aperture_peak
            sig_peak_array[j] = sig_peak
            n_obs_peak_array[j] = n_obs_peak
            n_obs_half_peak_array[j] = n_obs_half_peak
            n_model_peak_array[j] = n_model_peak

        # Sort by significance
        index_sort = np.argsort(sig_peak_array)[::-1]
        ra_peak_array = ra_peak_array[index_sort]
        dec_peak_array = dec_peak_array[index_sort]
        aperture_peak_array = aperture_peak_array[index_sort]
        sig_peak_array = sig_peak_array[index_sort]
        n_obs_peak_array = n_obs_peak_array[index_sort]
        n_obs_half_peak_array = n_obs_half_peak_array[index_sort]
        n_model_peak_array = n_model_peak_array[index_sort]

        # Consolidate peaks
        for i in range(len(sig_peak_array)):
            if sig_peak_array[i] < 0:
                continue
            angsep = ugali.utils.projector.angsep(ra_peak_array[i], dec_peak_array[i], ra_peak_array, dec_peak_array)
            sig_peak_array[(angsep < aperture_peak_array[i]) & (np.arange(len(sig_peak_array)) > i)] = -1.

        self.ra_peak_array = ra_peak_array[sig_peak_array > 0.]
        self.dec_peak_array = dec_peak_array[sig_peak_array > 0.]
        self.aperture_peak_array = aperture_peak_array[sig_peak_array > 0.]
        self.n_obs_peak_array = n_obs_peak_array[sig_peak_array > 0.]
        self.n_obs_half_peak_array = n_obs_half_peak_array[sig_peak_array > 0.]
        self.n_model_peak_array = n_model_peak_array[sig_peak_array > 0.]
        self.sig_peak_array = sig_peak_array[sig_peak_array > 0.] # Update the sig_peak_array last!

        #return ra_peak_array, dec_peak_array, aperture_peak_array, n_obs_peak_array, n_obs_half_peak_array, n_model_peak_array, sig_peak_array


def calc_sigma(distance, abs_mag, r_physical, plot=False, outname=None, inputs=None):
    if inputs is None:
        inputs = load_data.Inputs()
    patch = load_data.Patch()
    sat = SimSatellite(inputs, patch.center_ra, patch.center_dec, distance, abs_mag, r_physical)
    data = Dataset(patch, sat)

    data.fit_peaks(sat.iso)

    print data.sig_peak_array
    print data.aperture_peak_array
    print sat.a_h

    return data


def old():

    ###START OLD CODE###

    # Backround density
    field_pix = ugali.utils.healpix.angToPix(NSIDE, stars[cut_field]['RA'], stars[cut_field]['DEC'])
    sat_pix = ugali.utils.healpix.angToPix(NSIDE, lon[cut_sat], lat[cut_sat])
    # Annulus 
    inner_r, outer_r = 0.05, 0.10 # Degrees
    inner_pix = ugali.utils.healpix.angToDisc(NSIDE, center_ra, center_dec, inner_r)
    outer_pix = ugali.utils.healpix.angToDisc(NSIDE, center_ra, center_dec, outer_r)
    field_annulus = ~np.isin(field_pix, inner_pix) & np.isin(field_pix, outer_pix)
    sat_annulus = ~np.isin(sat_pix, inner_pix) & np.isin(sat_pix, outer_pix)

    rho_field = (sum(field_annulus) + sum(sat_annulus))/(np.pi*outer_r**2 - np.pi*inner_r**2)
    #true_rho_field = sum(cut_field)/1.0
    #true_background = true_rho_field * np.pi*a*b
    
    # Signal
    if aperture_type == 'factor':
        a = aperture_in*a_h
    elif aperture_type == 'radius':
        a = aperture_in
    else:
        raise ValueError('bad aperture_type')

    # Ellpise with known position angle and ellipticity, a = a_h
    if aperture_shape == 'ellipse':
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
        background = rho_field * np.pi*a*b

    # Circle of radius a
    # Avoding angToDisc since resolution may be a problem
    elif aperture_shape == 'circle':
        sat_in_circle = ((lon[cut_sat]-center_ra)**2 + (lat[cut_sat]-center_dec)) < a**2
        field_in_circle = ((stars[cut_field]['RA']-center_ra)**2 + (stars[cut_field]['DEC']-center_dec)**2) < a**2
        signal = sum(field_in_circle) + sum(sat_in_circle)
        background = rho_field * np.pi*a**2
    else:
        raise ValueError('bad aperture_shape')

    sigma = min(norm.isf(poisson.sf(signal, background)), 38.0)
    sigma = max(sigma, 0.0)

    if plot:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
        title = '$D = {}$ kpc, $M_V = {}$ mag, $r_{{1/2}} = {}$ pc\n$\sigma = {}$'.format(int(round(distance,0)), round(abs_mag_realized, 1), int(round(r_physical, 0)), round(sigma, 1))
        if flag_too_extended:
            title += ' ${\\rm (FLAG TOO EXTENDED)}$'
            # Had to add the $$ because tex was being weird

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
        plt.ylim(min(stars[cut_field][mag('r')])-0.5, max(stars[cut_field][mag('r')])+0.5)
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

        plt.scatter(lon[cut_sat]-center_ra, lat[cut_sat]-center_dec, s=6, color='black', label='Included satellite stars', zorder=5)
        plt.scatter(lon[~cut_sat]-center_ra, lat[~cut_sat]-center_dec, s=2, color='0.3', label='Excluded satellite stars', zorder=4)
        plt.scatter(stars[cut_field]['RA']-center_ra, stars[cut_field]['DEC']-center_dec, s=4, color='red', label='Included field stars', zorder=3)
        plt.scatter(stars[~cut_field]['RA']-center_ra, stars[~cut_field]['DEC']-center_dec, s=0.5, color='coral', label='Excluded field stars', zorder=2)

        half_light_ellipse = Ellipse(xy=(0,0), width=2*a_h, height=2*(1-ellipticity)*a_h, angle=90-position_angle, edgecolor='green', linewidth=1.5, linestyle='--', fill=False, zorder=10)
        half_light_ellipse_label = "$a_h = {}'$".format(round(a_h*60, 1))
        if aperture_shape == 'ellipse':
            aperture_patch = Ellipse(xy=(0,0), width=2*a, height=2*((1-ellipticity)*a), angle=90-position_angle, edgecolor='green', linewidth=1.5, fill=False, zorder=10)
            aperture_label = 'Aperture ($a = {}$)'.format(round(a*60, 1))
        elif aperture_shape == 'circle':
            aperture_patch = Circle(xy=(0,0), radius=a, edgecolor='green', linewidth=1.5, fill=False, zorder=10)
            aperture_label = 'Aperture ($r = {}$)'.format(round(a*60, 1))
        ax.add_patch(half_light_ellipse)
        ax.add_patch(aperture_patch)
        #plt.legend((half_light_ellipse, aperture_patch), ("$a_h = {}'$".format(round(a_h*60, 1)), '$3 a_h$'), loc='upper right')
        plt.legend((half_light_ellipse, aperture_patch), (half_light_ellipse_label, aperture_label), loc='upper right')

        plt.xlim(-5*a_h, 5*a_h)
        plt.ylim(-5*a_h, 5*a_h)
        plt.xlabel('$\Delta$RA (deg)')
        plt.ylabel('$\Delta$DEC (deg)')

        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, markerscale=2.0)

        if outname is None:
            outname = 'D={}_Mv={}_r={}'.format(int(distance), round(abs_mag,1), int(round(r_physical,0)))
        outdir = 'sat_plots/'
        subprocess.call('mkdir -p {}'.format(outdir).split())
        plt.savefig(outdir + outname + '.png')
        plt.close()

    #params = {'abs_mag_realized':abs_mag_realized, 'surface_brightness_realized':surface_brightness_realized} # Could add more later if needed
    return sigma


def calc_sigma_trials(inputs, distance, abs_mag, r_physical, aperture=1, aperture_type='factor', aperture_shape='circle', n_trials=10, percent_bar=False):
    sigmas = []
    for i in range(n_trials):
        sigmas.append(calc_sigma(inputs,distance,abs_mag,r_physical,aperture,aperture_type))
        if percent_bar: percent.bar(i+1, n_trials)
    return np.mean(sigmas), np.std(sigmas), sigmas


def create_sigma_table(distances, abs_mags, r_physicals, apertures=None, aperture_type='factor', aperture_shape='cirlce', outname='sigma_table', n_trials=1):
    """Input arrays are all the same size. Runs throught i in range(len(array)) and calcs sigma for the ith satellite"""
    if apertures is None:
        apertures = np.tile(1, len(distances))
        aperture_type = 'factor'

    ld = len(distances)
    lm = len(abs_mags)
    lr = len(r_physicals)
    la = len(apertures)
    if not (ld==lm and lm==lr and lr==la):
        raise ValueError("Input arrays are not all the same size")
    
    inputs = load_data.Inputs()
    sigma_table = []
    skipped_sats = []
    for i in range(ld):
        d, m, r, a = distances[i], abs_mags[i], r_physicals[i], apertures[i]
        if m < -10.0:
            skipped_sats.append((i,round(m,2)))
            sigma = 38.0
        elif n_trials > 1:
            sigma = calc_sigma_trials(inputs, d, m, r, a, aperture_type, aperture_shape, n_trials)[0]
        else:
            sigma = calc_sigma(inputs, d, m, r, a, aperture_type, aperture_shape, plot=False)
        sigma_table.append((d,m,r,a,sigma))
        percent.bar(i+1, ld)
    print '{} sats skipped due to large abs_mag'.format(len(skipped_sats)) + (':' if len(skipped_sats)>0 else '')
    if len(skipped_sats)>0:
        print np.array(skipped_sats)

    dtype = [('distance',float), ('abs_mag',float), ('r_physical',float), ('aperture',float), ('sigma',float)]
    sigma_table = np.array(sigma_table, dtype=dtype)
    fits.writeto(outname+'.fits', sigma_table, overwrite=True)
    

def create_sigma_matrix(distances, abs_mags, r_physicals, apertures=[1], aperture_type='factor', aperture_shape='circle', outname='sigma_matrix', n_trials=1):
    """ Scans over all combinations of distance, abs_mag, r_physical, and apertures in the input arrays"""
    n_d = len(distances)
    n_m = len(abs_mags)
    n_r = len(r_physicals)
    n_a = len(apertures)
    inputs = load_data.Inputs()

    #sigma_matrix = np.zeros((n_d, n_m, n_r))
    sigma_fits = []
    counter=0
    for i in range(n_d):
        for j in range(n_m):
            for k in range(n_r):
                for l in range(n_a):
                    d, m, r, a = distances[i], abs_mags[j], r_physicals[k], apertures[l]
                    if n_trials > 1:
                        sigma = calc_sigma_trials(inputs, d, m, r, a, aperture_type, aperture_shape, n_trials)[0]
                    else:
                        sigma = calc_sigma(inputs, d, m, r, a, aperture_type, aperture_shape, plot=False)

                    #sigma_matrix[i,j,k] = sigma
                    sigma_fits.append((d, m, r, a, sigma))

                    counter += 1
                    percent.bar(counter, n_d*n_m*n_r*n_a)

    #np.save(outname+'.npy', sigma_matrix) # Not used but I feel like I might as well make it
    dtype = [('distance',float), ('abs_mag',float), ('r_physical',float), ('aperture',float), ('sigma',float)]
    sigma_fits = np.array(sigma_fits, dtype=dtype)
    fits.writeto(outname+'.fits', sigma_fits, overwrite=True)



def plot_matrix(fname, *args, **kwargs):
    if 'factor' in fname:
        aperture_type = 'factor'
    elif 'radius' in fname:
        aperture_type = 'radius'

    if 'circle' in fname:
        aperture_type = 'cirlce'
    elif 'ellipse' in fname:
        aperture_type = 'ellipse'

    dic = {'distance': {'label':'$D$', 'conversion': lambda d:int(d), 'unit':'kpc', 'scale':'linear', 'reverse':False}, 
           'abs_mag': {'label':'$M_V$', 'conversion': lambda v:round(v,1), 'unit':'mag', 'scale':'linear', 'reverse':True},
           'r_physical': {'label':'$r_{1/2}$', 'conversion': lambda r:int(round(r,0)), 'unit':'pc', 'scale':'log', 'reverse':False},
           'stellar_mass': {'label':'$M_*$', 'conversion': lambda m: '$10^{{{}}}$'.format(round(np.log10(m),1)), 'unit':'$M_{\odot}$', 'scale':'log', 'reverse':False},
           'aperture': {'label':'aperture', 'conversion': lambda a:round(a,2), 'unit':'$a_h$' if aperture_type=='factor' else 'deg', 'scale':'linear', 'reverse':False}
           }
    def is_near(arr, val, e=0.001):
        return np.array([val-e < a < val+e for a in arr])

    if '.fits' not in fname:
        fname += '.fits'
    #if len(args)+len(kwargs) != 3:
    #    raise ValueError("Error in the number of specified parameters")

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

        #plt.figure(figsize=(len(x_vals)/2. + 3,len(y_vals)/2.))
        plt.figure(figsize=(len(x_vals) + 8,len(y_vals)/1.5))
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
            stellar_mass = utils.mag_to_mass(kwargs['abs_mag']) 
            title += "; {} = {} {}".format(dic['stellar_mass']['label'], dic['stellar_mass']['conversion'](stellar_mass), dic['stellar_mass']['unit'])
            plt.colorbar(label='$\sigma$')
            plt.title(title)
        elif 'abs_mag' in args:
            abs_mags = np.array( sorted(set(table['abs_mag']), reverse=True) )
            stellar_masses = utils.mag_to_mass(abs_mags)
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

        
        # Place known sats on plot:
        if 'abs_mag' in args and 'r_physical' in args:
            plt.sca(ax)
            translation = {'distance':'distance_kpc', 'abs_mag':'m_v', 'r_physical':'r_physical'}
            dwarfs = load_data.Satellites().dwarfs
            xmin, xmax = (x_vals[0], x_vals[-1]) if not dic[x]['reverse'] else (x_vals[-1], x_vals[0])
            ymin, ymax = (y_vals[0], y_vals[-1]) if not dic[y]['reverse'] else (y_vals[-1], y_vals[0])
            cut = xmin < dwarfs[translation[x]]
            cut &= dwarfs[translation[x]] < xmax
            cut &= ymin < dwarfs[translation[y]]
            cut &= dwarfs[translation[y]] < ymax
            #cut &= np.array(['des' in survey for survey in dwarfs['survey']])
            dwarfs = dwarfs[cut]

            def transform(value, axis_vals, log=False):
                if log:
                    axis_vals = np.log10(axis_vals)
                    value = np.log10(value)
                delta = axis_vals[1] - axis_vals[0]
                mn = axis_vals[0] - delta/2.
                mx = axis_vals[-1] + delta/2.
                return ((value-mn)/(mx-mn)) * len(axis_vals)

            sat_xs = transform(dwarfs[translation[x]], x_vals, dic[x]['scale']=='log')
            sat_ys = transform(dwarfs[translation[y]], y_vals, dic[y]['scale']=='log')
            plt.scatter(sat_xs, sat_ys, color='k')

            down = ['Boo II', 'Pic I', 'Ret II', 'Phe II', 'Leo V', 'Hyi I', 'UMa II']
            left = ['Phe II', 'Gru II', 'Psc II', 'Com', 'Sex', 'UMa II', 'Pic II']
            for i, d in enumerate(dwarfs):
                xy = (sat_xs[i], sat_ys[i])
                xytext = [3,3]
                ha = 'left'
                va = 'bottom'
                if d['abbreviation'] in down:
                    va = 'top'
                    xytext[1] = -3
                if d['abbreviation'] in left:
                    ha = 'right'
                    xytext[0] = -3
                plt.annotate(d['abbreviation'], xy, textcoords='offset points', xytext=xytext, ha=ha, va=va, 
                             bbox = dict(facecolor='white', boxstyle='round,pad=0.2'))

        outname = '{}_vs_{}__'.format(x, y) + '_'.join(['{}={}'.format(key, round(kwargs[key],3)) for key in kwargs])
        plt.savefig('mat_plots/2D_' + outname + '.png', bbox_inches='tight')
        plt.close()

        
def sim_known_sats():
    inputs = load_data.Inputs()
    dwarfs = load_data.Satellites().dwarfs
    cut = dwarfs['m_v'] > -10
    # Cuts out LMC, SMC, Fornax, Sculptor, Sagitarrius, Leo I
    dwarfs = dwarfs[cut]

    subprocess.call('mkdir -p sat_plots/known_sats'.split())
    distances = np.arange(400, 2200, 200)
    for i, dwarf in enumerate(dwarfs):
        name = dwarf['name'].replace(' ', '_')
        abbr = dwarf['abbreviation'].replace(' ','')
        abs_mag = dwarf['m_v']
        r_physical = dwarf['r_physical']
        subprocess.call('mkdir -p sat_plots/known_sats/{}'.format(name).split())

        outname = 'known_sats/{}/{}_D={}'.format(name, abbr, int(round(dwarf['distance_kpc'], 0)))
        calc_sigma(inputs, dwarf['distance_kpc'], abs_mag, r_physical, plot=True, outname=outname)
        for j, distance in enumerate(distances):
            outname = 'known_sats/{}/{}_D={}'.format(name, abbr, distance)
            sigma = calc_sigma(inputs, distance, abs_mag, r_physical, plot=True, outname=outname)
            percent.bar(i*(len(distances)+1)+j+1, len(dwarfs)*(len(distances)+1))
            if sigma < 4:
                # Don' waste time with farther distances
                break



def main():
    inputs = load_data.Inputs()
    distances = [400, 800, 1200]
    abs_mags = [-4, -6, -8]
    r_physicals = [50, 100, 300]
    counter = 0
    for d in distances:
        for m in abs_mags:
            for r in r_physicals:
                calc_sigma(inputs, d, m, r, plot=True)
                counter += 1
                percent.bar(counter, len(distances)*len(abs_mags)*len(r_physicals))

if __name__ == '__main__':
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
    p.add_argument('--known', action='store_true')
    p.add_argument('--main', action='store_true')
    args = p.parse_args()

    if args.cc or args.iso:
        distances = np.arange(200, 2100, 100)
        for distance in distances:
            if args.cc:
                plot_color_color(distance)
            if args.iso:
                plot_isochrone(distance)
    if args.sim:
        inputs = load_data.Inputs()
        calc_sigma(inputs, args.distance, args.abs_mag, args.r_physical, plot=True)
    if args.scan or args.plots:
        #distances = np.arange(400, 2200, 200)
        #abs_mags = np.arange(-2.5, -10.5, -0.5) # -2.5 to -10 inclusive
        log_r_physical_pcs = np.arange(1, 3.2, 0.2)
        r_physicals = 10**log_r_physical_pcs
        #n_trials=1

        distances = np.arange(400, 2200, 200)
        abs_mags = np.array([-6.5])
        #r_physicals = np.array([100])
        apertures = np.arange(0.01, 0.11, 0.01)
        aperture_type='radius'
        n_trials=1

        if args.scan:
            create_sigma_matrix(distances, abs_mags, r_physicals, apertures, aperture_type, outname='sigma_matrix_{}'.format(aperture_type), n_trials=n_trials)

        if args.plots:
            subprocess.call('mkdir -p {}'.format('mat_plots').split()) # Don't want to have this call happen for each and every plot

            fname = 'sigma_matrix_radius'
            for d in distances:
                plot_matrix(fname, 'aperture', 'r_physical', distance=d)
            for r in r_physicals:
                plot_matrix(fname, 'distance', 'aperture', r_physical=r)
            #for m in abs_mags:
            #    plot_matrix(fname, 'distance', 'r_physical', abs_mag=m)
            for a in apertures:
                plot_matrix(fname, 'distance', 'r_physical', aperture=a)

            #fname = 'sigma_matrix_factor'
            #for d in distances:
            #    for r in r_physicals:
            #        plot_matrix(fname, 'abs_mag', 'aperture', distance=d, r_physical=r)
            #    for m in abs_mags:
            #        plot_matrix(fname, 'r_physical', 'aperture', distance=d, abs_mag=m)
            #    for a in apertures:
            #        plot_matrix(fname, 'abs_mag', 'r_physical', distance=d, aperture=a)
            #for r in r_physicals:
            #    for m in abs_mags:
            #        plot_matrix(fname, 'distance', 'aperture', r_physical=r, abs_mag=m)
            #    for a in apertures:
            #        plot_matrix(fname, 'distance' 'abs_mag', r_physical=r, aperture=a)
            #for m in abs_mags:
            #    for a in apertures:
            #        plot_matrix(fname, 'distance', 'r_physical', abs_mag=m, aperture=a)

            plot_matrix('sigma_matrix_factor', 'distance', 'aperture', r_physical=100, abs_mag=-6.5)


    if args.known:
        sim_known_sats()
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
