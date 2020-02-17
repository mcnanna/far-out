#!/usr/bin/env python

from astropy.io import fits
import load_data
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ugali.utils.projector
import pandas as pd

matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
plt.ion()

des_patch = load_data.Patch()
d_des = des_patch.data

### Completeness ###

infile_hsc = 'datafiles/hsc_dr2_udeep_sxds_griz_test_flag_update.fits'
d_hsc = fits.open(infile_hsc)[1].data

# Spatial cut

ra_min, ra_max = 35, 36 
dec_min, dec_max = -5, -4

cut_des = (d_des['RA'] > ra_min) & (d_des['RA'] < ra_max) & \
          (d_des['DEC'] > dec_min) & (d_des['DEC'] < dec_max)
cut_hsc = (d_hsc['ra'] > ra_min) & (d_hsc['ra'] < ra_max) & \
          (d_hsc['dec'] > dec_min) & (d_hsc['dec'] < dec_max)
d_des = d_des[cut_des]
d_hsc = d_hsc[cut_hsc]

# Quality cuts

cut_hsc = np.isfinite(d_hsc['g_cmodel_mag']) & ~np.isnan(d_hsc['g_cmodel_mag']) & \
          np.isfinite(d_hsc['r_cmodel_mag']) & ~np.isnan(d_hsc['r_cmodel_mag']) & \
          np.isfinite(d_hsc['i_cmodel_mag']) & ~np.isnan(d_hsc['i_cmodel_mag']) & \
          np.isfinite(d_hsc['z_cmodel_mag']) & ~np.isnan(d_hsc['z_cmodel_mag']) & \
          np.isfinite(d_hsc['g_psfflux_mag']) & ~np.isnan(d_hsc['g_psfflux_mag']) & \
          np.isfinite(d_hsc['r_psfflux_mag']) & ~np.isnan(d_hsc['r_psfflux_mag']) & \
          np.isfinite(d_hsc['i_psfflux_mag']) & ~np.isnan(d_hsc['i_psfflux_mag']) & \
          np.isfinite(d_hsc['z_psfflux_mag']) & ~np.isnan(d_hsc['z_psfflux_mag'])
d_hsc = d_hsc[cut_hsc]

hsc_combo = ((d_hsc['r_psfflux_mag'] - d_hsc['r_cmodel_mag']) \
             + (d_hsc['i_psfflux_mag'] - d_hsc['i_cmodel_mag']) \
             + (d_hsc['z_psfflux_mag'] - d_hsc['z_cmodel_mag'])) / 3.

cut_hsc_star = (hsc_combo < 0.016)
cut_hsc_gals = ~cut_hsc_star

hsc_stars = d_hsc[cut_hsc_star]
hsc_galaxies = d_hsc[cut_hsc_gals]


# Matching
# Matching HSC stars to all DES detected objects
all_match_des, all_match_hsc, all_angsep = ugali.utils.projector.match(des_patch.data['RA'], des_patch.data['DEC'],
                                                                       hsc_stars['ra'], hsc_stars['dec'], tol=1/3600.)
hsc_all_match_cut = np.tile(False, len(hsc_stars))
hsc_all_match_cut[all_match_hsc] = True
# Matching HSC stars to DES classified stars
star_match_des, star_match_hsc, star_angsep = ugali.utils.projector.match(des_patch.stars['RA'], des_patch.stars['DEC'],
                                                                          hsc_stars['ra'], hsc_stars['dec'], tol=1/3600.)

hsc_star_match_cut = np.tile(False, len(hsc_stars))
hsc_star_match_cut[star_match_hsc] = True

# Matching HSC stars to DES classified galaxies
gal_match_des, gal_match_hsc, gal_angsep = ugali.utils.projector.match(des_patch.galaxies['RA'], des_patch.galaxies['DEC'],
                                                                       hsc_stars['ra'], hsc_stars['dec'], tol=1/3600.)

out_array = []

mag_edges = np.arange(20.0, 26.1, 0.1)
mag_centers = (mag_edges[:-1] + mag_edges[1:])/2.
for i in range(len(mag_centers)):
    hsc_mag_cut = (mag_edges[i] < hsc_stars['r_psfflux_mag']) & (hsc_stars['r_psfflux_mag'] < mag_edges[i+1])
    # Total detection efficiency
    num = sum(hsc_mag_cut & hsc_all_match_cut)
    denom = sum(hsc_mag_cut)
    detection_efficiency = float(num)/float(denom)

    # Detection and classification efficiency
    num = sum(hsc_mag_cut & hsc_star_match_cut)
    denom = sum(hsc_mag_cut)
    detection_and_classification_efficiency = float(num)/float(denom)
    print num, denom

    #des_mag_cut = (mag_edges[i] < des_patch.stars[des_patch.mag('r')]) & (des_patch.stars[des_patch.mag('r')] < mag_edges[i+1])
    #denom = sum(des_mag_cut)
    #purity.append(float(num)/float(denom))

    out_array.append((mag_centers[i], detection_efficiency, detection_and_classification_efficiency)) 
out_array = np.array(out_array, dtype=[('mag_r',float), ('eff',float), ('eff_star',float)])
pd.DataFrame(out_array).to_csv('datafiles/completeness.csv', index=False)

y3= np.recfromcsv('datafiles/y3a2_stellar_classification_summary_ext2.csv')


plt.plot(out_array['mag_r'], out_array['eff'], color='tab:blue', label='Y6 Detection efficiency')
plt.plot(y3['mag_r'], y3['eff'], color='tab:blue', linestyle='--', label='Y3 Detection efficiency')
plt.plot(out_array['mag_r'], out_array['eff_star'], color='tab:orange', label='Y6 Detection and classification efficiency')
plt.plot(y3['mag_r'], y3['eff_star'], color='tab:orange', linestyle='--', label='Y3 Detection and classification efficiency')
plt.grid(linestyle='--')
plt.legend()
plt.xlabel('mag')
plt.ylim(0, 1)
plt.savefig('completeness.png')








