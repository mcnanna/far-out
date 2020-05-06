#!/usr/bin/env python

import numpy as np
import astropy.io.fits as fits
import load_data
import significance
import argparse

p = argparse.ArgumentParser()
p.add_argument('distance', type=float, help='kpc')
p.add_argument('abs_mag', type=float)
p.add_argument('r_physical', type=float, help='pc')
args = p.parse_args()

inputs = load_data.Inputs()
data = fits.open('datafiles/y6_gold_1_1_patch.fits')[1].data
sat = significance.SimSatellite(inputs, 35.5, -4.5, args.distance, args.abs_mag, args.r_physical)

fake_data = []
for key in data.dtype.names:
    if key == 'RA':
        fake_data.append( np.concatenate((data[key], sat.stars['lon'])) )
    elif key == 'DEC':
        fake_data.append( np.concatenate((data[key], sat.stars['lat'])) )
    elif ('PSF_MAG' in key) and ('ERR' not in key):
        band = key[-1].lower()
        if band in ('g', 'r', 'i'):
            fake_data.append( np.concatenate((data[key], sat.stars['mag_{}'.format(band)])) )
        else:
            fake_data.append( np.concatenate((data[key], np.tile(0, len(sat.stars)))) )
    elif ('PSF_MAG' in key) and ('ERR' in key):
        band = key[-1].lower()
        if band in ('g', 'r', 'i'):
            fake_data.append( np.concatenate((data[key], sat.stars['mag_{}_err'.format(band)])) )
        else:
            fake_data.append( np.concatenate((data[key], np.tile(0, len(sat.stars)))) )
    else:
        fake_data.append( np.concatenate((data[key], np.tile(0, len(sat.stars)))) )

fake_data = np.array(fake_data)
fake_data = np.core.records.fromarrays(fake_data, data.dtype)

fits.writeto('datafiles/y6_gold_1_1_patch_simsat.fits', fake_data, overwrite=True)




