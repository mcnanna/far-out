#!/usr/env python

from astropy.io import fits
import numpy as np
import scipy
import ugali.utils.healpix
from utils import *

class Satellites:
    def __init__(self):
        self.master = np.recfromcsv('/Users/mcnanna/Research/y3-mw-sats/data/mw_sats_master.csv')
        self.all = master_csv[np.where(self.master['type2'] >= 0)[0]]
        self.dwarfs = master_csv[np.where(self.master['type2'] >= 3)[0]]
        
    def __getitem__(self, key):
        return self.master[np.where(self.master['name'] == name)[0]][0]


class Patch:
    def mag(self, n, typ='star'):
        out = 'SOF_'
        out += 'PSF_' if typ=='star' else 'BDF_'
        out += 'MAG_'
        out += n.upper()
        return out

    def magerr(self, n, typ='star'):
        s = self.mag(n, typ)
        key = 'MAG_'
        loc = s.find(key)
        out = s[:loc] + key + 'ERR_' + s[loc+len(key):]
        return out

    def __init__(self):
        data = fits.open('datafiles/y6_gold_1_1_patch.fits')[1].data
        self.data = data[data['FLAGS_GOLD'] < 4]

        # Set classification
        classifier = 'EXT_SOF'
        high_stars = data[data[classifier] == 0]
        low_stars = data[data[classifier] == 1]
        low_galaxies = data[data[classifier] == 2]
        high_galaxies = data[data[classifier] == 3]
        other = data[data[classifier] == -9]

        self.stars = np.sort(np.concatenate((high_stars, low_stars, low_galaxies)), order='COADD_OBJECT_ID')
        self.galaxies = high_galaxies

        self.center_ra = 35.5
        self.center_dec = -4.5


class Inputs:
    def getPhotoError(self, infile):
        d = np.recfromcsv(infile)

        x = d['mag']
        y = d['log_mag_err']

        x = np.insert(x, 0, -10.)
        y = np.insert(y, 0, y[0])

        f = scipy.interpolate.interp1d(x, y, bounds_error=False, fill_value=1.)

        return f

    def getCompleteness(self, infile):
        d = np.recfromcsv(infile)

        x = d['mag_r']
        y = d['eff_star']

        x = np.insert(x, 0, 16.)
        y = np.insert(y, 0, y[0])

        f = scipy.interpolate.interp1d(x, y, bounds_error=False, fill_value=0.)

        return f

    def __init__(self):
        self.log_photo_error = self.getPhotoError('datafiles/photo_error_model.csv')
        self.completeness = self.getCompleteness('datafiles/y3a2_stellar_classification_summary_ext2.csv')

        self.m_maglim_g = ugali.utils.healpix.read_map('datafiles/y6a1_raw_sys1.0_sof_v1_nside4096_ring_g_depth.fits.gz')
        self.m_maglim_r = ugali.utils.healpix.read_map('datafiles/y6a1_raw_sys1.0_sof_v1_nside4096_ring_r_depth.fits.gz')
        self.m_maglim_i = ugali.utils.healpix.read_map('datafiles/y6a1_raw_sys1.0_sof_v1_nside4096_ring_i_depth.fits.gz')

        self.m_ebv = ugali.utils.healpix.read_map('datafiles/ebv_sfd98_fullres_nside_4096_nest_equatorial.fits.gz')


