#!/usr/env python

from astropy.io import fits
import numpy as np
from numpy.lib.recfunctions import append_fields
import scipy
import ugali.utils.healpix
from utils import *
from helpers.SimulationAnalysis import readHlist

class Satellites:
    def __init__(self):
        self.master = np.recfromcsv('/Users/mcnanna/Research/y3-mw-sats/data/mw_sats_master.csv')
        self.all = self.master[np.where(self.master['type2'] >= 0)[0]]
        self.dwarfs = self.master[np.where(self.master['type2'] >= 3)[0]]
        
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


class Parameters:
    """
    params (dict): dict containing free parameters
    params['alpha'] (float): faint-end slope of satellite luminosity function
    params['sigma_M'] (float): lognormal scatter in M_V--V_peak relation (in dex)
    params['M50'] (float): mass at which 50% of halos host galaxies (in log10(M*/h))
    params['sigma_mpeak'] (float): scatter in galaxy occupation fraction
    params['B'] (float): subhalo disruption probability (due to baryons)
    params['A']: satellite size relation amplitude
    params['sigma_r']: satellite size relation scatter
    params['n']: satellite size relation slope

    hparams (dict): dict containing hyperparameters
    hparams['vpeak_cut']: subhalo vpeak resolution threshold
    hparams['vmax_cut']: subhalo vmax resolution threshold
    hparams['chi']: satellite radial scaling
    hparams['R0']: satellite size relation normalization
    hparams['gamma_r']: slope of concentration dependence in satellite size relation
    hparams['beta']: tidal stripping parameter
    hparams['O']: orphan satellite parameter

    cosmo_params (dict): dict containing cosmological parameters
    cosmo_params['omega_b']: baryon fraction
    cosmo_params['omega_m']: matter fraction
    cosmo_params['h']: dimensionless hubble parameter
    """

    def load_connectionparams(self):
        # Best fit parameters from Paper II
        params = {}
        params['alpha'] = -1.428
        params['sigma_M'] = 0.003
        params['M50'] = 7.51
        params['sigma_mpeak'] = 0.03 # sigma_gal
        params['B'] = 0.92
        params['A'] = 34
        params['sigma_r'] = 0.51
        params['n'] = 1.02
        return params

    def load_cosmoparams(self):
        cosmo_params = {}
        cosmo_params['omega_b'] = 0.0
        cosmo_params['omega_m'] = 0.286
        cosmo_params['h'] = 0.7
        return cosmo_params

    def load_hyperparams(self):
        hparams = {}
        hparams['vpeak_cut'] = 10. # Increase? Come from minimum trusted mass -> vpeak through NFW relations. Could increase probably
        hparams['vmax_cut'] = 9.
        hparams['chi'] = 1.
        hparams['R0'] = 10.0
        hparams['gamma_r'] = 0.0
        hparams['beta'] = 0.
        hparams['O'] = 1.
        return hparams

    def __init__(self):
        self.connection = self.load_connectionparams()
        self.cosmo = self.load_cosmoparams()
        self.hyper = self.load_hyperparams()

    def __getitem__(self, key):
        dics = (self.connection, self.cosmo, self.hyper)
        exception_count = 0
        for dic in dics:
            try:
                val = dic[key]
                return val
            except KeyError as e:
                exception_count += 1
                if exception_count == len(dics):
                    raise e
                else:
                    continue
    # I tried to make a __setitem__, but I couldn't make it work.
    # Ex: After running params['vpeak_cut'] = 5.1, calling
    # params['vpeak_cut'] == 5.1, but params.hyper['vpeak_cut'] != 5.1


class Halos():
    def __init__(self, pair):
        """Pair is either RJ (Romeo and Juliet) or TL (Thelma and Louise)"""
        fields = ['scale','id', 'upid', 'pid', 'mvir', 'mpeak', 'rvir', 'rs', 'vmax', 'vpeak', 'vacc', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'M200c', 'depth_first_id','scale_of_last_MM','jx','jy','jz']
        self.halos = readHlist('datafiles/elvis/{0}/hlist_1.00000.list'.format(pair), fields)

        if pair == 'TL':
            # Most massive subhalo is not MW or M31, so we must reorder 
            new_order = [1,2,0] + range(3,len(self.halos))
            self.halos = self.halos[np.array(new_order)]

        self.M31 = self.halos[0]
        self.MW = self.halos[1]
        self.subhalos = self.halos[2:]

        self.centerOnMW()

    def __getitem__(self, key):
        return self.halos[key]
    def __setitem__(self, key, val):
        self.halos[key] = val
    def __len__(self):
        return len(self.halos)

    def centerOnMW(self):
        x,y,z = self['x'], self['y'], self['z']
        mwx, mwy, mwz = self.MW['x'], self.MW['y'], self.MW['z']
        self['x'] = x-mwx
        self['y'] = y-mwy
        self['z'] = z-mwz

    def rotation(self, psi=0):
        """Rotation matrix to put M31 at its (RA, DEC), with arbitary rotation psi (radians)
        about MW-M31 vector."""

        m31_ra, m31_dec = 10.6846, 41.2692
        m31_theta = np.radians(90-m31_dec)
        m31_phi = np.radians(m31_ra)

        m31 = self.M31
        x31, y31, z31 = m31['x'], m31['y'], m31['z']
        r31 = np.sqrt(x31**2 + y31**2 + z31**2)
        # Normalized unit vector in direction of M31
        u = x31/r31
        v = y31/r31
        w = z31/r31

        # https://sites.google.com/site/glennmurray/Home/rotation-matrices-and-formulas/rotation-about-an-arbitrary-axis-in-3-dimensions
        # Rotate vector about z-axis to put it into the xz-plane
        Txz = np.array([[ u/np.sqrt(u**2+v**2), v/np.sqrt(u**2+v**2), 0],
                        [-v/np.sqrt(u**2+v**2), u/np.sqrt(u**2+v**2), 0],
                        [0,0,1]])
        # Rotate vector into the z-axis, about y-axis
        Tz = np.array([[w, 0, -np.sqrt(u**2+v**2)],
                       [0,1,0],
                       [np.sqrt(u**2+v**2), 0, w]])
        # Rotate about new z-axis by arbitrary angle psi
        Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                       [np.sin(psi),  np.cos(psi), 0],
                       [0,0,1]])
        # Rotate vector out of z-axis to desired theta
        Ttheta = np.array([[ np.cos(m31_theta), 0, np.sin(m31_theta)],
                          [0,1,0],
                          [-np.sin(m31_theta), 0, np.cos(m31_theta)]])
        # Rotate about z axis to get desired phi
        Tphi = np.array([[np.cos(m31_phi), -np.sin(m31_phi), 0],
                         [np.sin(m31_phi),  np.cos(m31_phi), 0],
                         [0,0,1]])

        transform = np.linalg.multi_dot((Tphi, Ttheta, Rz, Tz, Txz))
        return transform


    def mw_disk(self, n_points=1000):
        jx = self.MW['jx']
        jy = self.MW['jy']
        jz = self.MW['jz']
        # Normalize, probably unnecessary
        j = np.sqrt(jx**2+jy**2+jz**2)
        jx, jy, jz = jx/j, jy/j, jz/j

        rng = np.linspace(-1,1,n_points)

        disk_points = []
        for z in rng:
            a = 1 + jx**2/jy**2
            b = (2*jx*jz/jy**2)*z
            c = (1+jz**2/jy**2)*z**2 - 1

            discriminant = b**2 - 4*a*c
            if discriminant < 0:
                continue

            x1 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
            x2 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
            y1 = -1*(jx*x1 + jz*z)/jy
            y2 = -1*(jx*x2 + jz*z)/jy
            disk_points.append(np.array((x1,y1,z)))
            disk_points.append(np.array((x2,y2,z)))

        return np.array(disk_points)
