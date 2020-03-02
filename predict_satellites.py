#!/usr/bin/env python
import numpy as np
import sympy as sy
import pickle
import copy
import ugali.utils.healpix
import astropy.io.fits as fits
import load_data
import patch_analysis
import argparse
import utils
import matplotlib.pyplot as plt
import matplotlib
import skymap
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
plt.ion()

def subhalo_ra_dec(halos_in, gamma=0.):
    """Nowhere in this function have I thought carefully about how the azimuthal angle phi
    corresponds to DEC. I've just assumed they're equal. I'm hoping any inconsistencies 
    will cancel out
    alpha in radiaus"""
    
    m31_ra, m31_dec = 10.6846, 41.2692
    theta = np.radians(90-m31_dec) # The target theta, not the current theta
    phi = np.radians(m31_ra) # The target phi, not the current phi

    halos = copy.deepcopy(halos_in)
    # Put MW at origin
    halos.centerOnMW()
    
    m31 = halos.M31
    x31, y31, z31, r31 = m31['x'], m31['y'], m31['z'], m31['rho']

    # Solve for beta:
    beta = sy.Symbol('beta')
    sols = sy.solvers.solve(-sy.sin(beta)*x31 + sy.cos(beta)*sy.sin(gamma)*y31 + sy.cos(beta)*sy.cos(gamma)*z31 - r31*sy.cos(theta), beta, numerical=True, minimal=True)
    beta = float(sols[0]) # Arbitrary?

    # Solve for alpha
    alpha = sy.Symbol('alpha')
    sols = sy.solvers.solve(sy.cos(alpha)*sy.cos(beta)*x31 + 
                           (sy.cos(alpha)*sy.sin(beta)*sy.sin(gamma)-sy.sin(alpha)*sy.cos(gamma))*y31 + 
                           (sy.cos(alpha)*sy.sin(beta)*sy.cos(gamma)+sy.sin(alpha)*sy.sin(gamma))*z31 -
                           r31*sy.sin(theta)*sy.cos(phi), numerical=True, minimal=True)
    alpha = float(sols[0]) # Arbitrary?

    # Apply transformation to all halos
    x,y,z = halos.getCoords('cartesian')
    R = np.array([[np.cos(alpha)*np.cos(beta), np.cos(alpha)*np.sin(beta)*np.sin(gamma)-np.sin(alpha)*np.cos(gamma), np.cos(alpha)*np.sin(beta)*np.cos(gamma)-np.sin(alpha)*np.sin(gamma)],
                  [np.sin(alpha)*np.cos(beta), np.sin(alpha)*np.sin(beta)*np.sin(gamma)+np.cos(alpha)*np.cos(gamma), np.sin(alpha)*np.sin(beta)*np.cos(gamma)-np.cos(alpha)*np.sin(gamma)],
                  [-1*np.sin(beta), np.cos(beta)*np.sin(gamma), np.cos(beta)*np.cos(gamma)]])
    xp, yp, zp = np.dot(R, np.array((x,y,z)))
    halos.setCoords('cartesian', xp, yp, zp)

    """
    # Check alpha is compatible:
    m31 = halos.M31
    x31, y31, z31, r31 = m31['x'], m31['y'], m31['z'], m31['rho']
    print x31,y31,z31,r31,theta31
    if not r31**2 * np.sin(theta31)**2 > (x31*np.cos(alpha) + y31*np.sin(alpha))**2:
        print 'Non-compatible alpha'
        return

    # alpha transformation
    x,y,z = halos.getCoords('cartesian')
    xp = x*np.cos(alpha) + y*np.sin(alpha)
    yp = y*np.cos(alpha) - x*np.sin(alpha)
    zp = z
    x,y,z = xp,yp,zp
    halos.setCoords('cartesian', x,y,z)

    # Get new coords of M31 to determine beta
    x31, y31, z31, r31 = m31['x'], m31['y'], m31['z'], m31['rho']
    beta = np.arccos( (r31*np.cos(theta31)*z31 + y31*np.sqrt(-(r31*np.cos(theta31))**2+y31**2+z31**2)) / (y31**2+z31**2) )
    # beta transformation
    xp = x
    yp = y*np.cos(beta) + z*np.sin(beta)
    zp = z*np.cos(beta) - y*np.sin(beta)
    x,y,z = xp, yp, zp
    halos.setCoords('cartesian', x,y,z)

    # Get coords of M31 to determine gamma
    theta31, phi31 = m31['theta'], m31['phi']
    gamma = phi31 - np.radians(m31_ra)
    # gamma transformation
    rho, theta, phi = halos.getCoords('spherical')
    phip = phi - gamma
    halos.setCoords('spherical', rho,theta,phip)
    """
    # Check:
    #print np.degrees(m31['phi']), 90-np.degrees(m31['theta'])
    #raw_input()

    # Transformation is finished. Now, just convert theta,phi into RA, DEC
    ra = np.degrees(halos.subhalos['phi'])
    dec = (90 - np.degrees(halos.subhalos['theta']))
    return ra, dec

def satellite_properties(halos, parameters, vpeak_Mr_interp):
    satellite_properties = {}

    # Cut subhalo catalog
    halos.centerOnMW()
    subhalos = halos.subhalos
    #mpeak_cut = (subhalos['mpeak']*(1-params.cosmo['omega_b']/params.cosmo['omega_m']) > 10**7)
    #vpeak_cut = (subhalos['vpeak'] > params.hyper['vpeak_cut'])
    #vmax_cut = (subhalos['vmax'] > params.hyper['vmax_cut'])
    #cut = (mpeak_cut & vpeak_cut & vmax_cut)
    #subhalos = subhalos[cut]

    # Calulate luminosities
    sort_idx = np.argsort(np.argsort(subhalos['vpeak']))
    # The interpolater will return a sorted list of Mr, so we need to re-sort it 
    # to match the original ordering of subhalos
    Mr_mean = vpeak_Mr_interp(subhalos['vpeak'], params.connection['alpha'])[sort_idx]
    L_mean = 10**((-1.*Mr_mean + 4.81)/2.5 + np.log10(2))
    L = np.random.lognormal(np.log(L_mean), np.log(10)*params.connection['sigma_M'])
    satellite_properties['M_r'] = -1*(2.5*(np.log10(L) - np.log10(2))-4.81)

    # Calculate positions
    satellite_properties['distance'] = subhalos['rho']*params.hyper['chi']*(1000/params.cosmo['h'])

    halox = (subhalos['x'])*(1000/params.cosmo['h'])
    haloy = (subhalos['y'])*(1000/params.cosmo['h'])
    haloz = (subhalos['z'])*(1000/params.cosmo['h'])
    satellite_properties['pos'] = np.vstack((halox, haloy, halox)).T
    
    # Calculate sizes
    c = subhalos['rvir']/subhalos['rs']
    c_correction = (c/10.)**params.hyper['gamma_r']
    beta_correction = ((subhalos['vmax']/subhalos['vacc']).clip(max=1.0))**params.hyper['beta']
    halo_r12 = params.connection['A']*c_correction*beta_correction * ((subhalos['rvir']/(params.hyper['R0']*params.cosmo['h']))**params.connection['n'])
    satellite_properties['r_12'] = np.random.lognormal(np.log(halo_r12), np.log(10)*params.connection['sigma_r']) # pc

    return satellite_properties

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('pair')
    p.add_argument('-t', '--table', action='store_true')
    p.add_argument('-c', '--count', action='store_true')
    args = p.parse_args()

    halos = load_data.Halos(args.pair)

    params = load_data.Parameters()
    with open('datafiles/interpolator.pkl', 'rb') as interp:
        vpeak_Mr_interp = pickle.load(interp)
    sats = satellite_properties(halos, params, vpeak_Mr_interp)
    close_cut = (sats['distance'] > 300)
    far_cut = (sats['distance'] < 2000)
    cut = close_cut & far_cut

    if args.table:
        print '\n Excluding {} satelites closer than 300 kpc and {} beyond 2000 kpc'.format(sum(~close_cut), sum(~far_cut))
        patch_analysis.create_sigma_table(sats['distance'][cut], sats['M_r'][cut], sats['r_12'][cut], aperature_shape='ellipse', aperature_type='factor', outname='sats_table_ellipse_{}'.format(args.pair))

    if args.count:
        sats = fits.open('sats_table_ellipse_{}.fits'.format(args.pair))[1].data
        footprint = ugali.utils.healpix.read_map('datafiles/healpix_nest_y6a1_footprint_griz_frac05_nimages2.fits.gz', nest=True) 

        total_sats = []
        detectable_sats = []

        rots = 10
        for i in range(rots):
            gamma = 2*np.pi * float(i)/rots
            sat_ras, sat_decs = subhalo_ra_dec(halos, gamma)
            #smap = skymap.Skymap(projection='moll')
            #smap.scatter(sat_ras, sat_decs, s=1.0, latlon=True)
            #raw_input()
            pix = ugali.utils.healpix.angToPix(4096, sat_ras[cut], sat_decs[cut], nest=True)
            footprint_cut = footprint[pix] > 0

            total_sats.append(sum(footprint_cut))
            detectable_cut = sats['sigma'] > 6.0
            detectable_sats.append(sum(footprint_cut & detectable_cut))

            print "gamma = {}: {} total sats, {} detectable".format(gamma, sum(footprint_cut), sum(footprint_cut & detectable_cut))
