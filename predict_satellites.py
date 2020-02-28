#!/usr/bin/env python
import numpy as np
import pickle
import load_data
import patch_analysis
import argparse
import utils
import copy

def subhalo_ra_dec(halos_in, alpha=0.):
    """No where in this function have I thought carefully about how the azimuthal angle phi
    corresponds to DEC. I've just assumed they're equal. I'm hoping any inconsistencies 
    will cancel out"""
    
    halos = copy.deepcopy(halos_in)
    # Alpha in radians
    m31 = halos.M31
    m31_ra, m31_dec = 10.6846, 41.2692
    # Put MW at origin
    halos.centerOnMW()

    # alpha transformation
    x,y,z = halos.getCoords('cartesian')
    xp = x*np.cos(alpha) + y*np.sin(alpha)
    yp = y*np.cos(alpha) - x*np.sin(alpha)
    zp = z
    x,y,z = xp,yp,zp
    halos.setCoords('cartesian', x,y,z)

    # Get coords of M31 to determine beta
    # M31 ISN'T LOADING PROPERLY/NOT CHANGED YET
    x31, y31, z31, r31 = m31['x'], m31['y'], m31['z'], m31['rho']
    theta31 = np.radians(90-m31_ra) # The target theta, not the current theta
    #beta = np.arctan2( -y31*r31*np.cos(theta31) + np.sqrt(y31**2+z31**2-(r31*np.cos(theta31))**2), z31*r31*np.cos(theta31) + np.sqrt(y31**2+z31**2-(r31*np.cos(theta31))**2) )
    beta = np.arccos( (r31*np.cos(theta31)*z31 + y31*np.sqrt(-(r31*np.cos(theta31))**2+y31**2+z31**2)) / (y31**2+z31**2) )
    # beta transformation
    xp = x
    yp = y*np.cos(beta) + z*np.sin(beta)
    zp = z*np.cos(beta) - y*np.sin(beta)
    x,y,z = xp, yp, zp
    halos.setCoords('cartesian', x,y,z)

    # Get coords of M31 to determine gamma
    theta31, phi31 = m31['theta'], m31['phi']
    gamma = phi31 - np.radians(m31_dec)
    # gamma transformation
    rho, theta, phi = halos.getCoords('spherical')
    phip = phi - gamma
    halos.setCoords('spherical', rho,theta,phip)

    print 90-np.degrees(m31['theta']), np.degrees(m31['phi'])
    # Transformation is finished. Now, just convert theta,phi into RA, DEC
    ra = (90 - np.degrees(halos.subhalos['theta']))
    dec = np.degrees(halos.subhalos['phi'])
    return ra, dec

def satellite_properties(halos, parameters, vpeak_Mr_interp):
    satellite_properties = {}

    # Cut subhalo catalog
    subhalos = halos.subhalos
    mpeak_cut = (subhalos['mpeak']*(1-params.cosmo['omega_b']/params.cosmo['omega_m']) > 10**7)
    vpeak_cut = (subhalos['vpeak'] > params.hyper['vpeak_cut'])
    vmax_cut = (subhalos['vmax'] > params.hyper['vmax_cut'])
    cut = (mpeak_cut & vpeak_cut & vmax_cut)
    subhalos = subhalos[cut]

    # Calulate luminosities
    sort_idx = np.argsort(np.argsort(subhalos['vpeak']))
    # The interpolater will return a sorted list of Mr, so we need to re-sort it 
    # to match the original ordering of subhalos
    Mr_mean = vpeak_Mr_interp(subhalos['vpeak'], params.connection['alpha'])[sort_idx]
    L_mean = 10**((-1.*Mr_mean + 4.81)/2.5 + np.log10(2))
    L = np.random.lognormal(np.log(L_mean), np.log(10)*params.connection['sigma_M'])
    satellite_properties['M_r'] = -1*(2.5*(np.log10(L) - np.log10(2))-4.81)

    # Calculate positions
    MW = halos.MW()
    halox = params.hyper['chi']*(subhalos['x']-MW['x'])*(1000/params.cosmo['h'])
    haloy = params.hyper['chi']*(subhalos['y']-MW['y'])*(1000/params.cosmo['h'])
    haloz = params.hyper['chi']*(subhalos['z']-MW['z'])*(1000/params.cosmo['h'])
    satellite_properties['distance'] = np.sqrt(halox**2 + haloy**2 + haloz**2)*params.hyper['chi'] # kpc
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
    args = p.parse_args()

    halos = load_data.Halos(args.pair)
    params = load_data.Parameters()
    with open('datafiles/interpolator.pkl', 'rb') as interp:
        vpeak_Mr_interp = pickle.load(interp)

    sats = satellite_properties(halos, params, vpeak_Mr_interp)
    mag_cut = (sats['M_r'] > -10.0)
    close_cut = (sats['distance'] > 300)
    far_cut = (sats['distance'] < 2000)
    cut = mag_cut & close_cut & far_cut
    print '\n Excluding {} satelites closer than 300 kpc and {} beyond 2000 kpc'.format(sum(~close_cut), sum(~far_cut))
    print ' Additionally excluding {} satellites with M_r < -10.0, leaving {} total satellites\n'.format(sum(~mag_cut & close_cut & far_cut), sum(cut))
    patch_analysis.create_sigma_table(sats['distance'][cut], sats['M_r'][cut], sats['r_12'][cut], aperature_shape='ellipse', aperature_type='factor', outname='sats_table_ellipse_{}'.format(args.pair))


    
