#!/usr/bin/env python
import numpy as np
import pickle
import load_data
import patch_analysis

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
    MW = halos.MW
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

halos = load_data.Halos('RJ')
params = load_data.Parameters()
with open('datafiles/interpolator.pkl', 'rb') as interp:
    vpeak_Mr_interp = pickle.load(interp)

sats = satellite_properties(halos, params, vpeak_Mr_interp)
mag_cut = (sats['M_r'] > -10.0)
print '\nExcluding {} satellites with M_r < -10.0\n'.format(sum(~mag_cut))
patch_analysis.create_sigma_matrix(sats['distance'][mag_cut], sats['M_r'][mag_cut], sats['r_12'][mag_cut], aperature_shape='ellipse', aperature_type='factor', outname='RJ_sats_table_ellipse')


    
