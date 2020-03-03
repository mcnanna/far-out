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
import plot_utils
import matplotlib.markers as mmarkers
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
plt.ion()

class Satellites:
    def __init__(self,pair):
        self.halos = load_data.Halos(pair)
        params = load_data.Parameters()
        with open('datafiles/interpolator.pkl', 'rb') as interp:
            vpeak_Mr_interp = pickle.load(interp)

        # Cut subhalo catalog
        subhalos = self.halos.subhalos
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
        self.M_r = -1*(2.5*(np.log10(L) - np.log10(2))-4.81)

        # Calculate positions (in kpc)
        self.x = params.hyper['chi']*(subhalos['x'])*(1000/params.cosmo['h'])
        self.y = params.hyper['chi']*(subhalos['y'])*(1000/params.cosmo['h'])
        self.z = params.hyper['chi']*(subhalos['z'])*(1000/params.cosmo['h'])
        self.position = np.vstack((self.x, self.y, self.z)).T

        self.distance = np.sqrt(self.x**2+self.y**2+self.z**2) # kpc
        
        # Calculate sizes
        c = subhalos['rvir']/subhalos['rs']
        c_correction = (c/10.)**params.hyper['gamma_r']
        beta_correction = ((subhalos['vmax']/subhalos['vacc']).clip(max=1.0))**params.hyper['beta']
        halo_r12 = params.connection['A']*c_correction*beta_correction * ((subhalos['rvir']/(params.hyper['R0']*params.cosmo['h']))**params.connection['n'])
        self.r_physical = np.random.lognormal(np.log(halo_r12), np.log(10)*params.connection['sigma_r']) # pc

    
    def ra_dec(self, gamma=0.):
        m31_ra, m31_dec = 10.6846, 41.2692
        m31_theta = np.radians(90-m31_dec) # The target theta, not the current theta
        m31_phi = np.radians(m31_ra) # The target phi, not the current phi

        m31 = self.halos.M31
        x31, y31, z31 = m31['x'], m31['y'], m31['z'] # Scaling from chi and h shouldn't matter
        r31 = np.sqrt(x31**2 + y31**2 + z31**2)

        # Solve for beta:
        beta = sy.Symbol('beta')
        sols = sy.solvers.solve(-1*sy.sin(beta)*x31 + sy.cos(beta)*sy.sin(gamma)*y31 + sy.cos(beta)*sy.cos(gamma)*z31 - r31*sy.cos(m31_theta), beta, numerical=True, minimal=True)
        beta = float(sols[0]) # Arbitrary?

        # Solve for alpha
        alpha = sy.Symbol('alpha')
        sols = sy.solvers.solve(sy.cos(alpha)*sy.cos(beta)*x31 + 
                               (sy.cos(alpha)*sy.sin(beta)*sy.sin(gamma)-sy.sin(alpha)*sy.cos(gamma))*y31 + 
                               (sy.cos(alpha)*sy.sin(beta)*sy.cos(gamma)+sy.sin(alpha)*sy.sin(gamma))*z31 -
                               r31*sy.sin(m31_theta)*sy.cos(m31_phi), numerical=True, minimal=True)
        # Possible degeneracy in alpha
        for alpha in map(float,sols):
            R = np.array([[np.cos(alpha)*np.cos(beta), np.cos(alpha)*np.sin(beta)*np.sin(gamma)-np.sin(alpha)*np.cos(gamma), np.cos(alpha)*np.sin(beta)*np.cos(gamma)+np.sin(alpha)*np.sin(gamma)],
                          [np.sin(alpha)*np.cos(beta), np.sin(alpha)*np.sin(beta)*np.sin(gamma)+np.cos(alpha)*np.cos(gamma), np.sin(alpha)*np.sin(beta)*np.cos(gamma)-np.cos(alpha)*np.sin(gamma)],
                          [-1*np.sin(beta), np.cos(beta)*np.sin(gamma), np.cos(beta)*np.cos(gamma)]])

            # Check:
            x31p, y31p, z31p = np.dot(R, np.array((x31, y31, z31)))
            theta = np.arccos(z31p/np.sqrt(x31p**2+y31p**2+z31p**2))
            phi = np.arctan2(y31p,x31p)
            tol = 0.001
            if m31_theta-tol<theta<m31_theta+tol and m31_phi-tol<phi<m31_phi+tol:
                break
        else:
            raise Exception("No solution found")

        # Apply transformation to all halos
        x,y,z = self.x, self.y, self.z
        xp, yp, zp = np.dot(R, np.array((x,y,z)))
        # Convert xp,yp,zp into RA, DEC
        phi = np.arctan2(yp, xp)
        phi = np.array([(p if p>0 else p+2*np.pi) for p in phi])
        r = np.sqrt(xp**2 + yp**2 + zp**2)
        theta = np.arccos(zp/r)
        ra = np.degrees(phi)
        dec = 90-np.degrees(theta)

        return ra, dec
    

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('pair')
    p.add_argument('-t', '--table', action='store_true')
    p.add_argument('-c', '--count', action='store_true')
    p.add_argument('-f', '--fname', default='default')
    args = p.parse_args()

    if args.fname == 'default':
        args.fname = 'sats_table_ellipse_{}'.format(args.pair)

    sats = Satellites(args.pair)
    close_cut = (sats.distance > 300)
    far_cut = (sats.distance < 2000)
    cut = close_cut & far_cut

    #import warnings
    #warnings.filterwarnings('ignore')

    if args.table:
        print '\n Excluding {} satelites closer than 300 kpc and {} beyond 2000 kpc\n'.format(sum(~close_cut), sum(~far_cut))
        patch_analysis.create_sigma_table(sats.distance[cut], sats.M_r[cut], sats.distance[cut],  aperature_shape='ellipse', aperature_type='factor', outname='sim_results/{}/'.format(args.pair)+args.fname)

    if args.count:
        sigma_table = fits.open(args.fname+'.fits')[1].data
        footprint = ugali.utils.healpix.read_map('datafiles/healpix_nest_y6a1_footprint_griz_frac05_nimages2.fits.gz', nest=True) 

        total_sats = []
        detectable_sats = []

        rots = 30
        for i in range(rots):
            gamma = 2*np.pi * float(i)/rots
            sat_ras, sat_decs = sats.ra_dec(gamma)
            sat_ras, sat_decs = sat_ras[cut], sat_decs[cut]
            sigmas = sigma_table['sigma']

            pix = ugali.utils.healpix.angToPix(4096, sat_ras, sat_decs, nest=True)
            footprint_cut = footprint[pix] > 0
            detectable_cut = sigmas > 6.0
            print "gamma = {}: {} total sats, {} detectable".format(gamma, sum(footprint_cut), sum(footprint_cut & detectable_cut))
            total_sats.append(sum(footprint_cut))
            detectable_sats.append(sum(footprint_cut & detectable_cut))


            plt.figure(figsize=(12,8)) 
            smap = skymap.Skymap(projection='mbtfpq', lon_0=0)
            cmap=plot_utils.shiftedColorMap('seismic_r', min(sigmas), max(sigmas), 6)
            markers = np.tile('o', len(sigmas))
            markers[footprint_cut] = '*'
            sizes = np.tile(10.0, len(sigmas))
            sizes[footprint_cut] = 50.0
            def custom_scatter(smap,x,y,markers,**kwargs):
                sc = smap.scatter(x,y,**kwargs)
                
                paths=[]
                for marker in markers:
                    marker_obj = mmarkers.MarkerStyle(marker)
                    path = marker_obj.get_path().transformed(marker_obj.get_transform())
                    paths.append(path)
                sc.set_paths(paths)
                return sc
            custom_scatter(smap, sat_ras, sat_decs, c=sigmas, cmap=cmap, latlon=True, s=sizes, markers=markers, edgecolors='k', linewidths=0.2)
            plt.colorbar()
            #Add DES polygon
            des_poly = np.genfromtxt('/Users/mcnanna/Research/y3-mw-sats/data/round19_v0.txt',names=['ra','dec'])
            smap.plot(des_poly['ra'], des_poly['dec'], latlon=True, c='0.25', lw=3, alpha=0.3, zorder=0)
            plt.savefig('sim_results/{0}/{0}_skymap_gamma={1}.png'.format(args.pair, round(gamma,2)), bbox_inches='tight')
            plt.close()

        np.save('{}_total_sats'.format(args.pair),total_sats)
        np.save('{}_detectable_sats'.format(args.pair),detectable_sats)
            

