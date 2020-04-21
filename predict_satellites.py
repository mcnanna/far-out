#!/usr/bin/env python
import subprocess
import os.path
import numpy as np
import sympy as sy
import pickle
import copy
import ugali.utils.healpix
import astropy.io.fits as fits
import load_data
import significance
import argparse
import utils
import matplotlib.pyplot as plt
import matplotlib
import skymap
import plot_utils
import matplotlib.markers as mmarkers
from matplotlib.ticker import MaxNLocator
import percent
from helpers.SimulationAnalysis import SimulationAnalysis, iterTrees
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
        self.subhalos = subhalos

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
        fname = 'datafiles/elvis/{0}/{0}_accretion_rvir.npy'.format(pair)
        if os.path.isfile(fname):
            rvir_acc, rs_acc = np.load(fname)
        else:
            sim = SimulationAnalysis(trees_dir = 'datafiles/elvis/{}/trees'.format(pair))
            print "Loading M31 and MW main branches..."
            tree_M31 = sim.load_main_branch(self.halos.M31['id'])
            tree_MW = sim.load_main_branch(self.halos.MW['id'])

            #distance_cut = self.distance < 2000 # kpc 
            distance_cut = np.tile(True, len(subhalos))
            main_branches = []
            print "Loading {} subhalo branches...".format(sum(distance_cut))
            count = 0
            for subhalo in subhalos[distance_cut]:
                branch = sim.load_main_branch(subhalo['id'])
                main_branches.append(branch)
                count += 1
                percent.bar(count, sum(distance_cut))
            
            rvir_acc = np.zeros(len(main_branches))
            rs_acc = np.zeros(len(main_branches))
            for i in range(len(main_branches)):
                branch = main_branches[i]
                # Check if halo becomes M31 or MW subhalo
                M31_overlap = np.isin(branch['upid'], tree_M31['id'])
                MW_overlap = np.isin(branch['upid'], tree_MW['id'])
                if True in M31_overlap:
                    rvir_acc[i] = branch[M31_overlap]['rvir'][-1]
                    rs_acc[i] = branch[M31_overlap]['rs'][-1]
                elif True in MW_overlap:
                    rvir_acc[i] = branch[MW_overlap]['rvir'][-1]
                    rs_acc[i] = branch[MW_overlap]['rs'][-1]
                else:
                    peak_idx = np.argmax(branch['rvir'])
                    rvir_acc[i] = branch['rvir'][peak_idx]
                    rs_acc[i] = branch['rs'][peak_idx]

            np.save(fname, (rvir_acc, rs_acc))

        c = rvir_acc/rs_acc
        c_correction = (c/10.)**params.hyper['gamma_r']
        beta_correction = ((subhalos['vmax']/subhalos['vacc']).clip(max=1.0))**params.hyper['beta']
        halo_r12 = params.connection['A']*c_correction*beta_correction * ((rvir_acc/(params.hyper['R0']*params.cosmo['h']))**params.connection['n'])
        self.r_physical = np.random.lognormal(np.log(halo_r12), np.log(10)*params.connection['sigma_r']) # pc


    def __len__(self):
        return len(self.subhalos)

    def ra_dec(self, psi=0):
        transform = self.halos.rotation(psi)
        xp, yp, zp = np.dot(transform, np.array((self.x, self.y, self.z)))
        return utils.ra_dec(xp, yp, zp)


def main(pair):
    sats = Satellites(pair)
    if pair == 'RJ':
        psi = np.radians(66)
    elif pair == 'TL':
        psi = np.radians(330)
    sat_ras, sat_decs = sats.ra_dec(psi)

    close_cut = (sats.distance > 300)
    far_cut = (sats.distance < 2000)
    footprint = ugali.utils.healpix.read_map('datafiles/healpix_nest_y6a1_footprint_griz_frac05_nimages2.fits.gz')
    pix = ugali.utils.healpix.angToPix(4096, sat_ras, sat_decs, nest=True)
    footprint_cut = footprint[pix] > 0
    
    cut = footprint_cut & close_cut# & far_cut   


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('pair')
    p.add_argument('-t', '--table', action='store_true')
    p.add_argument('-c', '--count', action='store_true')
    p.add_argument('-r', '--rotations', default=60, type=int)
    p.add_argument('-p', '--plots', action='store_true')
    p.add_argument('-f', '--fname', default='default')
    args = p.parse_args()

    if args.fname == 'default':
        args.fname = 'sats_table_ellipse_{}'.format(args.pair)

    if args.table or args.count:
        sats = Satellites(args.pair)
        close_cut = (sats.distance > 300)
        far_cut = (sats.distance < 2000)
        #close_cut = (sats.distance <= 300)
        #far_cut = np.tile(True, len(sats))
        cut = close_cut & far_cut

    if args.table:
        print '\n Excluding {} satelites closer than 300 kpc and {} beyond 2000 kpc\n'.format(sum(~close_cut), sum(~far_cut))
        subprocess.call('mkdir -p sim_results/{}/'.format(args.pair).split())
        print "\nCalculating significances..."
        significance.create_sigma_table(sats.distance[cut], sats.M_r[cut], sats.r_physical[cut], outname='sim_results/{}/'.format(args.pair)+args.fname)

    if args.count:
        sigma_table = fits.open('sim_results/{}/'.format(args.pair)+args.fname+'.fits')[1].data
        subprocess.call('mkdir -p sim_results/{}/skymaps'.format(args.pair).split())
        footprint = ugali.utils.healpix.read_map('datafiles/healpix_nest_y6a1_footprint_griz_frac05_nimages2.fits.gz', nest=True) 

        total_sats = []
        detectable_sats = []

        print "\nPerforming rotations..."
        results = []
        for i in range(args.rotations):
            psi = 2*np.pi * float(i)/args.rotations
            sat_ras, sat_decs = sats.ra_dec(psi)
            sat_ras, sat_decs = sat_ras[cut], sat_decs[cut]
            sigmas = sigma_table['sigma']

            pix = ugali.utils.healpix.angToPix(4096, sat_ras, sat_decs, nest=True)
            footprint_cut = footprint[pix] > 0
            detectable_cut = sigmas > 6.0
            #print "psi = {}: {} total sats, {} detectable".format(psi, sum(footprint_cut), sum(footprint_cut & detectable_cut))
            results.append([footprint_cut, detectable_cut])

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
            #Add MW disk
            #import pdb; pdb.set_trace()
            disk_points = sats.halos.mw_disk()
            transform = sats.halos.rotation(psi)
            dx, dy, dz = np.dot(transform, disk_points.T)
            dras, ddecs = utils.ra_dec(dx, dy, dz)
            order = np.argsort(dras)
            dras, ddecs = dras[order], ddecs[order]
            smap.plot(dras, ddecs, latlon=True, c='green', lw=3, zorder=0)

            psideg = int(round(np.degrees(psi),0))
            plt.title('$\psi = {}^{{\circ}}$; {} total sats in footprint, {} detectable'.format(psideg, sum(footprint_cut), sum(footprint_cut & detectable_cut)))
            plt.savefig('sim_results/{0}/skymaps/{0}_skymap_psi={1:0>3d}.png'.format(args.pair, psideg), bbox_inches='tight')
            plt.close()
            percent.bar(i+1, args.rotations)

        # Merge skymaps into a .gif
        print '\nCreating .gif...'
        subprocess.call("convert -delay 30 -loop 0 sim_results/{0}/skymaps/*.png sim_results/{0}/{0}_skymap.gif".format(args.pair).split())
        print 'Done!'

        # Save results
        results = np.array(results)
        np.save('sim_results/{0}/{0}_results'.format(args.pair),results)

    
    if args.plots:
        # Plot results
        results = np.load('sim_results/{0}/{0}_results.npy'.format(args.pair))
        if args.pair == 'RJ':
            title = 'Romeo \& Juliet'
        elif args.pair == 'TL':
            title = 'Thelma \& Louise'

        def hist(result, xlabel, outname):
            mx = max(result)
            if mx<=30:
                bins = np.arange(mx+2)-0.5
            else:
                bins = 30
            plt.hist(result, bins=bins)
            if mx <= 20:
                xticks = range(mx+1)
                plt.xticks(xticks)
            plt.xlabel(xlabel)
            plt.title(title)
            plt.savefig('sim_results/{}/{}.png'.format(args.pair, outname), bbox_inches='tight')
            plt.close()

        total_sats = [sum(cuts[0]) for cuts in results]
        hist(total_sats, 'Satellites in footprint', 'total_sats_hist')
        detectable_sats = [sum(cuts[0]&cuts[1]) for cuts in results]
        hist(detectable_sats, 'Detectable satellites in foorprint', 'detectable_sats_hist')



