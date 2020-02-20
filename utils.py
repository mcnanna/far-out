#!/usr/bin/env python
import numpy as np
import scipy
from ugali.isochrone.parsec import Bressan2012
import ugali.utils.projector
import matplotlib.pyplot as plt
import copy

def magToFlux(mag):
    """
    Convert from an AB magnitude to a flux (Jy)
    """
    return 3631. * 10**(-0.4 * mag)

def fluxToMag(flux):
    """
    Convert from flux (Jy) to AB magnitude
    """
    return -2.5 * np.log10(flux / 3631.)

def getFluxError(mag, mag_error):
    return magToFlux(mag) * mag_error / 1.0857362


def color_cut(g, r, i, color_tol=0.2):
    # Central line, fitted in cc_cut.py
    # y = mx + b
    m, b = 0.369485, -0.0055077
    def distance(x, y):
        return np.sqrt( (m*x+b-y)**2 / (m**2+1) )

    x = g-r
    y = r-i
    cut = distance(x, y) < color_tol
    #cut = (g - r > 0.4) & (g - r < 1.1) & (r - i < 0.5)
    cut &= r < 24.5
    cut &= i < 24.5
    return cut 

def iso_cut(iso, band_1, mag_1, band_2, mag_2, iso_tol=0.1):
    seps = iso.separation(band_1, mag_1, band_2, mag_2)
    return seps < iso_tol

def cut(iso, g, r, i, color_tol=0.2, iso_tol=0.1):
    c_cut = color_cut(g, r, i, color_tol)
    gr_cut = iso_cut(iso, 'g', g, 'r', r, iso_tol)
    ri_cut = iso_cut(iso, 'r', r, 'i', i, iso_tol)
    return c_cut & gr_cut & ri_cut

a, b = -2.51758, 4.86721 # From abs_mag.py
def mass_to_mag(stellar_mass):
    return a*np.log10(stellar_mass) + b
def mag_to_mass(m_v):
    return 10**((m_v-b)/a)


class Isochrone(Bressan2012):
    def __init__(self, distance):
        age = 10.
        metal_z = 0.0001
        distance_modulus = ugali.utils.projector.distanceToDistanceModulus(distance)
        super(Isochrone, self).__init__(survey='des', age=age, z=metal_z, distance_modulus=distance_modulus) # Default bands being g and r should be ok...
        delattr(self, 'mag') # I will be overwriting mag to a function

    def mag(self, band):
        return self.data[band]+self.distance_modulus

    def simulate(self, abs_mag, distance_modulus=None, **kwargs):
        """
        Simulate a set of stellar magnitudes (no uncertainty) for a
        satellite of a given stellar mass and distance.

        Parameters:
        -----------
        abs_mag : the absoulte V-band magnitude of the system
        distance_modulus : distance modulus of the system (if None takes from isochrone)
        kwargs : passed to iso.imf.sample

        Returns:
        --------
        mag_g, mag_r, mag_i : simulated magnitudes with length stellar_mass/iso.stellar_mass()
        """
        stellar_mass = mag_to_mass(abs_mag)
        if distance_modulus is None: distance_modulus = self.distance_modulus
        # Total number of stars in system
        n = int(round(stellar_mass / self.stellar_mass()))
        f_g = scipy.interpolate.interp1d(self.mass_init, self.mag('g'))
        f_r = scipy.interpolate.interp1d(self.mass_init, self.mag('r'))
        f_i = scipy.interpolate.interp1d(self.mass_init, self.mag('i'))
        mass_init_sample = self.imf.sample(n, np.min(self.mass_init), np.max(self.mass_init), **kwargs)
        mag_g_sample, mag_r_sample, mag_i_sample = f_g(mass_init_sample), f_r(mass_init_sample), f_i(mass_init_sample) 
        return mag_g_sample, mag_r_sample, mag_i_sample

    def separation(self, band_1, mag_1, band_2, mag_2):
        """ 
        Calculate the separation between a specific point and the
        isochrone in magnitude-magnitude space. Uses an interpolation

        ADW: Could speed this up...

        Parameters:
        -----------
        mag_1 : The magnitude of the test points in the first band
        mag_2 : The magnitude of the test points in the second band

        Returns:
        --------
        sep : Minimum separation between test points and isochrone interpolation
        """
        iso_mag_1 = self.mag(band_1)
        iso_mag_2 = self.mag(band_2)
        
        def interp_iso(iso_mag_1,iso_mag_2,mag_1,mag_2):
            interp_1 = scipy.interpolate.interp1d(iso_mag_1,iso_mag_2,bounds_error=False)
            interp_2 = scipy.interpolate.interp1d(iso_mag_2,iso_mag_1,bounds_error=False)

            dy = interp_1(mag_1) - mag_2
            dx = interp_2(mag_2) - mag_1

            dmag_1 = np.fabs(dx*dy) / (dx**2 + dy**2) * dy
            dmag_2 = np.fabs(dx*dy) / (dx**2 + dy**2) * dx

            return dmag_1, dmag_2

        # Separate the various stellar evolution stages
        if np.issubdtype(self.stage.dtype,np.number):
            sel = (self.stage < self.hb_stage)
        else:
            sel = (self.stage != self.hb_stage)

        # First do the MS/RGB
        rgb_mag_1 = iso_mag_1[sel]
        rgb_mag_2 = iso_mag_2[sel]
        dmag_1,dmag_2 = interp_iso(rgb_mag_1,rgb_mag_2,mag_1,mag_2)

        # Then do the HB (if it exists)
        if not np.all(sel):
            hb_mag_1 = iso_mag_1[~sel]
            hb_mag_2 = iso_mag_2[~sel]

            hb_dmag_1,hb_dmag_2 = interp_iso(hb_mag_1,hb_mag_2,mag_1,mag_2)

            dmag_1 = np.nanmin([np.abs(dmag_1),np.abs(hb_dmag_1)],axis=0)
            dmag_2 = np.nanmin([np.abs(dmag_2),np.abs(hb_dmag_2)],axis=0)

        #return dmag_1,dmag_2
        return np.sqrt(dmag_1**2 + dmag_2**2)

    def sample(self, mode='data', mass_steps=1000, mass_min=0.1, full_data_range=False):
        """Sample the isochrone in steps of mass interpolating between
        the originally defined isochrone points.

        Parameters:
        -----------
        mode : 
        mass_steps : 
        mass_min : Minimum mass [Msun]
        full_data_range :
        
        Returns:
        --------
        mass_init : Initial mass of each point
        mass_pdf : PDF of number of stars in each point
        mass_act : Actual (current mass) of each stellar point
        mag_g : Array of absolute magnitudes in g band (no distance modulus applied)
        mag_r : Array of absolute magnitudes in r band (no distance modulus applied)
        mag_i : Array of absolute magnitudes in i band (no distance modulus applied)
        """

        if full_data_range:
            # ADW: Might be depricated 02/10/2015
            # Generate points over full isochrone data range
            select = slice(None)
        else:
            # Not generating points for the post-AGB stars,
            # but still count those stars towards the normalization
            select = slice(self.index)

        mass_steps = int(mass_steps)

        mass_init = self.mass_init[select]
        mass_act = self.mass_act[select]
        mag_g = self.mag('g')[select]
        mag_r = self.mag('r')[select]
        mag_i = self.mag('i')[select]
        
        # ADW: Assume that the isochrones are pre-sorted by mass_init
        # This avoids some numerical instability from points that have the same
        # mass_init value (discontinuities in the isochrone).
        # ADW: Might consider using np.interp for speed
        mass_act_interpolation = scipy.interpolate.interp1d(mass_init, mass_act,assume_sorted=True)
        mag_g_interpolation = scipy.interpolate.interp1d(mass_init, mag_g,assume_sorted=True)
        mag_r_interpolation = scipy.interpolate.interp1d(mass_init, mag_r,assume_sorted=True)
        mag_i_interpolation = scipy.interpolate.interp1d(mass_init, mag_i,assume_sorted=True)

        # ADW: Any other modes possible?
        if mode=='data':
            # Mass interpolation with uniform coverage between data points from isochrone file 
            mass_interpolation = scipy.interpolate.interp1d(np.arange(len(mass_init)), mass_init)
            mass_array = mass_interpolation(np.linspace(0, len(mass_init)-1, mass_steps+1))
            d_mass = mass_array[1:] - mass_array[:-1]
            mass_init_array = np.sqrt(mass_array[1:] * mass_array[:-1])
            mass_pdf_array = d_mass * self.imf.pdf(mass_init_array, log_mode=False)
            mass_act_array = mass_act_interpolation(mass_init_array)
            mag_g_array = mag_g_interpolation(mass_init_array)
            mag_r_array = mag_r_interpolation(mass_init_array)
            mag_i_array = mag_i_interpolation(mass_init_array)

        # Horizontal branch dispersion
        if self.hb_spread and (self.stage==self.hb_stage).any():
            logger.debug("Performing dispersion of horizontal branch...")
            mass_init_min = self.mass_init[self.stage==self.hb_stage].min()
            mass_init_max = self.mass_init[self.stage==self.hb_stage].max()
            cut = (mass_init_array>mass_init_min)&(mass_init_array<mass_init_max)
            if isinstance(self.hb_spread,collections.Iterable):
                # Explicit dispersion spacing
                dispersion_array = self.hb_spread
                n = len(dispersion_array)
            else:
                # Default dispersion spacing
                dispersion = self.hb_spread
                spacing = 0.025
                n = int(round(2.0*self.hb_spread/spacing))
                if n % 2 != 1: n += 1
                dispersion_array = np.linspace(-dispersion, dispersion, n)

            # Reset original values
            mass_pdf_array[cut] = mass_pdf_array[cut] / float(n)

            # Isochrone values for points on the HB
            mass_init_hb = mass_init_array[cut]
            mass_pdf_hb = mass_pdf_array[cut]
            mass_act_hb = mass_act_array[cut]
            mag_g_hb = mag_g_array[cut]
            mag_r_hb = mag_r_array[cut]
            mag_i_hb = mag_i_array[cut]

            # Add dispersed values
            for dispersion in dispersion_array:
                if dispersion == 0.: continue
                msg = 'Dispersion=%-.4g, HB Points=%i, Iso Points=%i'%(dispersion,cut.sum(),len(mass_init_array))
                logger.debug(msg)

                mass_init_array = np.append(mass_init_array, mass_init_hb) 
                mass_pdf_array = np.append(mass_pdf_array, mass_pdf_hb)
                mass_act_array = np.append(mass_act_array, mass_act_hb) 
                mag_g_array = np.append(mag_g_array, mag_g_hb + dispersion)
                mag_r_array = np.append(mag_r_array, mag_r_hb + dispersion)
                mag_i_array = np.append(mag_i_array, mag_i_hb + dispersion)

        # Note that the mass_pdf_array is not generally normalized to unity
        # since the isochrone data range typically covers a different range
        # of initial masses
        #mass_pdf_array /= np.sum(mass_pdf_array) # ORIGINAL
        # Normalize to the number of stars in the satellite with mass > mass_min
        mass_pdf_array /= self.imf.integrate(mass_min, self.mass_init_upper_bound)
        out = np.vstack([mass_init_array,mass_pdf_array,mass_act_array,mag_g_array,mag_r_array,mag_i_array])
        return out


    def draw(self, band_1, band_2, **kwargs):
        ax = plt.gca()
        if kwargs.pop('cookie',None):
            # Broad cookie cutter
            defaults = dict(alpha=0.5, color='0.5', zorder=0, 
                            linewidth=15, linestyle='-')
        else:
            # Thin lines
            defaults = dict(color='k', linestyle='-')
        kwargs = dict(list(defaults.items())+list(kwargs.items()))

        iso = copy.deepcopy(self)
        iso.hb_spread = False
        mass_init,mass_pdf,mass_act,mag_g,mag_r,mag_i = iso.sample(mass_steps=1e3)

        bands = band_1+band_2
        sel = np.array(('g' in bands, 'r' in bands, 'i' in bands))
        mag_1, mag_2 = np.array((mag_g,mag_r,mag_i))[sel]
        mag = mag_1
        color = mag_1 - mag_2

        # Find discontinuities in the color magnitude distributions
        dmag = np.fabs(mag[1:]-mag[:-1])
        dcolor = np.fabs(color[1:]-color[:-1])
        idx = np.where( (dmag>1.0) | (dcolor>0.25))[0]
        # +1 to map from difference array to original array
        mags = np.split(mag,idx+1)
        colors = np.split(color,idx+1)

        for i,(c,m) in enumerate(zip(colors,mags)):
            if i > 0:
                kwargs['label'] = None
            ax.plot(c,m,**kwargs)
        return ax
