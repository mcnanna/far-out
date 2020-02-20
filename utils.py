#!/usr/bin/env python
import numpy as np

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
    cut &= r < 24.25
    cut &= i < 24.5
    return cut 

def cut(iso, g, r, i, gerr, rerr, ierr, color_tol=0.2, iso_tol=0.1):
    c_cut = color_cut(g, r, i, color_tol)
    gr_cut = iso.cut_separation('g', 'r', g, r, gerr, rerr, radius=iso_tol)
    ri_cut = iso.cut_separation('r', 'i', r, i, rerr, ierr, radius=iso_tol)
    return c_cut & gr_cut & ri_cut


a, b = -2.51758, 4.86721 # From abs_mag.py
def mass_to_mag(stellar_mass):
    return a*np.log10(stellar_mass) + b
def mag_to_mass(m_v):
    return 10**((m_v-b)/a)


