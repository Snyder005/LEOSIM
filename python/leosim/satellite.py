import numpy as np
import os
from astropy.constants import G, M_earth, R_earth
import astropy.units as u
import galsim

import rubin_sim.phot_utils as photUtils
from rubin_sim.data import get_data_dir

RAD2DEG = 206265.

class BaseSatellite:
    """Base class representing a satellite.

    Parameters
    ----------
    height : `astropy.units.Quantity`
        Orbital height of satellite.
    zangle : 'astropy.units.Quantity'
        Angle of satellite from zenith.
    """    
    def __init__(self, height, zangle):
           
        self.height = height.to(u.kilometer)
        self.zangle = zangle.to(u.degree)
        if zangle.value < 0.:
            raise ValueError('zangle {0.1f} cannot be less than 0 deg'.format(zangle.value))

        x = np.arcsin(R_earth*np.sin(self.zangle)/(R_earth + self.height))
        if np.isclose(x.value, 0):
            distance = self.height
        else:
            distance = np.sin(zangle - x)*R_earth/np.sin(x)
        self.distance = distance.to(u.kilometer)

        self.sed = photUtils.Sed()
        self.sed.set_flat_sed()
        self.sed.flambda_tofnu()
        self.object = None

    @property
    def orbital_omega(self): # rad/s
        return np.sqrt(G*M_earth/(R_earth + self.height)**3)

    @property
    def orbital_velocity(self): # m/s
        return self.orbital_omega*(R_earth + self.height)

    @property
    def tangential_velocity(self): # m/s
        x = np.arcsin(R_earth*np.sin(self.zangle)/(R_earth + self.height))
        return self.orbital_velocity*np.cos(x)

    @property
    def tangential_omega(self): # rad/s
        return self.tangential_velocity/(self.distance**np.pi)
       
    # Need to ensure above angular velocities are in rad/second not dimensionless/second
    def get_flux(self, magnitude, band, plate_scale, gain=1.):

        dt = plate_scale/(self.tangential_omega/60.*3600) ## converting to arcsec/sec?

        filename = os.path.join(get_data_dir(), 'throughputs/baseline/total_{0}.dat'.format(band.lower()))
        bandpass = photUtils.Bandpass()
        bandpass.read_throughput(filename)
        photo_params = photUtils.PhotometricParameters(exptime=dt, nexp=1, gain=gain)

        m0_adu = self.sed.calc_adu(bandpass, phot_params=photo_params)
        adu = m0_adu*(10**(-magnitude/2.5))

        return adu

    def get_normalized_profile(self, seeing_psf, instrument, step_size, steps):

        outer_radius, inner_radius = instrument
        r_o = (outer_radius/(self.distance*1000.))*RAD2DEG
        r_i = (inner_radius/(self.distance*1000.))*RAD2DEG
        defocus = galsim.TopHat(r_o) - galsim.TopHat(r_i, flux=(r_i/r_o)**2.)
 
        final = galsim.Convolve([self.object, defocus, seeing_psf])
        image = final.drawImage(scale=step_size, nx=steps, ny=steps)
        profile = np.sum(image.array, axis=0)
        normalized_profile = profile/np.max(profile)
        scale = np.linspace(-int(steps*step_size/2), int(steps*step_size/2), steps)

        return scale, normalized_profile

    def get_surface_brightness_profile(self, magnitude, band, seeing_psf, instrument, step_size, steps, gain=1.0, 
                                       plate_scale=0.2):

        flux = self.get_flux(magnitude, band, plate_scale) 
        outer_radius, inner_radius = instrument
        r_o = (outer_radius/(self.distance*1000.))*RAD2DEG
        r_i = (inner_radius/(self.distance*1000.))*RAD2DEG
        defocus = galsim.TopHat(r_o) - galsim.TopHat(r_i, flux=(r_i/r_o)**2.)
 
        final = galsim.Convolve([self.object, defocus, seeing_psf])
        final = final.withFlux(flux)
        image = final.drawImage(scale=step_size, nx=steps, ny=steps)
        
        profile = np.sum(image.array, axis=0)*plate_scale/step_size
        scale = np.linspace(-int(steps*step_size/2), int(steps*step_size/2), steps)

        return scale, profile
   
class DiskSatellite(BaseSatellite):
    
    def __init__(self, height, zangle, radius):
        
        super().__init__(height, zangle)
        self.radius = radius
        r_sat = (radius/(self.distance*1000.))*RAD2DEG
        self.object = galsim.TopHat(r_sat)
