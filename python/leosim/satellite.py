import numpy as np
import os
from astropy.constants import G, M_earth, R_earth
import astropy.units as u
import galsim

import rubin_sim.phot_utils as photUtils
from rubin_sim.data import get_data_dir

RAD2DEG = 206265.

class BaseSatellite:
    """A base satellite object.

    Parameters
    ----------
    height : `astropy.units.Quantity`
        Orbital height.
    zangle : 'astropy.units.Quantity'
        Angle from zenith.
    """

    satellite_profile = None
    """Satellite surface brightness profile (galsim.gsobject.GSObject)."""

    def __init__(self, height, zangle):
           
        self.height = height.to(u.kilometer)
        self.zangle = zangle.to(u.degree)
        if zangle.value < 0.:
            raise ValueError('zangle {0.1f} cannot be less than 0 deg'.format(zangle.value))

        # Calculate distance to satellite
        x = np.arcsin(R_earth*np.sin(self.zangle)/(R_earth + self.height))
        if np.isclose(x.value, 0):
            distance = self.height
        else:
            distance = np.sin(zangle - x)*R_earth/np.sin(x)
        self.distance = distance.to(u.kilometer)

        # Set satellite SED
        self.sed = photUtils.Sed()
        self.sed.set_flat_sed()
        self.sed.flambda_tofnu()

    @property
    def orbital_omega(self):
        """Orbital angular velocity (`astropy.units.Quantity`, read-only)."""
        omega = np.sqrt(G*M_earth/(R_earth + self.height)**3)
        return omega.to(u.radian/u.second, equivalencies=u.dimensionless_angles())

    @property
    def orbital_velocity(self):
        """Orbital velocity (`astropy.units.Quantity`, read-only)."""
        v = self.orbital_omega*(R_earth + self.height)
        return v.to(u.meter/u.second, equivalencies=u.dimensionless_angles())

    @property
    def tangential_velocity(self):
        """Tangential velocity to line-of-sight (`astropy.units.Quantity`, 
        read-only).
        """
        x = np.arcsin(R_earth*np.sin(self.zangle)/(R_earth + self.height))
        return self.orbital_velocity*np.cos(x)

    @property
    def tangential_omega(self):
        """Tangential angular velocity to line-of-sight 
        (`astropy.units.Quantity`, read-only).
        """
        omega = self.tangential_velocity/(self.distance)
        return omega.to(u.radian/u.second, equivalencies=u.dimensionless_angles())

    def get_defocus_profile(self, instrument):

        outer_radius, inner_radius = instrument
        r_o = (outer_radius/self.distance).to_value(u.arcsecond, equivalencies=u.dimensionless_angles())
        r_i = (inner_radius/self.distance).to_value(u.arcsecond, equivalencies=u.dimensionless_angles())
        defocus = galsim.TopHat(r_o) - galsim.TopHat(r_i, flux=(r_i/r_o)**2.)

        return defocus_profile
    
    def get_flux(self, magnitude, band, plate_scale, gain=1.):
        """Calculate the number of ADU.

        Parameters
        ----------
        magnitude: `float`
            Stationary AB magnitude.
        band: `str`
            Name of filter band.
        plate_scale: `float`
            Plate scale of the instrument.
        gain: `float`
            Instrument gain in electrons per ADU.

        Returns
        -------
        adu : `float`
            Number of ADU.
        """
        dt = plate_scale/self.tangential_omega.to_value(u.arcsecond/u.second)

        filename = os.path.join(get_data_dir(), 'throughputs/baseline/total_{0}.dat'.format(band.lower()))
        bandpass = photUtils.Bandpass()
        bandpass.read_throughput(filename)
        photo_params = photUtils.PhotometricParameters(exptime=dt, nexp=1, gain=gain)

        m0_adu = self.sed.calc_adu(bandpass, phot_params=photo_params)
        adu = m0_adu*(10**(-magnitude/2.5))

        return adu

    def get_normalized_profile(self, seeing_profile, instrument, step_size, steps):

        defocus_profile = self.get_defocus_profile(instrument) 
        final_profile = galsim.Convolve([self.satellite_profile, defocus_profile, seeing_profile])
        image = final_profile.drawImage(scale=step_size, nx=steps, ny=steps)

        profile = np.sum(image.array, axis=0)
        normalized_profile = profile/np.max(profile)
        scale = np.linspace(-int(steps*step_size/2), int(steps*step_size/2), steps)

        return scale, normalized_profile

    def get_surface_brightness_profile(self, magnitude, band, seeing_profile, instrument, step_size, steps,
                                       gain=1.0, plate_scale=0.2):

        flux = self.get_flux(magnitude, band, plate_scale) 
        defocus_profile = self.get_defocus_profile(instrument)
        final_profile = galsim.Convolve([self.satellite_profile, defocus_profile, seeing_profile])
        final_profile = final_profile.withFlux(flux)
        image = final_profile.drawImage(scale=step_size, nx=steps, ny=steps)
       
        profile = np.sum(image.array, axis=0)*plate_scale/step_size
        scale = np.linspace(-int(steps*step_size/2), int(steps*step_size/2), steps)

        return scale, profile
   
class DiskSatellite(BaseSatellite):
    """A circular disk satellite.

    Parameters
    ----------
    height : `astropy.units.Quantity`
        Orbital height.
    zangle : 'astropy.units.Quantity'
        Angle from zenith.
    radius : `astropy.units.Quantity`
        Radius of satellite.
    """

    def __init__(self, height, zangle, radius):
        
        super().__init__(height, zangle)
        self.radius = radius.to(u.meter)
        r = (self.radius/self.distance).to_value(u.arcsecond, equivalencies=u.dimensionless_angles())
        self.satellite_profile = galsim.TopHat(r)
