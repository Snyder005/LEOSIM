# To Do: 
#   * Write getter/setter for satellite parameters.
#   * Make Sed a property? Can you still modify it? Is that needed? 
import numpy as np
import os
from astropy.constants import G, M_earth, R_earth
import astropy.units as u
import galsim

import rubin_sim.phot_utils as photUtils
from rubin_sim.data import get_data_dir

class BaseSatellite:
    """A base satellite object.

    Parameters
    ----------
    height : `astropy.units.Quantity`
        Orbital height.
    zangle : `astropy.units.Quantity`
        Observed angle from zenith.
    """

    height = None
    """Orbital height (`astropy.units.Quantity`)."""

    zangle = None
    """Observed angle from zenith (`astropy.units.Quantity)."""

    sed = None
    """Spectral energy distribution (`rubin_sim.phot_utils.sed.Sed`)."""

    def __init__(self, height, zangle):
           
        self.height = height.to(u.km)
        if zangle.to_value(u.deg) < 0.:
            raise ValueError('zangle {0.1f} cannot be less than 0 deg'.format(zangle.value))
        self.zangle = zangle.to(u.deg)

        # Set satellite SED
        self.sed = photUtils.Sed()
        self.sed.set_flat_sed()
        self.sed.flambda_tofnu()

    @property
    def distance(self):
        """Distance to satellite (`astropy.units.Quantity`, read-only)."""
        x = np.arcsin(R_earth*np.sin(self.zangle)/(R_earth + self.height))
        if np.isclose(x.value, 0):
            distance = self.height
        else:
            distance = np.sin(self.zangle - x)*R_earth/np.sin(x)
        return distance.to(u.km)

    @property
    def satellite_profile(self):
        """Surface brightness profile (`galsim.gsobject.GSObject`, read-only)
        """
        return self._satellite_profile

    @property
    def orbital_omega(self):
        """Orbital angular velocity (`astropy.units.Quantity`, read-only)."""
        omega = np.sqrt(G*M_earth/(R_earth + self.height)**3)
        return omega.to(u.rad/u.s, equivalencies=u.dimensionless_angles())

    @property
    def orbital_velocity(self):
        """Orbital velocity (`astropy.units.Quantity`, read-only)."""
        v = self.orbital_omega*(R_earth + self.height)
        return v.to(u.m/u.s, equivalencies=u.dimensionless_angles())

    @property
    def tangential_velocity(self):
        """Velocity tangential to the line-of-sight (`astropy.units.Quantity`, 
        read-only).
        """
        x = np.arcsin(R_earth*np.sin(self.zangle)/(R_earth + self.height))
        return self.orbital_velocity*np.cos(x)

    @property
    def tangential_omega(self):
        """Angular velocity tangential to the line-of-sight 
        (`astropy.units.Quantity`, read-only).
        """
        omega = self.tangential_velocity/(self.distance)
        return omega.to(u.rad/u.s, equivalencies=u.dimensionless_angles())

    def get_defocus_profile(self, instrument):
        """Calculate a defocusing profile for a given instrument.

        Parameters
        ----------
        instrument : `leosim.instrument.Instrument`
            Instrument used for observation.
        
        Returns
        -------
        defocus_profile : `galsim.gsobject.GSObject`
            Defocusing profile
        """
        outer_radius, inner_radius = instrument
        r_o = (instrument.outer_radius/self.distance).to_value(u.arcsec, 
                                                               equivalencies=u.dimensionless_angles())
        r_i = (instrument.inner_radius/self.distance).to_value(u.arcsec, 
                                                               equivalencies=u.dimensionless_angles())
        defocus = galsim.TopHat(r_o) - galsim.TopHat(r_i, flux=(r_i/r_o)**2.)

        return defocus_profile
    
    def get_flux(self, magnitude, band, instrument):
        """Calculate the number of ADU.

        Parameters
        ----------
        magnitude: `float`
            Stationary AB magnitude.
        band: `str`
            Name of filter band.
        instrument : `leosim.instrument.Instrument`
            Instrument used for observation.

        Returns
        -------
        adu : `float`
            Number of ADU.
        """
        dt = (instrument.pixel_scale/self.tangential_omega).to_value(u.s, equivalencies=[(u.pix, None)])

        filename = os.path.join(get_data_dir(), 'throughputs/baseline/total_{0}.dat'.format(band.lower()))
        bandpass = photUtils.Bandpass()
        bandpass.read_throughput(filename)
        photo_params = photUtils.PhotometricParameters(exptime=dt, nexp=1, gain=instrument.gain)

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

    def get_surface_brightness_profile(self, magnitude, band, seeing_profile, instrument, step_size, steps):

        flux = self.get_flux(magnitude, band, instrument)
        defocus_profile = self.get_defocus_profile(instrument)
        final_profile = galsim.Convolve([self.satellite_profile, defocus_profile, seeing_profile])
        final_profile = final_profile.withFlux(flux)
        image = final_profile.drawImage(scale=step_size, nx=steps, ny=steps)
       
        profile = np.sum(image.array, axis=0)*instrument.pixel_scale.to_value(u.arcsec/u.pix)/step_size
        scale = np.linspace(-int(steps*step_size/2), int(steps*step_size/2), steps)

        return scale, profile
   
class DiskSatellite(BaseSatellite):
    """A circular disk satellite.

    Parameters
    ----------
    height : `astropy.units.Quantity`
        Orbital height.
    zangle : `astropy.units.Quantity`
        Observed angle from zenith.
    radius : `astropy.units.Quantity`
        Radius of the disk.
    """

    radius = None
    """Radius of the disk (`astropy.units.Quantity`)."""

    def __init__(self, height, zangle, radius): 
        super().__init__(height, zangle)
        self.radius = radius.to(u.m)
        r = (self.radius/self.distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        self._satellite_profile = galsim.TopHat(r)
