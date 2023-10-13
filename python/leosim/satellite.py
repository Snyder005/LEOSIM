# To Do: 
#   * Write getter/setter for satellite parameters.
#   * Make Sed a property? Can you still modify it? Is that needed? 
import numpy as np
import os
from astropy.constants import G, M_earth, R_earth
import astropy.units as u
import galsim

import rubin_sim.phot_utils as photUtils

from .component import *

class BaseSatellite:
    """A base satellite object.

    Parameters
    ----------
    height : `astropy.units.Quantity`
        Orbital height.
    zangle : `astropy.units.Quantity`
        Observed angle from zenith.
    """

    def __init__(self, height, zangle):
           
        self._height = height.to(u.km)
        if zangle.to_value(u.deg) < 0.:
            raise ValueError('zangle {0.1f} cannot be less than 0 deg'.format(zangle.value))
        self._zangle = zangle.to(u.deg)

        # Set satellite SED
        self._sed = photUtils.Sed()
        self._sed.set_flat_sed()
        self._profile = None

    @property
    def height(self):
        """Orbital height (`astropy.units.Quantity`)."""
        return self._height

    @height.setter
    def height(self, value):
        self._height = value.to(u.km)

    @property
    def zangle(self):
        """Observed angle from zenith (`astropy.units.Quantity`)."""
        return self._zangle

    @zangle.setter
    def zangle(self, value):
        self._zangle = value.to(u.deg)

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
    def sed(self):
        """Spectral energy distribution (`rubin_sim.phot_utils.sed.Sed`)."""
        return self._sed

    @property
    def profile(self):
        """Surface brightness profile (`galsim.gsobject.GSObject`, read-only)
        """
        return self._profile

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
        r_o = (instrument.outer_radius/self.distance).to_value(u.arcsec, 
                                                               equivalencies=u.dimensionless_angles())
        r_i = (instrument.inner_radius/self.distance).to_value(u.arcsec, 
                                                               equivalencies=u.dimensionless_angles())
        defocus_profile = galsim.TopHat(r_o) - galsim.TopHat(r_i, flux=(r_i/r_o)**2.)

        return defocus_profile
    
    def get_flux(self, magnitude, band, instrument, wavelen=None, fnu=None):
        """Calculate the number of ADU for a given observation.

        Parameters
        ----------
        magnitude: `float`
            Stationary AB magnitude.
        band: `str`
            Name of filter band.
        instrument : `leosim.instrument.Instrument`
            Instrument used for observation.
        wavelen : `numpy.ndarray`, optional
            Wavelength array for spectral energy distribution (nm).
        fnu : `numpy.ndarray`, optional
            Flux density array for spectral energy distribution (Jy).

        Returns
        -------
        adu : `float`
            Number of ADU.
        """
        dt = (instrument.pixel_scale/self.tangential_omega).to_value(u.s, equivalencies=[(u.pix, None)])

        photo_params = instrument.get_photo_params(exptime=dt)
        bandpass = instrument.get_bandpass(band)

        m0_adu = self.sed.calc_adu(bandpass, phot_params=photo_params, wavelen=wavelen, fnu=fnu)
        adu = m0_adu*(10**(-magnitude/2.5))

        return adu

    def get_stationary_profile(self, seeing_profile, instrument, magnitude=None, band=None, **flux_kwargs):

        defocus_profile = self.get_defocus_profile(instrument)
        stationary_profile = galsim.Convolve([self.profile, defocus_profile, seeing_profile])

        if (magnitude is not None) and (band is not None):
            adu = self.get_flux(magnitude, band, instrument, **flux_kwargs)
        else:
            adu = 1.0
        stationary_profile = stationary_profile.withFlux(adu)

        return stationary_profile

    def get_normalized_profile(self, seeing_profile, instrument, step_size, steps):

        defocus_profile = self.get_defocus_profile(instrument)
        final_profile = galsim.Convolve([self.profile, defocus_profile, seeing_profile])
        image = final_profile.drawImage(scale=step_size, nx=steps, ny=steps)

        profile = np.sum(image.array, axis=0)
        normalized_profile = profile/np.max(profile)
        scale = np.linspace(-int(steps*step_size/2), int(steps*step_size/2), steps)

        return scale, normalized_profile

    def get_surface_brightness_profile(self, magnitude, band, seeing_profile, instrument, step_size, steps):

        flux = self.get_flux(magnitude, band, instrument)
        defocus_profile = self.get_defocus_profile(instrument)
        final_profile = galsim.Convolve([self.profile, defocus_profile, seeing_profile])
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
        Radius of the satellite.
    """

    def __init__(self, height, zangle, radius): 
        super().__init__(height, zangle)
        self._radius = radius.to(u.m)
        r = (self.radius/self.distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        self._profile = galsim.TopHat(r)

    @property
    def radius(self):
        """Radius of the satellite (`astropy.units.Quantity`)."""
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value.to(u.m)

class RectangularSatellite(BaseSatellite):
    """A rectangular satellite.

    Parameters
    ----------
    height : `astropy.units.Quantity`
        Orbital height.
    zangle : `astropy.units.Quantity`
        Observed angle from zenith.
    width : `astropy.units.Quantity`
        Width of the satellite.
    length : `astropy.units.Quantity`
        Length of the satellite.
    """

    def __init__(self, height, zangle, width, length):
        super().__init__(height, zangle)
        self._width = width.to(u.m)
        self._length = length.to(u.m)
        w = (self.width/self.distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        l = (self.length/self.distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        self._profile = galsim.Box(w, l)

    @property
    def width(self):
        """Width of the satellite (`astropy.units.Quantity`)."""
        return self._width

    @width.setter
    def width(self, value):
        self._width = value.to(u.m)

    @property
    def length(self):
        """Length of the satellite (`astropy.units.Quantity`)."""
        return self._length

    @length.setter
    def length(self, value):
        self._length = value.to(u.m)

class ArbitrarySatellite(BaseSatellite):
    """An arbitrarily shaped satellite.

    Parameters
    ----------
    height : `astropy.units.Quantity`
        Orbital height.
    zangle : `astropy.units.Quantity`
        Observed angle from zenith.
    profile : `galsim.GSObject`
        Satellite surface brightness profile.
    """

    def __init__(self, height, zangle, profile):
        super().__init__(height, zangle)
        self._profile = profile

class ComponentSatellite(BaseSatellite):

    def __init__(self, height, zangle, components=[]):
        
        super().__init__(height, zangle)
        self.components = components
        
    def add_component(self, component):
        
        self.components.append(component)
        
    @property
    def profile(self):
        
        profile = self.components[0].create_profile(self.distance)
        
        for component in self.components[1:]:
            profile += component.create_profile(self.distance)
            
        return profile
