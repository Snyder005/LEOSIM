import numpy as np
import os
import galsim

import rubin_sim.phot_utils as photUtils
from rubin_sim.data import get_data_dir
from astropy.constants import G, M_earth, R_earth

RAD2DEG = 206265.

class BaseSatellite:
    
    def __init__(self, height, zangle):
           
        self.height = height
        if zangle < 0.:
            raise ValueError('zangle {0.1f} cannot be less than 0 deg'.format(zangle))
        self.zangle = np.radians(zangle)

        ## Calculate relevant velocities
        h = self.height*1000.
        self.omega = np.sqrt(G.value*M_earth.value/(R_earth.value + h)**3)
        self.orbit_v = self.omega*(R_earth.value + h)

        x = np.arcsin(R_earth.value*np.sin(self.zangle)/(R_earth.value + h))
        if np.isclose(x, 0):
            self.distance = self.height
        else:
            self.distance = np.sin(self.zangle - x)*R_earth.value/np.sin(x)/1000.
            
        tan_v = self.orbit_v*np.cos(x)
        self.angular_v = tan_v*180.*60./(self.distance*1000.*np.pi)
        
        self.sed = photUtils.Sed()
        self.sed.set_flat_sed()
        self.sed.flambda_tofnu()
        self.object = None
       
    def get_flux(self, magnitude, band, plate_scale, gain=1.):

        dt = plate_scale/(self.angular_v/60.*3600)

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
