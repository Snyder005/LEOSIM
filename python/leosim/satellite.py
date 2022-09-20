import numpy as np

import rubin_sim.photUtils as photUtils
import syseng_throughputs as st

from leosim.profiles.convolution import convolve
from leosim.profiles.seeing import GausKolmogorov, VonKarman
from leosim.profiles.defocusing import FluxPerAngle
from leosim.profiles.objectprofiles import DiskSource, RectangularSource

LSSTM0ADU = {'u' : 7.070173e06,
             'g' : 2.543741e07,
             'r' : 2.073579e07,
             'i' : 1.576086e07,
             'z' : 1.092646e07,
             'Y' : 5.257237e06}

class BaseSatellite:
    
    def __init__(self, magnitude, height, zangle):
        
        G = 6.673E-11
        Me = 5.972E24
        Re = 6371E3

        if zangle < 0.:
            raise ValueError('zangle {0.1f} cannot be less than 0 deg'.format(zangle))
            
        self.magnitude = magnitude
        self.height = height
        self.zangle = np.radians(zangle)

        ## Calculate relevant velocities
        h = self.height*1000.
        self.omega = np.sqrt(G*Me/(Re+h)**3)
        self.orbit_v = self.omega*(Re+h)

        x = np.arcsin(Re*np.sin(self.zangle)/(Re+h))
        if np.isclose(x, 0):
            self.distance = self.height
        else:
            self.distance = np.sin(self.zangle-x)*Re/np.sin(x)/1000.
            
        tan_v = self.orbit_v*np.cos(x)
        self.angular_v = tan_v*180.*60./(self.distance*1000.*np.pi)
        
        self.sed = photUtils.Sed()
        self.sed.setFlatSED()
        self.sed.flambdaTofnu()
        self.profile = None
        
    def get_trail_profile(self, seeing_fwhm, instrument, scale=None, atmosphere='GausKolmogorov'):
        
        if atmosphere == 'GausKolmogorov':
            seeing_profile = GausKolmogorov(seeing_fwhm, scale=scale)
        elif atmosphere == 'VonKarman':
            seeing_profile = VonKarman(seeing_fwhm, scale=scale)
        else:
            raise ValueError('{0} is not a supported atmosphere type.'.format(atmosphere))
        defocus_profile = FluxPerAngle(self.distance, instrument)
        trail_profile = convolve(self.profile, seeing_profile, defocus_profile)
        
        return trail_profile
    
    def get_total_flux(self, band, effarea, pixel_scale, gain=1.0):

        dt = pixel_scale/(self.angular_v/60.*3600)
        defaultDirs = st.setDefaultDirs()
        hardware, system = st.buildHardwareAndSystem(defaultDirs)

        photo_params = photUtils.PhotometricParameters(exptime=dt, nexp=1, gain=gain, effarea=effarea,
                                                       readnoise=0.0, othernoise=0.0, darkcurrent=0.0)
        m0_adu = self.sed.calcADU(system[band], photParams=photo_params)

        adu = m0_adu*(10**(-self.magnitude/2.5))
        
        return adu
    
    @staticmethod
    def surface_brightness_profile(profile, adu, plate_scale):

        ratio = np.max(profile.obj)/np.trapz(profile.obj, x=profile.scale/plate_scale)
        
        x = profile.scale/plate_scale
        y = ratio*adu*profile.obj/np.max(profile.obj)

        return x, y

   
class DiskSatellite(BaseSatellite):
    
    def __init__(self, magnitude, height, zangle, radius):
        
        super().__init__(magnitude, height, zangle)
        self.radius = radius
        self.profile = DiskSource(self.distance, self.radius)
        
class RectSatellite(BaseSatellite):
    
    def __init__(self, magnitude, height, zangle, length, width):
        
        super().__init__(magnitude, height, zangle)
        self.length = length
        self.width = width
        self.profile = RectangularSource(self.distance, length, width)
