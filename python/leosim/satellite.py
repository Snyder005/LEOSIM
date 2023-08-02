import numpy as np

import rubin_sim.phot_utils as photUtils

from leosim.profiles.convolution import convolve
from leosim.profiles.seeing import GausKolmogorov, VonKarman
from leosim.profiles.defocusing import FluxPerAngle
from leosim.profiles.objectprofiles import DiskSource, RectangularSource
from np.consts import G, M_earth, R_earth

class BaseSatellite:
    
    def __init__(self, height, zangle):
           
        self.height = height
        if zangle < 0.:
            raise ValueError('zangle {0.1f} cannot be less than 0 deg'.format(zangle))
        self.zangle = np.radians(zangle)

        ## Calculate relevant velocities
        h = self.height*1000.
        self.omega = np.sqrt(G*M_earth/(R_earth + h)**3)
        self.orbit_v = self.omega*(R_earth + h)

        x = np.arcsin(R_earth*np.sin(self.zangle)/(R_earth + h))
        if np.isclose(x, 0):
            self.distance = self.height
        else:
            self.distance = np.sin(self.zangle - x)*R_earth/np.sin(x)/1000.
            
        tan_v = self.orbit_v*np.cos(x)
        self.angular_v = tan_v*180.*60./(self.distance*1000.*np.pi)
        
        self.sed = photUtils.Sed()
        self.sed.set_flat_sed()
        self.sed.flambda_tofnu()
        self.source_profile = None
        
    def get_normalized_profile(self, seeing_fwhm, instrument, scale=None, atmosphere='GausKolmogorov'):
        
        if atmosphere == 'GausKolmogorov':
            seeing_profile = GausKolmogorov(seeing_fwhm, scale=scale)
        elif atmosphere == 'VonKarman':
            seeing_profile = VonKarman(seeing_fwhm, scale=scale)
        else:
            raise ValueError('{0} is not a supported atmosphere type.'.format(atmosphere))
        defocus_profile = FluxPerAngle(self.distance, instrument)
        streak_profile = convolve(self.source_profile, seeing_profile, defocus_profile)
        streak_profile.norm()        

        return streak_profile

    def get_surface_brightness_profile(self, magnitude, band, seeing_fwhm, instrument, gain=1.0, 
                                       plate_scale=0.2, scale=None, atmosphere='GausKolmogorov'):

        dt = plate_scale/(self.angular_v/60.*3600)

        filename = os.path.join(get_data_dir(), 'throughputs/baseline/total_{0}.dat'.format(band.lower()))
        bandpass = photUtils.Bandpass()
        bandpass.read_throughputs(filename)
        photo_params = photUtils.PhotometricParameters(exptime=dt, nexp=1, gain=gain)

        m0_adu = self.sed.calc_adu(bandpass, phot_params=photo_params)
        adu = m0_adu*(10**(-self.magnitude/2.5))

        streak_profile = get_normalized_profile(seeing_fwhm, instrument, scale=scale, atmosphere=atmosphere)
        
        streak_profile.obj = adu*streak_profile.obj/np.trapz(streak_profile.obj, x=streak_profile.scale/plate_scale)    
        streak_profile.scale = streak_profile.scale/plate_scale
   
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
