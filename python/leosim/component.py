import astropy.units as u
import galsim

class Component:
    
    def __init__(self, x0, y0, flux=1.0):
        
        self.x0 = x0
        self.y0 = y0
        self.flux = flux
        
class Panel(Component):
        
    def __init__(self, width, length, x0, y0, flux=1.0):
        
        super().__init__(x0, y0, flux=flux)
        
        self.width = width
        self.length = length
        
    def create_profile(self, distance):
        
        w = (self.width/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        l = (self.length/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        profile = galsim.Box(w, l)
        profile = profile.withFlux(self.flux)
        
        dx = (self.x0/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        dy = (self.y0/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        profile = profile.shift(dx, dy)
        
        return profile
    
class Bus(Component):
        
    def __init__(self, width, length, x0, y0, flux=1.0):
        
        super().__init__(x0, y0, flux=flux)
        
        self.width = width
        self.length = length
        
    def create_profile(self, distance):
        
        w = (self.width/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        l = (self.length/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        profile = galsim.Box(w, l)
        profile = profile.withFlux(self.flux)
        
        dx = (self.x0/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        dy = (self.y0/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        profile = profile.shift(dx, dy)
        
        return profile
    
class Dish(Component):
    
    def __init__(self, radius, x0, y0, flux=1.0):
        
        super().__init__(x0, y0, flux=flux)
        
        self.radius = radius
        
    def create_profile(self, distance):
        
        r = (self.radius/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        profile = galsim.TopHat(r)
        profile = profile.withFlux(self.flux)
        
        dx = (self.x0/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        dy = (self.y0/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        profile = profile.shift(dx, dy)
        
        return profile
