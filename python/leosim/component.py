import astropy.units as u
import galsim

class Component:
    """A class representing a satellite component.

    Parameters
    ----------
    x0 : `astropy.units.Quantity`
        Component centroid position in x-direction.
    y0 : `astropy.units.Quantity`
        Component centroid position in y-direction.
    flux : `float`
        Total flux of component in electrons.
    """

    flux=None
    """Total flux of the component in electrons (`float`)."""
    
    def __init__(self, x0, y0, flux=1.0):
        
        self._x0 = x0.to(u*m)
        self._y0 = y0.to(u*m)
        self.flux = flux

    @property
    def x0(self):
        """Component centroid position in x-direction 
        (`astropy.units.Quantity`, read-only)."""
        return self._x0

    @property
    def y0(self):
        """Component centroid position in y-direction 
        (`astropy.units.Quantity`, read-only)."""
        return self._y0
        
class Panel(Component):
    """A class representing a satellite solar panel.

    Parameters
    ----------
    x0 : `astropy.units.Quantity`
        Component centroid position in x-direction.
    y0 : `astropy.units.Quantity`
        Component centroid position in y-direction.
    width : `astropy.units.Quantity`
        Width of the solar panel.
    length : `astropy.units.Quantity`
        Length of the solar panel.
    flux : `float`
        Total flux of component in electrons.
    """
        
    def __init__(self, x0, y0, width, length, flux=1.0):
        
        super().__init__(x0, y0, flux=flux)
        
        self._width = width.to(u.m)
        self._length = length.to(u.m)

    @property
    def width(self):
        """Width of the solar panel (`astropy.units.Quantity`, read-only)."""
        return self._width

    @property
    def length(self):
        """Length of the solar panel (`astropy.units.Quantity`, read-only)."""
        return self._length        

    def create_profile(self, distance):
        """Create solar panel profile.
    
        Parameters
        ----------
        distance : `astropy.units.Quantity`
            Distance to the satellite.

        Returns
        -------
        profile : `galsim.gsobject.GSObject`
            Solar panel profile.
        """
        w = (self.width/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        l = (self.length/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        profile = galsim.Box(w, l)
        profile = profile.withFlux(self.flux)
        
        dx = (self.x0/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        dy = (self.y0/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        profile = profile.shift(dx, dy)
        
        return profile
    
class Bus(Component):
        
    def __init__(self, x0, y0, width, length, flux=1.0):
        
        super().__init__(x0, y0, flux=flux)
        
        self._width = width.to(u.m)
        self._length = length.to(u.m)

    @property
    def width(self):
        return self._width

    @property
    def length(self):
        return self._length
        
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
    
    def __init__(self, x0, y0, radius, flux=1.0):
        
        super().__init__(x0, y0, flux=flux)
        
        self._radius = radius

    @property
    def radius(self):
        return self._radius
        
    def create_profile(self, distance):
        
        r = (self.radius/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        profile = galsim.TopHat(r)
        profile = profile.withFlux(self.flux)
        
        dx = (self.x0/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        dy = (self.y0/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        profile = profile.shift(dx, dy)
        
        return profile
