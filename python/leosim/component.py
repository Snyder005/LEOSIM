# This file is part of obs_lsst.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

__all__ = ("Panel", "Bus", "Dish")

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
    flux : `float`, optional
        Total flux of component in electrons (1.0, by default).
    """

    flux=None
    """Total flux of component in electrons (`float`)."""
    
    def __init__(self, x0, y0, flux=1.0):        
        self._x0 = x0.to(u*m)
        self._y0 = y0.to(u*m)
        self.flux = flux

    @property
    def x0(self):
        """Component centroid position in x-direction 
        (`astropy.units.Quantity`, read-only).
        """
        return self._x0

    @property
    def y0(self):
        """Component centroid position in y-direction 
        (`astropy.units.Quantity`, read-only).
        """
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
    flux : `float`, optional
        Total flux of component in electrons (1.0, by default).
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
        profile : `galsim.GSObject`
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
    """A class representing a satellite bus.

    Parameters
    ----------
    x0 : `astropy.units.Quantity`
        Component centroid position in x-direction.
    y0 : `astropy.units.Quantity`
        Component centroid position in y-direction.
    width : `astropy.units.Quantity`
        Width of the bus.
    length : `astropy.units.Quantity`
        Length of the bus.
    flux : `float`, optional
        Total flux of component in electrons (1.0, by default).
    """
        
    def __init__(self, x0, y0, width, length, flux=1.0):
        super().__init__(x0, y0, flux=flux)
        
        self._width = width.to(u.m)
        self._length = length.to(u.m)

    @property
    def width(self):
        """Width of the bus (`astropy.units.Quantity`, read-only)."""
        return self._width

    @property
    def length(self):
        """Length of the bus (`astropy.units.Quantity`, read-only)."""
        return self._length
        
    def create_profile(self, distance):
        """Create bus profile.
    
        Parameters
        ----------
        distance : `astropy.units.Quantity`
            Distance to the satellite.

        Returns
        -------
        profile : `galsim.GSObject`
            Bus profile.
        """
        w = (self.width/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        l = (self.length/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        profile = galsim.Box(w, l)
        profile = profile.withFlux(self.flux)
        
        dx = (self.x0/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        dy = (self.y0/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        profile = profile.shift(dx, dy)
        
        return profile
    
class Dish(Component):
    """A class representing a satellite dish.

    Parameters
    ----------
    x0 : `astropy.units.Quantity`
        Component centroid position in x-direction.
    y0 : `astropy.units.Quantity`
        Component centroid position in y-direction.
    radius : `astropy.units.Quantity`
        Radius of the dish.
    flux : `float`, optional
        Total flux of component in electrons (1.0, by default).
    """
    
    def __init__(self, x0, y0, radius, flux=1.0):
        super().__init__(x0, y0, flux=flux)
        
        self._radius = radius

    @property
    def radius(self):
        """Radius of the dish (`astropy.units.Quantity`, read-only)."""
        return self._radius
        
    def create_profile(self, distance):
        """Create dish profile.
    
        Parameters
        ----------
        distance : `astropy.units.Quantity`
            Distance to the satellite.

        Returns
        -------
        profile : `galsim.GSObject`
            Dish profile.
        """
        r = (self.radius/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        profile = galsim.TopHat(r)
        profile = profile.withFlux(self.flux)
        
        dx = (self.x0/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        dy = (self.y0/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        profile = profile.shift(dx, dy)
        
        return profile
