# To Do: make all instrument attributes static.
# Integrate throughputs and photometric parameters.

import astropy.units as u

class Instrument:
    """A class representing a telescope and camera.

    Parameters
    ----------
    outer_radius : `(astropy.units.Quantity)`
        Outer radius of the primary mirror.
    inner_radius : `(astropy.units.Quantity)`
        Inner radius of the primary mirror.
    pixel_scale : `(astropy.units.Quantity)`
        Pixel scale of the instrument camera.
    gain : `float`
        Gain of the instrument camera (electrons per ADU).
    """

    outer_radius = None
    """Outer radius of the primary mirror (astropy.units.Quantity)."""

    inner_radius = None
    """Inner radius of the primary mirror (astropy.units.Quantity)."""

    pixel_scale = None
    """Pixel scale of the instrument camera (astropy.units.Quantity)."""

    gain = None
    """Gain of the instrument camera (`float`)."""

    def __init__(self, outer_radius, inner_radius, pixel_scale, gain):
        self.outer_radius = outer_radius.to(u.m)
        self.inner_radius = inner_radius.to(u.m)
        self.pixel_scale = pixel_scale.to(u.arcsec/u.pix)
        self.gain = gain
