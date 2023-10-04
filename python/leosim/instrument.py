import astropy.units as u

class Instrument:
    """A class representing a telescope and camera.

    Parameters
    ----------
    outer_radius : `float`
        Outer radius of the primary mirror (meters).
    inner_radius : `float`
        Inner radius of the primary mirror (meters).
    pixel_scale : `float`
        Pixel scale of the instrument camera (arcsecond per pixel).
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
        self.outer_radius = outer_radius*u.m
        self.inner_radius = inner_radius*u.m
        self.pixel_scale = pixel_scale*u.arcsec/u.pix
        self.gain = gain
