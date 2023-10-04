import astropy.units as u

class Instrument:
    """A class representing a telescope and camera.

    Parameters
    ----------
    outer_radius : `float`
        Outer radius of the primary mirror (meters).
    inner_radius : `float`
        Inner radius of the primary mirror (meters).
    plate_scale : `float`
        Plate scale of the instrument camera (arcsecond per pixel).
    """

    outer_radius = None
    """Outer radius of the primary mirror (astropy.units.Quantity)."""

    inner_radius = None
    """Inner radius of the primary mirror (astropy.units.Quantity)."""

    plate_scale = None
    """Plate scale of the instrument camera (astropy.units.Quantity)."""

    def __init__(self, outer_radius, inner_radius, plate_scale):
        self.outer_radius = outer_radius*u.meter
        self.inner_radius = inner_radius*u.meter
        self.plate_scale = plate_scale
