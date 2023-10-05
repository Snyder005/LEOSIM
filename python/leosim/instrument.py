import os
import astropy.units as u

import rubin_sim.phot_utils as photUtils
from rubin_sim.data import get_data_dir

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
    gain : `float`, optional
        Gain of the instrument camera (electrons per ADU). Default is 1.

    Raises
    ------
    ValueError
        Raised if parameter ``outer_radius`` is not greater than parameter
        ``inner_radius``.
    """

    gain = None
    """Gain of the instrument camera (`float`)."""

    def __init__(self, outer_radius, inner_radius, pixel_scale, gain=1.0):

        if outer_radiuas < inner_radius:
            raise ValueError("Outer radius must be greater than inner radius.") 
        self._outer_radius = outer_radius.to(u.m)
        self._inner_radius = inner_radius.to(u.m)
        self._pixel_scale = pixel_scale.to(u.arcsec/u.pix)
        self.gain = gain

    @property
    def outer_radius(self):
        """Outer radius of the primary mirror (`astropy.units.Quantity`, 
        read-only).
        """
        return self._outer_radius

    @property
    def inner_radius(self):
        """Inner radius of the primary mirror (`astropy.units.Quantity`, 
        read-only).
        """
        return self._inner_radius

    @property
    def pixel_scale(self):
        """Pixel scale of camera (`astropy.units.Quantity`, read-only)."""
        return self._pixel_scale

    @property
    def effarea(self):
        """Effective collecting area (`astropy.units.Quantity`, read-only)."""
        return np.pi*(outer_radius**2 - inner_radius**2)

    def get_photo_params(self, exptime):
        """Generate photometric parameters for a given exposure.

        Parameters
        ----------
        exptime : `float`
            Exposure time (seconds).

        Returns
        -------
        photo_params : `rubin_sim.phot_utils.PhotometricParameters`
            Photometric parameters for the exposure.
        """
        pixel_scale = self.pixel_scale.to_value(u.arcsec/u.pix)
        effarea = self.effarea.to_value(u.cm*u.cm)
        photo_params = photUtils.PhotometricParameters(exptime=exptime, nexp=1, effarea=effarea,
                                                       gain=self.gain, platescale=pixel_scale)
        return photUtils.PhotometricParameters(exptime=exptime, nexp=1)

    @staticmethod
    def get_bandpass(bandname):
        """Get the telescope bandpass throughput curve.

        Parameters
        ----------
        bandname : `str`
            Name of filter band.

        Returns
        -------
        bandpass : `rubin_sim.phot_utils.bandpass.Bandpass`
            Telescope throughput curves.
        """
        filename = os.path.join(get_data_dir(), 
                                'throughputs/baseline/total_{0}.dat'.format(bandname.lower()))
        bandpass = photUtils.Bandpass()
        bandpass.read_throughput(filename)
        return bandpass
