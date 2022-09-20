"""Object profiles contains commonly used profiles of different types of source
objects such as PointSource, Disk etc...
"""
import os
import warnings

import numpy as np
from scipy import stats
import cv2

from leosim.profiles.convolutionobj import ConvolutionObject
from leosim.profiles.consts import RAD2ARCSEC

class PointSource(ConvolutionObject):
    """Simple point like source. Point-like sources are not resolved, therefore
    regardless of scale and size only a single element of obj will have a value
    Parameters
    ----------
    h : float
      height of the object, in meters
    res : float
      desired resolution in arcseconds. The scale step (the difference between
      two neighbouring "x-axis" points) is then determined by scaling the
      appropriate angular size of the object by the resolution, so that the
      object remains unresolved at the required resolution.
    """
    def __init__(self, h, res=0.001, **kwargs):
        self.h = h
        theta = 1.0/(h*1000.) * RAD2ARCSEC
        scale = np.arange(-theta, theta, theta*res)

        obj = self.f(scale)
        ConvolutionObject.__init__(self, obj, scale)

    def f(self, r):
        """Returns the intensity value of the object at a point. Evaluating a
        point source over only handfull of points is not well defined. The
        function may not behave properly if number of points is very small,
        i.e. 2 or 3 points only.
        """
        if hasattr(r, "__iter__"):
            # if PointSource is evaluated over an array with finer grid than
            # scale and we calculate the values we would get max value for all
            # points closer than current scale step - this wouldn't be a point
            # so instead a new array of same shape is returned with only 1
            # element at max - a point source. We make a check if r is ndarray
            # or just a list or tuple
            try:
                obj = np.zeros(r.shape)
            except AttributeError:
                obj = np.zeros(len(r))
            obj[int(len(obj)/2)] += 1.0
            return obj
        else:
            # in the case where we evaluate at a singular point
            # if the position is closer to the location of the peak than the
            # resolution of our scale - we are looking at the source, otherwise
            # we are looking somewhere else
            peak = np.where(self.obj == self.obj.max())[0][0]
            if (r - self.scale[peak]) <= self.step:
                # standardize the output format to numpy array even in case of
                # a single number
                return np.array([self.obj.max()])
            else:
                return np.array([0.0])

class GaussianSource(ConvolutionObject):
    """Simple gaussian intensity profile.
    Parameters
    ----------
    h : float
      height of the object, in meters
    fwhm : float
      FWHM of the gaussian profile
    res : float
      desired resolution, in arcseconds
    units : string
      spatial units (meters by default) - currently not very well supported
    """
    def __init__(self, h, fwhm, res=0.001, units="meters", **kwargs):
        self.h = h
        self.theta = fwhm/(h*1000.)*RAD2ARCSEC
        self.sigma = fwhm/2.355
        self._f = stats.norm(scale=fwhm/2.355).pdf

        scale = np.arange(-1.7*self.theta, 1.7*self.theta, self.theta*res)
        obj = self.f(scale)
        ConvolutionObject.__init__(self, obj, scale)

    def f(self, r):
        """Evaluate the gaussian at a point."""
        # standardize the output format to numpy array even in case of a number
        if any((isinstance(r, int), isinstance(r, float),
               isinstance(r, complex))):
            rr = np.array([r], dtype=float)
        else:
            rr = np.array(r, dtype=float)

        return self._f(rr)

class DiskSource(ConvolutionObject):
    """Brightness profile of a disk-like source.
    Parameters
    ----------
    h : float
      height of the object, in meters
    radius : float
      radius of the objects disk, in meters
    res : float
      desired resolution, in arcseconds
    """
    def __init__(self, h, radius, res=0.001, **kwargs):
        self.h = h
        self.r = radius
        self.theta = radius/(h*1000.) * RAD2ARCSEC

        scale = np.arange(-2*self.theta, 2*self.theta, self.theta*res)
        obj = self.f(scale)

        ConvolutionObject.__init__(self, obj, scale)

    def f(self, r, units="arcsec"):
        """Returns the brightness value of the object at a point. By default
        the units of the scale are arcseconds but radians are also accepted.
        """
        # 99% (578x) speedup was achieved refactoring this code - Jan 2018

        # standardize the output format to numpy array even in case of a number
        if any((isinstance(r, int), isinstance(r, float),
               isinstance(r, complex))):
            rr = np.array([r], dtype=float)
        else:
            rr = np.array(r, dtype=float)

        if units.upper() == "RAD":
            rr = rr*RAD2ARCSEC

        theta = self.theta

        def _f(x):
            return 2*np.sqrt(theta**2-x**2)/(np.pi*theta**2)

        # testing showed faster abs performs faster for smaller arrays and
        # logical_or outperforms abs by 12% for larger ones
        if len(rr) < 50000:
            mask = np.abs(rr) >= theta
        else:
            mask = np.logical_or(rr > theta, rr < -theta)

        rr[mask] = 0
        rr[~mask] = _f(rr[~mask])

        return rr

    def width(self):
        """FWHM is not a good metric for measuring the end size of the object
        because disk-like profiles do not taper off towards the top. Instead
        width of the object (difference between first and last point with
        brightness above zero) is a more appropriate measure of the size of the
        object.
        """
        left = self.scale[np.where(self.obj > 0)[0][0]]
        right = self.scale[np.where(self.obj > 0)[0][-1]]
        return right-left

class RectangularSource(ConvolutionObject):
    """Brightness profile of a rectangular-like source.
    Parameters
    ----------
    h : float
      height of the object, in meters
    length : float
      length of the object, in meters
    width : float
      width of the object, in meters
    res : float
      desired resolution, in arcseconds"""
    def __init__(self, h, length, width, res=0.001, **kwargs):
        self.h = h
        self.length = length
        self.width = width
        self.theta = width/(2*h*1000.) * RAD2ARCSEC
        
        scale = np.arange(-2*self.theta, 2*self.theta, self.theta*res)
        obj = self.f(scale)
        
        ConvolutionObject.__init__(self, obj, scale)
        
    def f(self, r, units="arcsec"):
        """Returns the brightness value of the object at a point. By default
        the units of the scale are arcseconds but radians are also accepted.
        """

        if any((isinstance(r, int), isinstance(r, float),
               isinstance(r, complex))):
            rr = np.array([r], dtype=float)
        else:
            rr = np.array(r, dtype=float)

        if units.upper() == "RAD":
            rr = rr*RAD2ARCSEC

        theta = self.theta
        
        def _f(x):
            return 1/(2*theta)

        # testing showed faster abs performs faster for smaller arrays and
        # logical_or outperforms abs by 12% for larger ones
        if len(rr) < 50000:
            mask = np.abs(rr) >= theta
        else:
            mask = np.logical_or(rr > theta, rr < -theta)
            
        rr[mask] = 0
        rr[~mask] = _f(rr[~mask])

        return rr        

    def width(self):
        """FWHM is not a good metric for measuring the end size of the object
        because disk-like profiles do not taper off towards the top. Instead
        width of the object (difference between first and last point with
        brightness above zero) is a more appropriate measure of the size of the
        object.
        """
        left = self.scale[np.where(self.obj > 0)[0][0]]
        right = self.scale[np.where(self.obj > 0)[0][-1]]
        return right-left
