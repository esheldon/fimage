from __future__ import print_function
import numpy

from . import _fconv

_gc_errmap={1:'determinant <= 0'}

def gaussconv(image, covar):
    imout=numpy.zeros(image.shape, dtype='f4', order='f')
    Irr,Irc,Icc=covar
    flag=_fconv.gaussconv_f4(image,Irr,Irc,Icc,imout)

    if flag != 0:
        flagstring = _gc_errmap.get(flag, 'unknown error')
        raise RuntimeError("Error convolving image: %s" % flagstring)

    return imout

_ci_errmap={1:'npix must be odd for second image in both dimensions'}
def convolve_images(image1, image2):
    imout=numpy.zeros(image1.shape, dtype='f4', order='f')
    flag=_fconv.conv_images_f4(image1,image2,imout)

    if flag != 0:
        flagstring = _ci_errmap.get(flag, 'unknown error')
        raise RuntimeError("Error convolving image: %s" % flagstring)

    return imout

