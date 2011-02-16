from __future__ import print_function
import numpy

from . import _fconv

_emap={1:'determinant <= 0'}

def gaussconv(image, covar):
    imout=numpy.zeros(image.shape, dtype='f4', order='f')
    Irr,Irc,Icc=covar
    flag=_fconv.gaussconvf4image(image,Irr,Irc,Icc,imout)

    if flag != 0:
        flagstring = _emap.get(flag, 'unknown error')
        raise RuntimeError("Error convolving image: %s" % flagstring)

    return imout
