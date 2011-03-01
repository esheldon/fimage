from __future__ import print_function
import numpy
from numpy import arange, array, ceil, zeros, sqrt, log10

from . import _fmom

from .pixmodel import model_image

def moments(image, cen=None):
    """
    Measure the unweighted centroid and second moments of an image.

    Parameters
    ----------
    image: array
        A two-dimensional numpy array.

    cen: length two sequence, optional
        cen=(rowcen,colcen). If sent it is used for the
        center instead of a calculated centroid.

    Returns
    -------
    moments: dictionary
        {'cen':[rowcen,colcen],
         'cov':[Irr,Irc,Icc]}
    Example:
        mom = second_moments(image)
        mom = second_moments(image,cen)

        print("center:",mom['cen'])
        print("covariance matrix:",mom['cov'])

    """

    # ogrid is so useful
    row,col=numpy.ogrid[0:image.shape[0], 0:image.shape[1]]

    Isum = image.sum()

    if cen is None:
        rowcen = (image*row).sum()/Isum
        colcen = (image*col).sum()/Isum
        cen = (rowcen,colcen)
    else:
        if len(cen) != 2:
            raise ValueError("cen must be length 2 (row,col)")

    rm = row - cen[0]
    cm = col - cen[1]

    Irr = (image*rm**2).sum()/Isum
    Irc = (image*rm*cm).sum()/Isum
    Icc = (image*cm**2).sum()/Isum

    T=Irr+Icc
    e1=(Icc-Irr)/T
    e2=2.*Irc/T

    cen = numpy.array(cen,dtype='f8')
    cov = numpy.array([Irr,Irc,Icc], dtype='f8')
    return {'cen': cen, 'cov':cov, 'e1':e1, 'e2':e2}

def second_moments(image, cen=None):
    """
    Measure the second moments of an image.

    This just calls moments() and returns the covariance matrix elements in a
    array [Irr,Irc,Icc].  See docs for fimage.stat.moments for more details.

    """
    t=moments(image,cen)
    return t['cov']


def fmom(image, nsub=1):
    cen=numpy.zeros(2,dtype='f8')
    cov=numpy.zeros(3,dtype='f8')

    nsub=int(nsub)
    if nsub < 1:
        raise ValueError("nsub must be >= 1")
    if image.dtype.char == 'f' and image.dtype.itemsize==8:
        isdouble=True
    else:
        isdouble=False

    if nsub == 1:
        if isdouble:
            _fmom.mom_f8(image,cen,cov)
        else:
            _fmom.mom_f4(image,cen,cov)
    else:
        # note we only have an f4 version of this, so
        # f8 will be converted
        _fmom.mom_bilin_f4(image,nsub,cen,cov)

    cen -= 1

    T=cov[0] + cov[2]
    e1 = (cov[2]-cov[0])/T
    e2 = 2*cov[1]/T

    return {'cen':cen, 'cov':cov, 'e1':e1, 'e2':e2}
