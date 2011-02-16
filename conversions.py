import numpy
from numpy import log, sqrt, cos, sin, pi as PI

_fwhm_fac = 2*sqrt(2*log(2))

def cov2det(cov):
    return cov[0]*cov[2] - cov[1]**2

def fwhm2sigma(fwhm):
    return fwhm/_fwhm_fac
def sigma2fwhm(sigma):
    return sigma*_fwhm_fac

def mom2fwhm(T, pixscale=1.0):
    sigma = mom2sigma(T)
    return pixscale*sigma2fwhm(sigma)

def fwhm2mom(fwhm, pixscale=1.0):
    sigma = fwhm2sigma(fwhm)/pixscale
    return sigma2mom(sigma)


def mom2sigma(T):
    is_scalar = numpy.isscalar(T)
    T = numpy.array(T, ndmin=1)
    sigma = numpy.empty(T.size, dtype='f4')

    sigma = numpy.where(T > 0, sqrt(T/2), -9999.0)

    if is_scalar:
        sigma = sigma[0]

    return sigma

def sigma2mom(sigma):
    T = 2*sigma**2
    return T

def mom2ellip(Irr, Irc, Icc):
    T = Irr+Icc
    e1=(Icc-Irr)/T
    e2=2.0*Irc/T

    return e1,e2,T

def ellip2mom(T, e1=None, e2=None, e=None, theta=None):
    if e is not None and theta is not None:
        e1 = e*cos(2*theta*PI/180.0)
        e2 = e*sin(2*theta*PI/180.0)

    if e1 is None or e2 is None:
        raise ValueError("send e1/e2 or e/theta")

    Irc = e2*T/2.0
    Icc = (1+e1)*T/2.0
    Irr = (1-e1)*T/2.0

    return Irr, Irc, Icc

def momdet(Irr, Irc, Icc):
    return Irr*Icc-Irc**2
