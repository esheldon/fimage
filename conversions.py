from __future__ import print_function
import numpy
from numpy import log, sqrt, cos, sin, pi as PI, exp, linspace, \
        where, array
from sys import stderr

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


def cov2sigma(cov):
    T = cov[0] + cov[2]
    return mom2sigma(T)

def mom2sigma(T):
    is_scalar = numpy.isscalar(T)
    T = array(T, ndmin=1)
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

    return array([e1,e2,T])

def ellip2mom(T, e1=None, e2=None, e=None, theta=None, units='deg'):
    if e is not None and theta is not None:
        if units == 'deg':
            e1 = e*cos(2*theta*PI/180.0)
            e2 = e*sin(2*theta*PI/180.0)
        else:
            e1 = e*cos(2*theta)
            e2 = e*sin(2*theta)

    if e1 is None or e2 is None:
        raise ValueError("send e1/e2 or e/theta")

    Irc = e2*T/2.0
    Icc = (1+e1)*T/2.0
    Irr = (1-e1)*T/2.0

    return array([Irr, Irc, Icc])

def etheta2e1e2(e, theta, units='deg'):
    if units == 'deg':
        e1 = e*cos(2*theta*PI/180.0)
        e2 = e*sin(2*theta*PI/180.0)
    else:
        e1 = e*cos(2*theta)
        e2 = e*sin(2*theta)

    return e1,e2

def momdet(Irr, Irc, Icc):
    return Irr*Icc-Irc**2

def calculate_nsigma(type,show=False):
    from scipy.special import erf
    import pcolors
    x = linspace(0,20,10000)

    if type == 'gauss':
        # just to make sure what we're doing makes sense here
        y = exp(-0.5*x**2)
        xlabel = r'$x/\sigma$'
    elif type == 'exp':
        y = exp(-x)
        xlabel = r'$x/x_0$'
    else:
        raise ValueError("type should be gauss,exp")

    ysum = y.cumsum()
    norm = y.sum()
    ysum /= norm

    xval = []
    ival = []
    percval = []

    for nsig in [1,2,3,4,4.5]:
        perc = erf(nsig/sqrt(2))
        w,=where(ysum <= perc)
        print("nsig:",nsig,"xval:",x[w[-1]],file=stderr)

        xval.append(x[w[-1]])
        ival.append(w[-1])
        percval.append(perc)


    if show:
        import biggles
        c = biggles.Curve(x,ysum)
        plt=biggles.FramedPlot()
        plt.add(c)

        colors=pcolors.rainbow(len(ival), 'hex')
        j=0
        ps=[]
        for i in ival:
            p = biggles.Points([x[i]], [ysum[i]], type='filled circle', 
                               color=colors[j])
            p.label = '%f' % (100.0*percval[j],)
            plt.add(p)
            ps.append(p)
            j+=1

        key = biggles.PlotKey(0.95,0.3,ps,halign='right')
        plt.add(key)
        plt.xlabel = xlabel
        plt.ylabel = 'cumulative PDF'

        plt.xrange=[0.0, 1.1*xval[-1] ]

        plt.show()

