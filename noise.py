"""
For unweighted

          sum(pix)
    -------------------   = S/N
    sqrt(npix*skysig**2)

    thus
        
        sum(pix)
    ----------------   = skysig
    sqrt(npix)*(S/N)

"""
from sys import stderr
from numpy import ogrid, array, sqrt, where, ogrid, zeros, arange
from numpy.random import randn

def add_noise_uw(im, s2n, check=False):
    """
    Add gaussian noise to an image.

          sum(pix)
    -------------------   = S/N
    sqrt(npix*skysig**2)

    thus
        
        sum(pix)
    ----------------   = skysig
    sqrt(npix)*(S/N)


    parameters
    ----------
    im: numpy array
        The image
    s2n:
        The requested S/N

    outputs
    -------
    image, skysig
        A tuple with the image and error per pixel.

    """

    skysig = im.sum()/sqrt(im.size)/s2n

    noise_image = skysig*randn(im.size).reshape(im.shape)
    image = im + noise_image

    if check:
        s2n_check = get_s2n_uw(image, skysig)
        print >>stderr,"S/N goal:",s2n,"found:",s2n_check

    return image, skysig

def add_noise_matched(im, s2n, cen, fluxfrac=None, check=False):
    """
    Add gaussian noise to an image assuming
    a matched filter is used.

     sum(pix^2)
    ------------ = S/N^2
      skysig^2

    thus
        
    sum(pix^2)
    ---------- = skysig^2
      (S/N)^2

    parameters
    ----------
    im: numpy array
        The image
    s2n:
        The requested S/N

    outputs
    -------
    image, skysig
        A tuple with the image and error per pixel.

    """

    if fluxfrac is not None:
        row,col=ogrid[0:im.shape[0], 
                      0:im.shape[1]]
        rm = array(row - cen[0], dtype='f8')
        cm = array(col - cen[1], dtype='f8')
        radm = sqrt(rm**2 + cm**2)

        radii = arange(1,im.shape[0]/2)
        cnts=zeros(radii.size)
        for ir,r in enumerate(radii):
            w=where(radm <= r)
            if w[0].size > 0:
                cnts[ir] = im[w].sum()

        cnts /= cnts.max()
        wr,=where(cnts > fluxfrac)
        if wr.size > 0:
            radmax = radii[wr.min()]
        else:
            radmax = radii[-1]


        w=where(radm <= radmax)
        #print 'radmax:',radmax,'image shape:',im.shape,"wsize:",w[0].size

        skysig2 = (im[w]**2).sum()/s2n**2
        skysig = sqrt(skysig2)

    else:

        skysig2 = (im**2).sum()/s2n**2
        skysig = sqrt(skysig2)

    noise_image = skysig*randn(im.size).reshape(im.shape)
    image = im + noise_image

    if check:
        s2n_check = get_s2n_matched(image, skysig)
        print >>stderr,"S/N goal:",s2n,"found:",s2n_check

    return image, skysig


def get_s2n_matched(im, skysig):
    """
    im should be sky subtracted
    """
    return sqrt( (im**2).sum()/skysig**2 )

def get_s2n_uw(im, skysig):
    """
    im should be sky subtracted
    """
    return im.sum()/sqrt(im.size)/skysig


def add_noise_dev(im, cen, re, s2n, fluxfrac=0.85):
    """
    Add noise to an image assumed to contain an r^1/4 surface
    brightness profile.

    parameters
    ----------
    im: numpy array
        The image
    cen: 2 element sequence
        Center position
    re: float
        Half life radius.
    s2n:
        The requested S/N
   
    fluxfrac: float, optional
        What fraction of the flux to use.  Default 0.85, which
        means count the flux within r85.

        Note 
            fluxfrac=0.5 => 0.5*re
            fluxfrac=0.85 => 4*re
            fluxfrac=0.9 => 5.5*re

    outputs
    -------
    image, skysig
        A tuple with the image and error per pixel.

    """
    from .statistics import interplin

    row,col=ogrid[0:im.shape[0], 0:im.shape[1]]
    rm = array(row - cen[0], dtype='f8')
    cm = array(col - cen[1], dtype='f8')
    radm = sqrt(rm**2 + cm**2)

    # find radius r/re where we expect fluxfrac
    r = interplin(_dev_r_over_re,
                  _dev_fluxfrac, 
                  fluxfrac)
    r = r*re

    w=where(radm <= r)
    while w[0].size == 0:
        r +=1 
        w=where(radm <= r)

    skysig = im[w].sum()/sqrt(w[0].size)/s2n

    noise_image = skysig*randn(im.size).reshape(im.shape)
    image = im + noise_image

    return image, skysig

def add_noise_admom(im, s2n, check=False):
    """
    Add noise to a convolved image based on requested S/N.  This
    will be the adaptive moments S/N so is gaussian weighted.  
    This is *not* appropriate for all profiles, and is in fact
    very poor for some.

    parameters
    ----------
    im: numpy array
        The image
    s2n:
        The requested S/N
    check: bool
        If True, print out the measured S/N
    outputs
    -------
    image, skysig
        A tuple with the image and error per pixel.

    """

    import admom
    from .statistics import fmom
    from .conversions import mom2sigma

    moms = fmom(im)

    cen = moms['cen']
    cov = moms['cov']
    T = cov[0] + cov[2]

    sigma = mom2sigma(T)
    shape = im.shape

    row,col=ogrid[0:shape[0], 0:shape[1]]
    rm = array(row - cen[0], dtype='f8')
    cm = array(col - cen[1], dtype='f8')

    radpix = sqrt(rm**2 + cm**2)
    # sigfac is 4 in admom
    sigfac = 4.0
    step=0.1
    w = where(radpix <= sigfac*sigma)
    npix = w[0].size + w[1].size
    while npix == 0:
        sigfac += step
        w = where(radpix <= sigfac*sigma)
        npix = w[0].size + w[1].size

    pix = im[w]
    wt = sqrt(pix)
    wsignal = (wt*pix).sum()
    wsum = wt.sum()

    skysig = wsignal/sqrt(wsum)/s2n

    noise_image = \
        skysig*randn(im.size).reshape(shape)
    image = im + noise_image

    out = admom.admom(image, cen[0], cen[1], guess=T/2., sigsky=skysig)

    # fix up skysig based on measurement
    skysig = out['s2n']/s2n*skysig
    noise_image = skysig*randn(im.size).reshape(shape)
    image = im + noise_image


    if check:
        out = admom.admom(image, cen[0], cen[1], guess=T/2., sigsky=skysig)
        print >>stderr,"    target S/N:            ",s2n
        print >>stderr,"    meas S/N after noise:  ",out['s2n']

    return image, skysig

def get_s2n_admom(image, cen, skysig):
    import admom
    Tguess=2.
    out = admom.admom(image, cen[0], cen[1], guess=Tguess, sigsky=skysig)
    return out['s2n']


_dev_r_over_re=\
    array([  0.33333333,   0.66666667,   1.        ,   1.33333333,
           1.66666667,   2.        ,   2.33333333,   2.66666667,
           3.        ,   3.33333333,   3.66666667,   4.        ,
           4.33333333,   4.66666667,   5.        ,   5.33333333,
           5.66666667,   6.        ,   6.33333333,   6.66666667,
           7.        ,   7.33333333,   7.66666667,   8.        ,
           8.33333333,   8.66666667,   9.        ,   9.33333333,
           9.66666667,  10.        ,  10.33333333,  10.66666667,
           11.        ,  11.33333333,  11.66666667,  12.        ,
           12.33333333,  12.66666667,  13.        ,  13.33333333,
           13.66666667,  14.        ,  14.33333333,  14.66666667,
           15.        ,  15.33333333,  15.66666667,  16.        ,
           16.33333333,  16.66666667])
_dev_fluxfrac=\
    array([ 0.25775525,  0.38473126,  0.51940452,  0.58706012,  0.64598991,
           0.69070953,  0.73257906,  0.76709472,  0.79043855,  0.81285342,
           0.83243716,  0.84700322,  0.86369613,  0.87470347,  0.88661544,
           0.89599153,  0.90416058,  0.91162952,  0.91775997,  0.92479378,
           0.93039566,  0.93522643,  0.93956254,  0.94346488,  0.94765705,
           0.95088503,  0.95396993,  0.95696255,  0.95949455,  0.96191822,
           0.96421254,  0.96635124,  0.9683028 ,  0.970024  ,  0.97164435,
           0.97308427,  0.97461923,  0.97600281,  0.97725996,  0.97835   ,
           0.97946764,  0.98049026,  0.98143899,  0.98237593,  0.98323327,
           0.9840437 ,  0.98475971,  0.9854314 ,  0.98614041,  0.98675285])
