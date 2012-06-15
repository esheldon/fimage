from sys import stderr
from numpy import ogrid, array, sqrt, where
from numpy.random import randn


def add_noise(im, s2n, check=False):
    """
    Add noise to a convolved image based on requested S/N.  We only add
    background noise so the S/N is

          sum(pix)
    -------------------   = S/N
    sqrt(npix*skysig**2)

    thus
        
        sum(pix)
    ----------------   = skysig
    sqrt(npix)*(S/N)

    
    We use an aperture of 3sigma but if no pixels
    are returned we increase the aperture until some
    are returned.

    Side effects:
        ci.image is set to the noisy image
        ci['skysig'] is set to the noise level
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


