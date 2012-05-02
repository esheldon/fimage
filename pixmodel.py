from __future__ import print_function

try:
    from scipy.fftpack import fftn
    have_scipy=True
except:
    have_scipy=False

import numpy
from numpy import ogrid, array, sqrt, exp, ceil, log2,  pi
from numpy.fft import fftshift

import _fimage

_tmap={'gauss':1,'exp':2,'dev':3}
_emap={0:'ok',2**0:'invalid model',2**1:'determinant <= 0',2**2:'invalid nsub'}

def model_image(model, dims, cen, cov, nsub=4, counts=1.0, order='f', dtype='f8'):
    """
    Create in image with the specified model using sub-pixel integration

    Parameters
    ----------
    model: string
        The model type: 'gauss', 'exp', 'dev'
    dims: sequence
        The dimensions of the image
    cen: sequence
        The center in [row,col]
    cov: sequence
        A three element sequence representing the covariance matrix
        [Irr,Irc,Icc].  Note this only corresponds exactly to the moments of
        the object for a gaussian model.

        For an simple bivariate gaussian, Irr and Icc are sigma1**2 sigma2**2,
        but using the full matrix allows for other angles of oriention.

    counts: number, optional
        The total counts in the image.  Default 1.0.  If None, the image is not
        normalized. 

    nsub: integer, optional
        The size of the sub-pixel grid used to integrate the model.  Default
        is 4.  Send 1 for no sub-pixel integration.

    order: string
        Send either 'c' for C order or 'f' for fortran order.  By
        default the result is fortran since the data is created
        in fortran.

    dtype: string or numpy dtype
        The data type, default 'f8'.  Can be 4-byte float or 8 byte float.

    Returns
    -------
    Image: 2-d array
        The returned image is a 2-d numpy array of 4 byte floats.  The image if
        in fortran-contiguous order by default; use order='c' to get it in 
        c order.

    Example
    -------
        dims=[41,41]
        cen=[20,20]
        cov=[8,2,4] # [Irr,Irc,Icc]

        im=model_image('gauss',dims,cen,cov)

    Notes
    ----- 

    The input Irr,Irc,Icc go into the covariance matrix in the following
    sense: 

        det=Irr*Icc-Irc**2
        w1=Irr/det
        w2=Icc/det
        w12=Irc/det
        r2 = row**2*w2 + col**2*w1 - 2.*row*col*w12
        r=sqrt(r2)

        gaussian
            exp(-0.5*r2)
        exp
            exp( -sqrt(r2*3) )
        dev
            exp(-7.67*(r**0.25 - 1) )


    Fortran
    -------
    The code for creating the images is written in fortran and is compiled
    during installation.

    Sub-pixel integration
    ---------------------

        Each pixel is divided into a 4x4 grid.  If the following represents a
        pixel
      
            -----------------
            |   |   |   |   |
            -----------------
            |   |   |   |   |
            -------cen-------
            |   |   |   |   |
            -----------------
            |   |   |   |   |
            -----------------
      
        The center of the upper left corner is (-3/8, 3/8).

        The value in the pixel is the sum of all sub-pixels.
 
    """
    modelnum = _tmap.get( model.lower() )
    if modelnum is None:
        raise ValueError("invalid model '%s' "
                         "model must be in: %s" % (model,str(_tmap.keys())))

    if len(cov) != 3:
        raise ValueError("covariance must be a sequence of length 3")

    dt = numpy.dtype(dtype)
    if dt.name == 'float32':
        dt='f4'
    elif dt.name == 'float64':
        dt='f8'
    else:
        raise ValueError("dtype must be 4-byte float or 8-byte float")

    # the fortan code expects zeros in all pixels
    imf = numpy.zeros(dims, order='f', dtype=dt)

    # note offsetting the dimensions for fortran indexing

    Irr,Irc,Icc=cov
    if dt == 'f4':
        flag=_fimage.model_f4image(modelnum,imf,cen[0]+1,cen[1]+1,Irr,Irc,Icc,nsub)
    else:
        flag=_fimage.model_f8image(modelnum,imf,cen[0]+1,cen[1]+1,Irr,Irc,Icc,nsub)

    if flag != 0:
        flagstring = _emap.get(flag, 'unknown error')
        raise RuntimeError("Error creating image: %s" % flagstring)

    if counts is not None:
        imf *= (counts/imf.sum())

    if order.lower() == 'c':
        im = numpy.array(imf, order='c')
        return im

    return imf


def double_gauss(dims, cen, cenrat, cov1,cov2,
                 nsub=4, counts=1.0, all=False, 
                 dtype='f8'):
    """
    Make a double gaussian image.

    Parameters
    ----------
    dims: sequence, length 2
        The dimensions of the image [nrow,ncol]
    cen: sequence, length 2
        The center of the image.

    cenrat: scalar
        The ratio of the gaussians at the origin, second to first.
            cenrat = guass2(origin)/gauss1(origin)

    cov1: sequence
        The covariance matrix of the first gaussian, [Irr,Irc,Icc]
        where r->row c->column
    cov2: sequence
        The covariane matrix of the second gaussian.

    nsub: integer, optional
        The size of the sub-pixel grid used to integrate the model.  Default
        is 4.  Send 1 for no sub-pixel integration.

    counts: scalar, optional
        The total counts in the image.  Default 1.0

    all: boolean, optional
        If True, return a tuple (image, image1, image2). Default False

    Returns
    -------
    image: 2-d array
        The image. If all=True, it will be a tuple of images
        (image, image1, image2)

    Examples
    --------
        im=double_gauss([41,41],[20,20], 0.05, [8, 2, 4], [32, 8, 16])

    Notes
    -----
    Formula is the following, for im1 and im2 normalized, and
    det = Irr*Icc-Irc
    
        b = cenrat

        image = (im1 + b*im2)/(1+b)

    """

    # for shorthand
    b = cenrat

    
    # we'll just use these
    det1 = cov1[0]*cov1[2] - cov1[1]**2
    det2 = cov2[0]*cov2[2] - cov2[1]**2
    
    if det1 == 0:
        raise ValueError("determinat of first covariance matrix is 0")
    if det2 == 0:
        raise ValueError("determinat of second covariance matrix is 0")

    im1 = model_image('gauss',dims,cen,cov1,nsub=nsub,dtype=dtype)
    im2 = model_image('gauss',dims,cen,cov2,nsub=nsub,dtype=dtype)

    im = im1 + b*im2
    im /= (1+b)

    return im


def ogrid_image(model, dims, cen, cov, counts=1.0, order='f', dtype='f8'):

    Irr,Irc,Icc = cov
    det = Irr*Icc - Irc**2
    if det == 0.0:
        raise RuntimeError("Determinant is zero")

    Wrr = Irr/det
    Wrc = Irc/det
    Wcc = Icc/det

    # ogrid is so useful
    row,col=ogrid[0:dims[0], 0:dims[1]]

    rm = numpy.array(row - cen[0], dtype=dtype)
    cm = numpy.array(col - cen[1], dtype=dtype)

    rr = rm**2*Wcc -2*rm*cm*Wrc + cm**2*Wrr

    model = model.lower()
    if model == 'gauss':
        rr = 0.5*rr
    elif model == 'exp':
        rr = sqrt(rr*3.)
    elif model == 'dev':
        rr = 7.67*( (rr)**(.125) -1 )
    else: 
        raise ValueError("model must be one of gauss, exp, or dev")

    image = exp(-rr)

    return image

def ogrid_turb_psf(dims, fwhm, counts=1.):
    """
    Create an image of a PSF produced by atmospheric turbulence.  
    
    The image is created in k space using ogrid_turb_kimage and then Fourier
    transformed.  The form in k space is exp(-0.5(k/k0)^(5/3))

    parameters
    ----------
    dims: [nrows,ncols]
        The dimensions of the result.  Must be square and even.
    fwhm:
        The fwhm for the result in real space.  Note
            k0 = 2.92/fwhm
    counts: optional
        Counts for the image, default 1
    """
    if dims[0] != dims[1]:
        raise ValueError("only square turbulence psf allowed")
    if (dims[0] % 2) != 0:
        raise ValueError("only even dimensions for turbulence psf")

    # Always use 2**n-sized FFT
    kdims = 2**ceil(log2(dims))
    # on a pixel
    kcen = kdims/2.

    k0 = 2.92/fwhm
    # now account for scaling in fft
    k0 *= kdims[0]/(2*pi)

    otfk = ogrid_turb_kimage(kdims, kcen, k0)

    im = fftn(otfk)[0:dims[0], 0:dims[1]]
    im = sqrt(im.real**2 + im.imag**2)
    im = fftshift(im)

    im *= counts/im.sum()
    return im

def ogrid_turb_kimage(dims, cen, k0):
    """
    Return a circular k-space turbulence limited PSF model.

    This is exp(-(k/k0)^5/3) = exp(-(k^2/k0^2)^(5./6)

    k0 = 2.92/fwhm

    parameters
    ----------
    dims: [nrows_k,ncols_k]
        The dimensions of the image. Should be square the side a power of 2 if
        you are going to use it in an fft.
    cen:
        The center in k space.
    k0: 
        k0 = 2.92/fwhm
    """

    row,col=ogrid[0:dims[0], 0:dims[1]]

    rm = array(row - cen[0], dtype='f8')
    cm = array(col - cen[1], dtype='f8')

    rr2 = rm**2 + cm**2
    rr2 /= k0**2

    arg = rr2**(5./6.)

    image = exp(-arg)
    return image


