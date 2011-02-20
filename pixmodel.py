from __future__ import print_function

import numpy
import _fimage

_tmap={'gauss':1,'exp':2,'dev':3}
_emap={0:'ok',2**0:'invalid model',2**1:'determinant <= 0',2**2:'invalid nsub'}

def model_image(model, dims, cen, covar, nsub=4, counts=1.0, order='f'):
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
    covar: sequence
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
        covar=[8,2,4] # [Irr,Irc,Icc]

        im=model_image('gauss',dims,cen,covar)

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
            exp(-r))
        dev
            exp(-7.67*(r**0.25 - 1) )

    Thus the actual moments of the image will be different from entered
    except for guassians.


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

    if len(covar) != 3:
        raise ValueError("covariance must be a sequence of length 3")

    # the fortan code expects zeros in all pixels
    imf = numpy.zeros(dims, order='f', dtype='f4')

    # note offsetting the dimensions for fortran indexing

    Irr,Irc,Icc=covar
    flag=_fimage.model_f4image(modelnum,imf,cen[0]+1,cen[1]+1,Irr,Irc,Icc,nsub)

    if flag != 0:
        flagstring = _emap.get(flag, 'unknown error')
        raise RuntimeError("Error creating image: %s" % flagstring)

    if counts is not None:
        imf *= (counts/imf.sum())

    if order.lower() == 'c':
        im = numpy.array(imf, order='c')
        return im

    return imf


def double_gauss(dims, cen, cenrat, covar1,covar2,
                 nsub=4, counts=1.0, all=False):
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

    covar1: sequence
        The covariance matrix of the first gaussian, [Irr,Irc,Icc]
        where r->row c->column
    covar2: sequence
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
        s2 = sqrt(det2/det1)

        image = (im1 + b*s2*im2)/(1+b*s2)

    """

    # for shorthand
    b = cenrat

    
    # we'll just use these
    det1 = covar1[0]*covar1[2] - covar1[1]**2
    det2 = covar2[0]*covar2[2] - covar2[1]**2
    
    if det1 == 0:
        raise ValueError("determinat of first covariance matrix is 0")
    if det2 == 0:
        raise ValueError("determinat of second covariance matrix is 0")
    s2 = numpy.sqrt( det2/det1 )

    im1 = model_image('gauss',dims,cen,covar1,nsub=nsub)
    im2 = model_image('gauss',dims,cen,covar2,nsub=nsub)

    im = im1 + b*s2*im2
    im /= (1+b*s2)

    return im

