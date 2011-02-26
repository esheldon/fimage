from __future__ import print_function
import numpy

from . import _fconv
#from . import _testconv

_gc_errmap={1:'determinant <= 0'}

def gaussconv(image, cov):
    imout=numpy.zeros(image.shape, dtype='f4', order='f')
    Irr,Irc,Icc=cov
    flag=_fconv.gaussconv_f4(image,Irr,Irc,Icc,imout)

    if flag != 0:
        flagstring = _gc_errmap.get(flag, 'unknown error')
        raise RuntimeError("Error convolving image: %s" % flagstring)

    return imout


_expgauss_errmap={1:'determinant of exp is <= 0',
                  2:'determinant of exp is <= 0'}

def conv_exp_gauss(dims, expcen, 
                   exp_cov, gauss_cov,
                   nsub=4, type='f'):
    imout=numpy.zeros(dims, dtype='f4', order='f')
    expIxx,expIxy,expIyy = exp_cov
    gaussIxx,gaussIxy,gaussIyy = gauss_cov

    if type == 'f':
        flag=_fconv.conv_exp_gauss_f4(expIxx,expIxy,expIyy,expcen[0]+1,expcen[1]+1,
                                      gaussIxx,gaussIxy,gaussIyy,nsub,imout)
    else:
        flag=_testconv.conv_exp_gauss_f4(expIxx,expIxy,expIyy,expcen[0]+1,expcen[1]+1,
                                         gaussIxx,gaussIxy,gaussIyy,nsub,imout)

    #                             expixx,expixy,expiyy,expxcen,expycen,
    #                             gixx,gixy,giyy,nsub,image,[nx,ny])


    if flag != 0:
        flagstring = _expgauss_errmap.get(flag, 'unknown error')
        raise RuntimeError("Error convolving exp and gauss: %s" % flagstring)

    imout /= imout.sum()

    return imout

_ci_errmap={1:'npix must be odd for second image in both dimensions'}
def convolve_images(image1, image2):
    imout=numpy.zeros(image1.shape, dtype='f4', order='f')
    flag=_fconv.conv_images_f4(image1,image2,imout)

    if flag != 0:
        flagstring = _ci_errmap.get(flag, 'unknown error')
        raise RuntimeError("Error convolving image: %s" % flagstring)


    return imout

