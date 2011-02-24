import numpy
from . import _frebin

# errors from fortran code, these should be caught in the python wrapper
_fr_emap={2**0:'rebin factor must be same in each dimension',
          2**1:'rebin factor must be >= 1',
          2**2:'rimage dims must be multiple of image sims'}
def rebin(image, fac, order='f'):
    fac = int(fac)

    if len(image.shape) != 2:
        raise ValueError("image must be 2-dimensional")
    
    rowsize, colsize = image.shape
    if (rowsize % fac) != 0 or (colsize % fac) != 0:
        raise ValueError("dimensions must be divisible by the rebin factor")

    rimage = numpy.zeros((rowsize/fac, colsize/fac), dtype='f4', order='f')

    flag = _frebin.rebin_f4image(image, rimage)
    if flag != 0:
        flagstring = _fr_emap.get(flag, 'unknown error')
        raise RuntimeError("Error creating image: %s" % flagstring)

    if order.lower() == 'c':
        rimage = numpy.array(rimage, order='c')

    return rimage
