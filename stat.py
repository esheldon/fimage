import numpy

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

    cen = numpy.array(cen,dtype='f8')
    cov = numpy.array([Irr,Irc,Icc], dtype='f8')
    return {'cen': cen, 'cov':cov}

def second_moments(image, cen=None):
    """
    Measure the second moments of an image.

    This just calls moments() and returns the covariance matrix elements in a
    array [Irr,Irc,Icc].  See docs for fimage.stat.moments for more details.

    """
    t=moments(image,cen)
    return t['cov']
