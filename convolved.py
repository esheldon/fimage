from __future__ import print_function
from pprint import pprint
import numpy
from numpy import sqrt, exp, array
import images

from . import stat
from . import pixmodel
from . import fconv
from . import analytic
from . import conversions

import admom

try:
    import scipy.signal
    have_scipy=True
except:
    have_scipy=False




class ConvolvedImage(dict):
    """
    Simulate an object convolved with a psf.  

    DOC THE INPUTS
    
    For exponential disks, a the image is created to be 2*7*sigma
    wide, where sigma is determined from the expected size after
    convolution.  For gaussians, a this is 2*4*sigma

    """

    def __init__(self, objpars, psfpars, 
                 conv='fconv',           # 'fconv','fft', or 'func'
                 eps=1.e-4,              # for FuncConvolver
                 forcegauss=False,       # force numerical convolutio for psf and obj gauss
                 verbose=False):

        if conv not in ['fconv','fft','func']:
            raise ValueError("conv must be 'fconv','fft','func'")
        self.conv=conv
        self['eps'] = eps
        self.verbose=verbose
        self.forcegauss=forcegauss
        self.objpars = objpars
        self.psfpars = psfpars

        if 'counts' not in self.objpars:
            self.objpars['counts'] = 1.0

        self.make_psf()
        self.make_image0()
        self.make_image()


    def make_psf(self):

        psfmodel = self.psfpars['model']
        if psfmodel == 'gauss':
            self.make_gauss_psf()
        elif psfmodel == 'dgauss':
            self.make_dgauss_psf()
        else:
            raise ValueError("unknown model type: '%s'" % psfmodel)

        cen = self.psfpars['cen']
        tcovar = stat.second_moments(self.psf, cen)
        res = admom.admom(self.psf, cen[0],cen[1], guess=(tcovar[0]+tcovar[2])/2)
        covar_meas = array([res['Irr'],res['Irc'],res['Icc']])
        self.psfpars['covar_meas'] = covar_meas
        self['covar_psf'] = covar_meas

        if self.verbose:
            print("PSF model")
            pprint(self.psfpars)


    def make_gauss_psf(self):
        pars=self.psfpars
        covar = pars['covar']

        T = 2*max(covar)
        dims,cen = self.getdimcen(T)

        self.psf = pixmodel.model_image('gauss',dims,cen,covar)

        pars['dims'] = dims
        pars['cen'] = cen

        if self.conv == 'func':
            self.psf_func=analytic.Gauss(covar)
       
    def make_dgauss_psf(self):

        pars=self.psfpars
        cenrat = pars['cenrat']
        covar1 = pars['covar1']
        covar2 = pars['covar2']

        Tmax = 2*max( max(covar1), max(covar2) )

        dims=pars.get('dims')
        cen=pars.get('cen')
        if dims is not None or cen is not None:
            if dims is None or cen is None:
                raise ValueError("If you send psf cen or dims, then send BOTH")
        else:
            dims,cen = self.getdimcen(Tmax)

        self.psf = pixmodel.double_gauss(dims,cen,cenrat,covar1,covar2)
        det1=conversions.cov2det(covar1)
        det2=conversions.cov2det(covar2)
        pars['s2'] = sqrt(det2/det1)

        pars['dims'] = dims
        pars['cen'] = cen

        if self.conv == 'func':
            self.psf_func=analytic.DoubleGauss(cenrat,covar1,covar2)


    def make_image0(self):
        # run make_psf first!
        #
        # unconvolved model
        # even though it is unconvolved, make it as large as we 
        # would expect if it were convolved with the PSF
        # also make sure the psf image size is still smaller

        pars = self.objpars
        psfpars = self.psfpars

        objmodel = pars['model']
        covar = pars['covar']

        if objmodel == 'gauss':
            sigfac = 4.5
        elif objmodel == 'exp':
            #sigfac = 5.0
            sigfac = 7.0
        else:
            raise ValueError("Unsupported obj model: '%s'" % objmodel)

        # expected size
        Irr,Irc,Icc = covar

        Irr_exp = Irr + psfpars['covar_meas'][0]
        Icc_exp = Icc + psfpars['covar_meas'][2]

        Texp = 2*max(Irr_exp,Icc_exp)
        dims, cen = self.getdimcen(Texp, sigfac=sigfac)

        self.image0 = pixmodel.model_image(objmodel,dims,cen,covar,
                                         counts=pars['counts'])

        pars['dims'] = dims
        pars['cen'] = cen
        tcovar = stat.second_moments(self.image0, cen)
        res = admom.admom(self.image0, cen[0],cen[1], guess=(tcovar[0]+tcovar[1])/2 )
        covar_meas = array([res['Irr'],res['Irc'],res['Icc']])
        pars['covar_meas'] = covar_meas
        self['covar_image0'] = covar_meas

        if self.conv == 'func':
            # turns out we can use very low precision..
            if objmodel == 'exp':
                self.obj_func=analytic.Exp(covar)
            elif objmodel == 'gauss':
                self.obj_func=analytic.Gauss(covar)
            else:
                raise ValueError("only support objmodel gauss or exp for "
                                 "slow convolution")

        if self.verbose:
            print("image0 pars")
            pprint(self.objpars)


    def make_image(self):

        pars=self.objpars
        psfpars=self.psfpars

        if self.verbose:
            print("Convolving to final image")

        dims = pars['dims']
        cen = pars['cen']
        image0 = self.image0
        psf = self.psf

        # we don't have to do a numerical convolution if both are gaussians
        bothgauss=False
        psfmodel = psfpars['model']
        objmodel = pars['model']
        if (psfmodel == 'gauss' and objmodel == 'gauss') and not self.forcegauss:
            bothgauss=True
            ocovar=pars['covar']
            pcovar=psfpars['covar']
            covar = [ocovar[0]+pcovar[0],
                     ocovar[1]+pcovar[1],
                     ocovar[2]+pcovar[2]]

            image = pixmodel.model_image('gauss',dims,cen,covar,
                                       counts=pars['counts'])
        else:
            if not have_scipy:
                raise ImportError("Could not import scipy")


            if self.conv == 'fft':
                # this should be un-necessary
                image0_expand = images.expand(self.image0, self.psf.shape, verbose=self.verbose)
                image = scipy.signal.fftconvolve(image0_expand, self.psf, mode='same')

                if image.shape[0] > dims[0] or image.shape[1] > dims[1]:
                    if self.verbose:
                        print("  Trimming back to requested size")
                    image = image[ 0:dims[0], 0:dims[1] ]
            elif self.conv == 'fconv':
                print("running fconv")
                if psfmodel == 'gauss':
                    image = fconv.gaussconv(self.image0, psfpars['covar'])
                elif psfmodel == 'dgauss':
                    s2 = psfpars['s2']
                    b = psfpars['cenrat']
                    im1 = fconv.gaussconv(self.image0, psfpars['covar1'])
                    im2 = fconv.gaussconv(self.image0, psfpars['covar2'])

                    image = im1 + b*s2*im2
                    image /= (1+b*s2)

            elif self.conv == 'func': 
                psf_func=self.psf_func
                obj_func=self.obj_func
                orange = obj_func.range()
                prange = psf_func.range()
                #rng = min(orange[1],prange[1])
                rng = 2*max(orange[1],prange[1])
                intrange = (-rng,rng)
                tdim = int( numpy.ceil( 2*sqrt( orange[1]**2 + prange[1]**2 ) ) )
                print("intrange:",intrange,"needed dim:",tdim)
                c = FuncConvolver(obj_func, psf_func, intrange, epsrel=self['eps'],epsabs=self['eps'])
                image = c.make_image(dims, cen)
                image *= (pars['counts']/image.sum())

        self.image = image
        self['dims'] = dims
        self['cen'] = cen
        if bothgauss:
            covar_meas = array(pars['covar']) + array(psfpars['covar'])
        else:
            tcovar = stat.second_moments(self.image, cen)
            res = admom.admom(self.image, cen[0],cen[1], guess=(tcovar[0]+tcovar[2])/2)
            covar_meas = array([res['Irr'],res['Irc'],res['Icc']])

        self['covar'] = covar_meas

        if self.verbose:
            print("convolved image pars")
            pprint(self)

    def mom2sigma(self, T):
        return sqrt(T/2)
    def getdimcen(self, T, sigfac=4.5):
        sigma = sqrt(T/2)
        imsize = int( numpy.ceil(2*sigfac*sigma) )

        # MUST BE ODD!
        if (imsize % 2) == 0:
            imsize+=1
        dims = [imsize]*2
        cen = [(imsize-1)/2]*2

        return dims,cen

 
    def show(self):
        import biggles

        levels=7
        tab = biggles.Table(2,2)
        tab[0,0] = images.view(self.image0, show=False, levels=levels, min=0)
        tab[0,1] = images.view(self.psf, show=False, levels=levels, min=0)
        tab[1,0] = images.view(self.image, show=False, levels=levels, min=0)

        # cross-section across rows
        cen = self['cen']
        imrows = self.image[:, cen[1]]
        imcols = self.image[cen[0], :]

        crossplt = biggles.FramedPlot()
        hrows = biggles.Histogram(imrows, color='blue')
        hrows.label = 'Center rows'
        hcols = biggles.Histogram(imcols, color='red')
        hcols.label = 'Center columns'

        key = biggles.PlotKey(0.1, 0.9, [hrows, hcols])

        crossplt.add(hrows, hcols, key)
        crossplt.aspect_ratio=1
        yr = crossplt._limits1().yrange()
        yrange = (yr[0], yr[1]*1.2)
        crossplt.yrange = yrange

        tab[1,1] = crossplt

        tab.show()



def test_dgauss_conv(conv='fconv'):
    # test at different resolutions to make sure
    # the recovered ellipticity is as expected
    # {'b': 0.079171792, 'sigma1': 1.2308958, 'sigma2': 2.8572364}
    import copy
    import biggles

    # typical SDSS size, resolution
    psfpars1={'model':'dgauss',
              'cenrat':0.08,
              'covar1':array([1.23**2, 0.0, 1.23**2]),
              'covar2':array([2.9**2, 0.0, 2.9**2])}

    objpars1={'model':'gauss',
              'covar':array([1.5**2, 0.0, 0.5**2])}

    n=20
    sizefac_vals = numpy.linspace(1.1,20.0)

    e1=numpy.zeros(n,dtype='f4')
    e2=numpy.zeros(n,dtype='f4')
    s=numpy.zeros(n,dtype='f4')

    for i in xrange(n):
        sizefac = sizefac_vals[i]
        sizefac2 = sizefac**2

        print("sizefac:",sizefac)

        psfpars = copy.deepcopy( psfpars1 )

        psfpars['covar1'] *= sizefac2
        psfpars['covar2'] *= sizefac2

        objpars = copy.deepcopy( objpars1 )

        objpars['covar'] *= sizefac2

        ci = ConvolvedImage(objpars,psfpars, conv=conv)

        T = ci['covar'][0] + ci['covar'][2]

        e1t = (ci['covar'][2]-ci['covar'][0])/T
        e2t =  2*ci['covar'][1]/T

        e1[i] = e1t
        e2[i] = e2t
        det = conversions.cov2det(ci['covar'])
        s[i] = det**0.25
        print("s:",s[i])


    etot = sqrt(e1**2 + e2**2) 

    ebest = etot[-1]

    pdiff = etot/ebest - 1

    plt=biggles.FramedPlot()
    c = biggles.Curve(s, pdiff)
    plt.add(c)
    plt.xlabel = 'size (pixels)'
    plt.ylable = '(e-ebest)/ebest'
    plt.title = 'conv method: '+conv

    plt.show()




class FuncConvolver:
    def __init__(self, func1, func2, range,
                 epsabs=1.4899999999999999e-08,
                 epsrel=1.4899999999999999e-08,
                 verbose=False):
        self.func1=func1
        self.func2=func2
        self.range=range

        self.epsabs=epsabs
        self.epsrel=epsrel

        self.verbose=verbose

    def _intfunc(self, row, col):
        return self.func1(row,col)*self.func2(row-self.row0, col-self.col0)

    def _intfunc2(self, col, a, b):
        res=scipy.integrate.quad(self._intfunc, a, b, args=(col,), 
                                 epsabs=self.epsabs, epsrel=self.epsrel)
        return res[0]

    def _conv1(self, row0, col0):
        self.row0=row0
        self.col0=col0
        ret = scipy.integrate.quad(self._intfunc2, self.range[0], self.range[1], args=self.range,
                                  epsabs=self.epsabs,epsrel=self.epsrel)
        return ret[0]

    def make_image(self, dims, cen):
        im=numpy.zeros(dims,dtype='f4')

        self.cen = cen

        for irow in xrange(dims[0]):
            row=irow-cen[0]
            for icol in xrange(dims[1]):
                col=icol-cen[1]
                if self.verbose:
                    print("(%d,%d)" % (row,col))
                im[irow,icol] = self._conv1(row,col)
        return im

    def make_image_fast(self, dims, cen, all=False):
        import scipy.signal

        im1=numpy.zeros(dims,dtype='f4')
        im2=numpy.zeros(dims,dtype='f4')

        for irow in xrange(dims[0]):
            row=irow-cen[0]
            for icol in xrange(dims[1]):
                col=icol-cen[1]

                im1[irow,icol] = self.func1(row,col)
                im2[irow,icol] = self.func2(row,col)
        

        im=scipy.signal.fftconvolve(im1,im2,mode='same')
        if all:
            return im, im1, im2
        return im

def test_func_convolver_gauss(epsabs=1.4899999999999999e-08, epsrel=1.4899999999999999e-08):
    from . import analytic

    Irr1=2.0
    Irc1=0.5
    Icc1=1.0

    Irr2=1.0
    Irc2=0.0
    Icc2=0.5
    g1 = analytic.Gauss(2.0, 0.5, 1.0)
    g2 = analytic.Gauss(1.0, 0.0, 0.5)

    # these are symmetric
    r1 = g1.range()
    r2 = g2.range()

    # take smaller of the two for integration range
    #rng = min(r1[1],r2[1])
    rng = max(r1[1],r2[1])
    intrange = (-rng,rng)


    dim = int( numpy.ceil( 2*sqrt( r1[1]**2 + r2[1]**2 ) ) )
    # make odd so the fast convolution will work
    if (dim % 2) == 0:
        dim += 1
    dims = [dim,dim]
    cen = [(dim-1)/2]*2


    print("range1:",r1)
    print("range2:",r2)
    print("intrange:",intrange)
    print("dims:",dims)
    print("cen:",cen)
    print("Irr expected:",Irr1+Irr2)
    print("Irc expected:",Irc1+Irc2)
    print("Icc expected:",Icc1+Icc2)


    c = FuncConvolver(g1,g2, intrange, epsabs=epsabs,epsrel=epsrel)

    imfast = c.make_image_fast(dims, cen)
    Irrfast,Ircfast,Iccfast = stat.second_moments(imfast,cen)
    print("Irr fast:",Irrfast)
    print("Irc fast:",Ircfast)
    print("Icc fast:",Iccfast)

    im = c.make_image(dims, cen)

    Irracc,Ircacc,Iccacc = stat.second_moments(im,cen)


    print("Irr acc:",Irracc)
    print("Irc acc:",Ircacc)
    print("Icc acc:",Iccacc)


    imfast /= imfast.sum()
    im /=im.sum()
    test_convolver_plot(im,imfast)

    return im, imfast
    
def test_func_convolver(epsabs=1.4899999999999999e-08, epsrel=1.4899999999999999e-08):
    from . import analytic

    cenrat=0.1
    sigma1=1.13
    sigma2=2.6

    #dg = analytic.DoubleGauss(cenrat,
    #                          sigma1**2, 0.0, sigma1**2,
    #                          sigma2**2, 0.0, sigma2**2)
    dg = analytic.DoubleGauss(cenrat,
                              sigma1**2, 0.0, sigma1**2,
                              sigma2**2, 0.0, sigma2**2)

    #e = analytic.Exp(1.0, 0.0, 1.0)
    e = analytic.Exp(8.0, 2.0, 4.0)

    # these are symmetric
    dgrange = dg.range()
    erange = e.range()

    # take smaller of the two for integration range
    rng = min(dgrange[1],erange[1])
    intrange = (-rng,rng)


    dim = int( numpy.ceil( 2*sqrt( dgrange[1]**2 + erange[1]**2 ) ) )
    # make odd so the fast convolution will work
    if (dim % 2) == 0:
        dim += 1
    dims = [dim,dim]
    cen = [(dim-1)/2]*2


    print("dgrange:",dgrange)
    print("erange:",erange)
    print("intrange:",intrange)
    print("dims:",dims)
    print("cen:",cen)


    #c = Convolver(e, dg, intrange, epsabs=epsabs,epsrel=epsrel)
    c = FuncConvolver(dg, e, intrange, epsabs=epsabs,epsrel=epsrel)

    imfast, imexp, imdg = c.make_image_fast(dims, cen, all=True)

    '''
    pexp=images.multiview(imexp,show=False)
    pexp.title='exp'
    pexp.show()
    pdg=images.multiview(imdg, show=False)
    pdg.title='double gauss'
    pdg.show()
    imconv=images.multiview(imfast, show=False)
    imconv.title='Convolved'
    imconv.show()
    '''


    #imdg2 = pixmodel.double_gauss(dims,cen,cenrat,
    #                            sigma1**2, 0.0, sigma1**2,
    #                            sigma2**2, 0.0, sigma2**2)

    #imdg /= imdg.sum()
    #imdg2 /= imdg2.sum()
    #return imdg, imdg2

    im = c.make_image(dims, cen)

    imfast /= imfast.sum()
    im /=im.sum()

    test_convolver_plot(im,imfast)

    return im, imfast
    
def test_convolver_plot(im, imfast):
    images.compare_images(im,imfast,label1='accurate',label2='fast')

