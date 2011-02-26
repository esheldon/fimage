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

from .transform import rebin

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

    def __init__(self, objpars, psfpars, **keys):

        self.objpars = objpars
        self.psfpars = psfpars

        self.conv=keys.get('conv', 'fft')

        conv_allow=['fconv','fconvint','fft','func']
        if self.conv not in conv_allow:
            raise ValueError("conv must be "+str(conv_allow))


        self.nsub=keys.get('nsub', 16)
        self.fft_nsub = keys.get('fft_nsub',1)
        print("  -> ConvolvedImage nsub:",self.nsub)
        if self.conv == 'fft':
            print("  -> ConvolvedImage fft_nsub:",self.fft_nsub)


        self.eps = keys.get('eps', 1.e-4)
        self.forcegauss=keys.get('forcegauss',False)

        self.debug=keys.get('debug',False)
        self.verbose=keys.get('verbose', False)


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
        covar_meas = stat.second_moments(self.psf, cen)
        res = admom.admom(self.psf, cen[0],cen[1], guess=(covar_meas[0]+covar_meas[2])/2)

        covar_meas_admom = array([res['Irr'],res['Irc'],res['Icc']])

        self.psfpars['covar_meas'] = covar_meas
        self.psfpars['covar_meas_admom'] = covar_meas_admom

        self['covar_psf'] = covar_meas
        self['covar_psf_admom'] = covar_meas_admom

        if self.verbose:
            print("PSF model")
            pprint(self.psfpars)

        if self.debug:
            plt=images.multiview(self.psf,levels=7, show=False)
            plt.title='PSF image'
            plt.show()

    def make_gauss_psf(self):
        pars=self.psfpars
        pars['covar'] = array(pars['covar'])
        covar = pars['covar']

        T = 2*max(covar)
        # getdimcen returns length 2 arrays
        dims,cen = self.getdimcen(T)

        self.psf = pixmodel.model_image('gauss',dims,cen,covar,
                                        nsub=self.nsub)

        pars['dims'] = dims
        pars['cen'] = cen

        if self.conv == 'func':
            self.psf_func=analytic.Gauss(covar)
       
    def make_dgauss_psf(self):

        pars=self.psfpars
        pars['covar1'] = array(pars['covar1'])
        pars['covar2'] = array(pars['covar2'])

        cenrat = pars['cenrat']
        covar1 = pars['covar1']
        covar2 = pars['covar2']

        Tmax = 2*max( max(covar1), max(covar2) )

        dims,cen = self.getdimcen(Tmax)

        self.psf = pixmodel.double_gauss(dims,cen,cenrat,covar1,covar2,
                                         nsub=self.nsub)
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
        pars['covar'] = array(pars['covar'])
        covar = pars['covar']

        if objmodel == 'gauss':
            # could probably be 3.5
            sigfac = 4.5
        elif objmodel == 'exp':
            sigfac = 7.0
        else:
            raise ValueError("Unsupported obj model: '%s'" % objmodel)

        # expected size
        Irr,Irc,Icc = covar

        # don't use admom here, we want the unweighted size!
        Irr_expect = Irr + psfpars['covar_meas'][0]
        Icc_expect = Icc + psfpars['covar_meas'][2]

        Texpect = 2*max(Irr_expect,Icc_expect)
        dims, cen = self.getdimcen(Texpect, sigfac=sigfac)

        if dims[0] < psfpars['dims'][0] and old == False:
            dims = psfpars['dims']
            cen = psfpars['cen']

        print("  dims:",dims,"cen:",cen)

        self.image0 = pixmodel.model_image(objmodel,dims,cen,covar,
                                           counts=pars['counts'], 
                                           nsub=self.nsub)

        pars['dims'] = dims
        pars['cen'] = cen
        covar_meas = stat.second_moments(self.image0, cen)
        res = admom.admom(self.image0, cen[0],cen[1], guess=(covar_meas[0]+covar_meas[1])/2 )
        covar_meas_admom = array([res['Irr'],res['Irc'],res['Icc']])
        pars['covar_meas'] = covar_meas
        pars['covar_meas_admom'] = covar_meas_admom
        self['covar_image0'] = covar_meas
        self['covar_image0_admom'] = covar_meas_admom

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

        if self.debug:
            plt=images.multiview(self.image0,levels=7, show=False)
            plt.title='image0'
            plt.show()



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
        if (objmodel == 'gauss' and psfmodel == 'gauss') and not self.forcegauss:
            print("doing analytic convolution of guassians for gauss")
            bothgauss=True
            ocovar=pars['covar']
            pcovar=psfpars['covar']
            covar = [ocovar[0]+pcovar[0],
                     ocovar[1]+pcovar[1],
                     ocovar[2]+pcovar[2]]

            image = pixmodel.model_image('gauss',dims,cen,covar,
                                       counts=pars['counts'], nsub=self.nsub)
        elif (objmodel == 'gauss' and psfmodel == 'dgauss') and not self.forcegauss:
            print("doing analytic convolution of guassians for dgauss")
            ocovar=pars['covar']
            pcovar1=psfpars['covar1']
            covar1 = [ocovar[0]+pcovar1[0],
                      ocovar[1]+pcovar1[1],
                      ocovar[2]+pcovar1[2]]

            im1 = pixmodel.model_image('gauss',dims,cen,covar1,
                                       counts=pars['counts'], nsub=self.nsub)

            pcovar2=psfpars['covar2']
            covar2 = [ocovar[0]+pcovar2[0],
                      ocovar[1]+pcovar2[1],
                      ocovar[2]+pcovar2[2]]

            im2 = pixmodel.model_image('gauss',dims,cen,covar2,
                                       counts=pars['counts'], nsub=self.nsub)

            s2 = psfpars['s2']
            b = psfpars['cenrat']
            image = im1 + b*s2*im2
            image /= (1+b*s2)

        else:
            if not have_scipy:
                raise ImportError("Could not import scipy")

            if self.conv == 'fft':
                image = self.make_image_fft()
            elif self.conv == 'fconvint':
                if objmodel != 'exp' or psfmodel not in ['gauss','dgauss']:
                    raise ValueError('fconvfunc only works on exp-gauss or exp-dgauss')

                print("running fconvint")
                if psfmodel == 'gauss':
                    image = fconv.conv_exp_gauss(dims,cen,pars['covar'], psfpars['covar'])
                elif psfmodel == 'dgauss':
                    s2 = psfpars['s2']
                    b = psfpars['cenrat']
                    im1= fconv.conv_exp_gauss(dims,cen,pars['covar'], psfpars['covar1'])
                    im2= fconv.conv_exp_gauss(dims,cen,pars['covar'], psfpars['covar2'])

                    image = im1 + b*s2*im2
                    image /= (1+b*s2)

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
                c = FuncConvolver(obj_func, psf_func, intrange, epsrel=self.eps,epsabs=self.eps)
                image = c.make_image(dims, cen)
                image *= (pars['counts']/image.sum())

        self.image = image
        self['dims'] = dims
        self['cen'] = cen
        if bothgauss:
            covar_meas = array(pars['covar']) + array(psfpars['covar'])
            covar_meas_admom = covar_meas
        else:
            covar_meas = stat.second_moments(self.image, cen)
            res = admom.admom(self.image, cen[0],cen[1], guess=(covar_meas[0]+covar_meas[2])/2)
            covar_meas_admom = array([res['Irr'],res['Irc'],res['Icc']])

        self['covar'] = covar_meas
        self['covar_admom'] = covar_meas_admom

        if self.verbose:
            print("convolved image pars")
            pprint(self)

        if self.debug:
            plt=images.multiview(self.image,levels=7, show=False)
            plt.title='convolved image'
            plt.show()



    def make_image_fft(self):
        fft_nsub=self.fft_nsub

        # this relies on the center being on a pixel
        # and fft_nsub being odd so the center doesn't shift

        # note by this point in the code, dims will be >= psf dims
        # so don't worry about that.

        pars=self.objpars
        ppars=self.psfpars

        objmodel = pars['model']
        dims = pars['dims']*fft_nsub
        cen_orig = pars['cen']
        covar = pars['covar']*fft_nsub**2
        
        psfmodel = ppars['model']
        psfdims = ppars['dims']*fft_nsub
        psfcen_orig = ppars['cen']

        # this is an array operation
        cen = (dims-1.)/2.

        image0_boosted = pixmodel.model_image(objmodel,dims,cen,covar,
                                              counts=pars['counts'], 
                                              nsub=self.nsub)

        psfcen = (psfdims-1.)/2.
        print("running fft convolve")
        if psfmodel == 'gauss':
            psfcovar = ppars['covar']*fft_nsub**2
            psf_boosted = pixmodel.model_image('gauss',psfdims,psfcen,psfcovar, nsub=self.nsub)

            print("  running fftconvolve")
            image_boosted = scipy.signal.fftconvolve(image0_boosted, psf_boosted, mode='same')
        else:
            b = ppars['cenrat']
            s2 = ppars['s2']
            covar1 = ppars['covar1']*fft_nsub**2
            covar2 = ppars['covar2']*fft_nsub**2
            g1 = pixmodel.model_image('gauss',psfdims,psfcen,covar1, nsub=self.nsub)
            g2 = pixmodel.model_image('gauss',psfdims,psfcen,covar2, nsub=self.nsub)

            print("  running fftconvolve1")
            im1 = scipy.signal.fftconvolve(image0_boosted, g1, mode='same')
            print("  running fftconvolve2")
            im2 = scipy.signal.fftconvolve(image0_boosted, g2, mode='same')

            image_boosted = im1 + b*s2*im2
            image_boosted /= (1+b*s2)

        if fft_nsub > 1:
            image = rebin(image_boosted, fft_nsub)
        else:
            image = image_boosted

        # make sure the center didn't shift
        mom0 = stat.moments(self.image0)
        mom = stat.moments(image)

        if self.verbose:
            print("  psf dims:",self.psf.shape)
            print("  psf cen: ",self.psfpars['cen'])
            #print("  boosted psf dims:",psf_boosted.shape)
            #pbmom = stat.moments(psf_boosted)
            #print("  boosted psf cen:", pbmom['cen'])
            print("  image0 dims:",self.image0.shape)
            print("  image0 cen:",self.objpars['cen'])
            print("  boosted image0 dims:",image0_boosted.shape)
            print("  boosted image0 cen:",cen)

            print("  fft created image dims:",image.shape)
            print("  fft created image cen:",mom['cen'])

        max_shift = max( abs(mom['cen'][0]-mom0['cen'][0]), abs(mom['cen'][1]-mom0['cen'][1]) )
        if (max_shift/mom['cen'][0]-1) > 0.0033:
            raise ValueError("Center rel shifted greater than 0.0033: %f" % max_shift)

        return image

    def mom2sigma(self, T):
        return sqrt(T/2)

    def getdimcen(self, T, sigfac=4.5):
        sigma = sqrt(T/2)
        imsize = int( numpy.ceil(2*sigfac*sigma) )

        # MUST BE ODD!
        if (imsize % 2) == 0:
            imsize+=1
        dims = array( [imsize]*2 )
        cen = array( [(imsize-1)/2]*2 )

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
        ret = scipy.integrate.quad(self._intfunc2, 
                                   self.range[0], 
                                   self.range[1], 
                                   args=self.range,
                                   epsabs=self.epsabs,
                                   epsrel=self.epsrel)
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

