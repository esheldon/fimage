from __future__ import print_function
from pprint import pprint
import numpy
from numpy import sqrt, exp, array, ceil, log2, pi, ogrid, zeros
from numpy.fft import fftshift

import images
from sys import stderr

from . import statistics as stat
from . import pixmodel
from . import fconv
from . import analytic
from . import conversions
from .conversions import mom2sigma, cov2sigma

from .transform import rebin

import time
import admom

try:
    import scipy.signal
    from scipy.fftpack import ifftn, fftn
    have_scipy=True
except:
    have_scipy=False



# sigma ~ fwhm/TURB_SIGMA_FAC
TURB_SIGMA_FAC=1.6
TURB_PADDING=10.0

GAUSS_PADDING=5.0
EXP_PADDING=7.0

def wlog(*args):
    narg = len(args)
    for i,arg in enumerate(args):
        stderr.write("%s" % arg)
        if i < (narg-1):
            stderr.write(" ")
    stderr.write('\n')


def convolve_gauss(image, sigma, get_psf=False):
    """
    Convolve with a round gaussian.
    
    The kernel is generated and applied in fourier space.

    Note for more general convolutions, use the ConvolvedImageFFT class.

    parameters
    ----------
    image:
        A two dimensional image.
    sigma:
        The sigma of the gaussian kernel.
    get_psf:
        If True, return an image of the psf as well.
            (im, psf)
        Dims should be even to get a psf that is centered
    """

    dims = array(image.shape)
    if dims[0] != dims[1]:
        raise ValueError("only square images for now")

    # Always use 2**n-sized FFT
    size=dims
    kdims = 2**ceil(log2(size))
    kcen = kdims/2.

    imfft = fftn(image,kdims)

    krow,kcol=ogrid[0:kdims[0], 0:kdims[1]]

    kr = array(krow - kcen[0], dtype='f8')
    kc = array(kcol - kcen[1], dtype='f8')
    k2 = kr**2 + kc**2

    ksigma = 1.0/sigma
    # get into fft units
    ksigma *= kdims[0]/(2.*pi)

    ksigma2 = ksigma**2
    gk = exp(-0.5*k2/ksigma2)
    gk = fftshift(gk) 

    ckim = gk*imfft

    cim = ifftn(ckim)[0:dims[0], 0:dims[1]]
    cim = cim.real

    if get_psf:
        psf = ifftn(gk)
        psf = fftshift(psf)
        psf = sqrt(psf.real**2 + psf.imag**2)
        psf = pixmodel._centered(psf, dims)
        return cim, psf
    else:
        return cim


def convolve_turb(image, fwhm, get_psf=False):
    """
    Convolve the input image with a turbulent psf

    parameters
    ----------
    image:
        A numpy array
    fwhm:
        The FWHM of the turbulent psf.
    get_psf:
        If True, return a tuple (im,psf)

    The images dimensions should be square, and even in order for the psf to be
    centered.
    """

    dims = array(image.shape)
    if dims[0] != dims[1]:
        raise ValueError("only square images for now")

    # Always use 2**n-sized FFT
    kdims = 2**ceil(log2(dims))
    kcen = kdims/2.

    imfft = fftn(image,kdims)

    k0 = 2.92/fwhm
    # in fft units
    k0 *= kdims[0]/(2*pi)

    otf = pixmodel.ogrid_turb_kimage(kdims, kcen, k0)
    otf = fftshift(otf) 

    ckim = otf*imfft
    cim = ifftn(ckim)[0:dims[0], 0:dims[1]]
    cim = cim.real

    if get_psf:
        psf = ifftn(otf)
        psf = fftshift(psf)
        psf = sqrt(psf.real**2 + psf.imag**2)
        psf = pixmodel._centered(psf, dims)
        return cim, psf
    else:
        return cim


class ConvolverBase(dict):
    def __init__(self, objpars, psfpars, **keys):
        """
        Abstract base class
        """
        self.objpars=objpars
        self.psfpars=psfpars

        if self.objpars['model'] not in ['gauss','exp']:
            raise ValueError("only support gauss/exp objects")
        if self.psfpars['model'] not in ['gauss','dgauss','turb']:
            raise ValueError("only support gauss/dgauss/turb psf")

        if 'cov' in self.objpars:
            self.objpars['cov'] = array(self.objpars['cov'],dtype='f8')
        if 'cov' in self.psfpars:
            self.psfpars['cov'] = array(self.psfpars['cov'],dtype='f8')
 
        # we want to try to let the expansion factor do the trick
        self['image_nsub'] = keys.get('image_nsub', 1)

        # for calculations we will demand sigma > minres pixels
        # then sample back
        self['minres'] = keys.get('minres',12)

        # Hint for how large to make the base image
        if self.objpars['model'] == 'exp':
            self['objfac']=EXP_PADDING
        else:
            self['objfac']=GAUSS_PADDING
        self.image0=None
        self.image=None
        self.psf=None

    def make_images(self):
        """
        Make an image convolved with the psf.
        """
        raise RuntimeError("over-ride this")

    def set_cov_and_etrue(self):
        cov = self.objpars['cov']
        T=(cov[2]+cov[0])
        self['covtrue'] = self.objpars['cov']
        self['e1true'] = (cov[2]-cov[0])/T
        self['e2true'] = 2*cov[1]/T
        self['etrue'] = sqrt( self['e1true']**2 + self['e2true']**2 )


    def get_image0(self, verify=False, expand=False):
        """
        Create the pre-psf model

        run these first
            self.set_cov_and_etrue()
            self.set_dims()
        """

        pars = self.objpars
        objmodel = pars['model']

        if expand:
            cen  = self['ecen']
            dims = self['edims']
            cov  = pars['ecov']
        else:
            cen  = self['cen']
            dims = self['dims']
            cov  = pars['cov']

        image0 = pixmodel.model_image(objmodel,dims,cen,cov,
                                      nsub=self['image_nsub'])

        if verify:
            self.verify_image(image0, cov)
        return image0


    def add_image0_stats(self):
        mom = stat.fmom(self.image0)
        cov_meas = mom['cov']
        cen_meas = mom['cen']

        res = admom.admom(self.image0, cen_meas[0],cen_meas[1], 
                          guess=(cov_meas[0]+cov_meas[1])/2 )
        cov_meas_admom = array([res['Irr'],res['Irc'],res['Icc']])

        pars = self.objpars

        mom = stat.fmom(self.image0)
        cov_uw = mom['cov']
        cen_uw = mom['cen']
        res = admom.admom(self.image0, cen_uw[0],cen_uw[1], 
                          guess=(cov_uw[0]+cov_uw[2])/2)
        
        cov_admom = array([res['Irr'],res['Irc'],res['Icc']])
        cen_admom = array([res['wrow'], res['wcol']])

        pars['cov_uw'] = cov_uw
        pars['cen_uw'] = cen_uw
        pars['cov_admom'] = cov_admom
        pars['cen_admom'] = cen_admom
        self['cov_image0_uw'] = cov_uw
        self['cen_image0_uw'] = cen_uw
        self['cov_image0_admom'] = cov_admom
        self['cen_image0_admom'] = cen_admom


    def add_psf_stats(self):
        mom = stat.fmom(self.psf)
        cov_uw = mom['cov']
        cen_uw = mom['cen']

        res = admom.admom(self.psf,cen_uw[0],cen_uw[1], 
                          guess=(cov_uw[0]+cov_uw[2])/2)

        
        cov_admom = array([res['Irr'],res['Irc'],res['Icc']])
        cen_admom = array([res['wrow'], res['wcol']])

        self.psfpars['cov_uw'] = cov_uw
        self.psfpars['cen_uw'] = cen_uw
        self.psfpars['cov_admom'] = cov_admom
        self.psfpars['cen_admom'] = cen_admom

        self['cov_psf_uw'] = cov_uw
        self['cen_psf_uw'] = cen_uw
        self['cov_psf_admom'] = cov_admom
        self['cen_psf_admom'] = cen_admom

    def add_image_stats(self):
        mom_uw = stat.fmom(self.image)
        cov_uw = mom_uw['cov']
        cen_uw = mom_uw['cen']

        res = admom.admom(self.image, cen_uw[0], cen_uw[1], 
                          guess=(cov_uw[0]+cov_uw[2])/2)
        cov_admom = array([res['Irr'],res['Irc'],res['Icc']])
        cen_admom = array([res['wrow'], res['wcol']])

        self['cov_uw'] = cov_uw
        self['cen_uw'] = cen_uw
        self['cov_admom'] = cov_admom
        self['cen_admom'] = cen_admom

    def verify_image(self, image, cov, eps=2.e-3):
        '''

        Ensure that the *unweighted* moments are equal to input moments

        This is only useful for expanded images since unweighted moments don't
        include sub-pixel effects

        '''

        mom = stat.fmom(image)
        mcov = mom['cov']

        rowrel = abs(mcov[0]/cov[0]-1)
        colrel = abs(mcov[2]/cov[2]-1)

        pdiff = max(rowrel,colrel)
        if pdiff > eps:
            raise ValueError("row pdiff %f not within "
                             "tolerance %f" % (pdiff,eps))

        T = mcov[2] + mcov[0]
        e1 = (mcov[2]-mcov[1])/T
        e2 = 2*mcov[1]/T
        e = sqrt(e1**2 + e2**2)

        Ttrue = cov[2] + cov[0]
        e1true = (cov[2]-cov[1])/T
        e2true = 2*cov[1]/T
        etrue = sqrt(e1true**2 + e2true**2)

        erel = abs(e/etrue-1)
        if erel > eps:
            raise ValueError("moments pdiff %f not within "
                             "tolerance %f" % (erel,eps))


class ConvolverGaussFFT(ConvolverBase):
    """
    Convolve models with a gaussian or double gaussian psf
    """
    def __init__(self, objpars, psfpars, **keys):
        """
        Gaussian fft with some model
        """
        super(ConvolverGaussFFT,self).__init__(objpars,psfpars,**keys)
            
        self['psffac'] = GAUSS_PADDING

        # inherited
        self.set_cov_and_etrue()

        # these implemented in this class
        self.set_dims_and_expansion()
        self.make_images()

        # these inherited
        self.add_image0_stats()
        self.add_psf_stats()
        self.add_image_stats()

    def set_dims_and_expansion(self):
        """
        All images are created in reall space, so want odd
        """

        psffac = GAUSS_PADDING
        objfac = self['objfac']

        obj_cov = self.objpars['cov']
        if self.psfpars['model'] == 'gauss':
            psf_cov=self.psfpars['cov']
        elif self.psfpars['model'] == 'dgauss':
            psf_cov1 = self.psfpars['cov1']
            psf_cov2 = self.psfpars['cov2']
            psf_cov  = zeros(3)
            psf_cov[0] = max(psf_cov1[0], psf_cov2[0])
            psf_cov[2] = max(psf_cov1[2], psf_cov2[2])
        else:
            raise ValueError("model should be gauss or dgauss")

        sigma_psf = cov2sigma(psf_cov)
        sigma_obj = cov2sigma(obj_cov)

        imsize = 2*sqrt( psffac**2*sigma_psf**2 + objfac**2*sigma_obj**2)
        dims = array([imsize]*2,dtype='i8')

        self['dims'] = dims
        if (dims[0] % 2) != 0:
            dims += 1

        self['cen'] = (dims-1.)/2.


        #
        # do we need to expand before convolving?
        #

        # the idea of "sigma" has no meaning for this type of psf
        # but this constant is reasonably close for our apertures
        sigma_min = min(sigma_psf,sigma_obj)

        fac = 1
        if sigma_min < self['minres']:
            # find the odd integer expansion that will get sigma > minres
            fac = int(self['minres']/sigma_min)

        if (fac % 2) == 0:
            fac += 1
        self['expand_fac'] = fac
        if fac > 1:
            self['edims'] = fac*self['dims']
            self['ecen'] = (self['edims']-1.)/2.
            self.objpars['ecov'] = self.objpars['cov']*fac**2
        else:
            self['edims'] = self['dims']
            self['ecen'] = self['cen']
            self.objpars['ecov'] = self.objpars['cov']



    def make_images(self):
        """
        Make an image in real space, go to fourier space and multiply by
        the psf, then fft back.

        If we are expanding, we create an expanded pre-psf image and convolve
        it.  The convolved image and psf are rebinned back.
        """
        nsub = self['image_nsub']

        if self['expand_fac'] < 1:
            raise ValueError("expected expansion >= 1")

        fac = self['expand_fac']
        if fac == 1:
            dims = self['dims']
            cen = self['cen']
            expand=False
        else:
            expand=True
            dims = self['edims']
            cen = self['ecen']

        verify=False
        image0 = self.get_image0(expand=expand,verify=verify)
        if self.psfpars['model'] == 'dgauss':
            b=self.psfpars['cenrat']
            psf_cov1=self.psfpars['cov1']*fac**2
            psf_cov2=self.psfpars['cov2']*fac**2
            psf1 = pixmodel.model_image('gauss', dims, cen,
                                         psf_cov1,nsub=nsub)
            psf2 = pixmodel.model_image('gauss', dims, cen,
                                         psf_cov2,nsub=nsub)
            eim1 = scipy.signal.fftconvolve(image0, psf1, mode='same')
            eim2 = scipy.signal.fftconvolve(image0, psf2, mode='same')

            image = (eim1 + b*eim2)/(1.+b)
            psf = (psf1 + b*psf2)/(1.+b)
        else:
            psf_cov=self.psfpars['cov']*fac**2
            psf = pixmodel.model_image('gauss', dims, cen,
                                        psf_cov,nsub=nsub)
            image = scipy.signal.fftconvolve(image0, psf, mode='same')

        if fac > 1:
            image0 = rebin(image0, fac)
            image = rebin(image, fac)
            psf   = rebin(psf, fac)

        self.image0 = image0
        self.image = image
        self.psf   = psf




class ConvolverAllGauss(ConvolverBase):
    """
    Convolve gauss models with a gaussian or double gaussian psf
    """
    def __init__(self, objpars, psfpars, **keys):
        """
        all gaussians all the time
        """
        super(ConvolverAllGauss,self).__init__(objpars,psfpars,**keys)

        # these inherited
        self.set_cov_and_etrue()

        # these implemented in this class
        self.set_dims()
        self.make_images()

        # these inherited
        self.add_image0_stats()
        self.add_psf_stats()
        self.add_image_stats()

    def set_dims(self):
        """
        Simple for analytic convolutions
        """

        
        fac = GAUSS_PADDING

        obj_cov = self.objpars['cov']

        if self.psfpars['model'] == 'gauss':
            psf_cov=self.psfpars['cov']
        elif self.psfpars['model'] == 'dgauss':
            psf_cov1 = self.psfpars['cov1']
            psf_cov2 = self.psfpars['cov2']
            psf_cov  = zeros(3)
            psf_cov[0] = max(psf_cov1[0], psf_cov2[0])
            psf_cov[2] = max(psf_cov1[2], psf_cov2[2])
        else:
            raise ValueError("model should be gauss or dgauss")

        cov = obj_cov + psf_cov
        sigma = cov2sigma(cov) 
        imsize = fac*2*sigma
        dims = array([imsize]*2,dtype='i8')
        if (dims[0] % 2) == 0:
            dims += 1
        cen=(dims-1)/2

        self['dims'] = dims
        self['cen'] = cen


    def make_images(self):
        """
        This is easy!  We also force nsub=16 since there is no expansion crap
        to deal with
        """
        nsub=16

        objcov = self.objpars['cov']
        self.image0 = pixmodel.model_image('gauss',
                                           self['dims'],
                                           self['cen'],
                                           objcov,nsub=nsub)

        if self.psfpars['model'] == 'dgauss':
            psf_cov1 = self.psfpars['cov1']
            psf_cov2 = self.psfpars['cov2']

            cov1 = objcov + psf_cov1
            cov2 = objcov + psf_cov2
            b = self.psfpars['cenrat']

            im1 = pixmodel.model_image('gauss', self['dims'], self['cen'],
                                       cov1,nsub=nsub)
            im2 = pixmodel.model_image('gauss', self['dims'], self['cen'],
                                       cov2,nsub=nsub)
            self.image = (im1 + b*im2)/(1+b)

            psf1 = pixmodel.model_image('gauss', self['dims'], self['cen'],
                                        psf_cov1,nsub=nsub)
            psf2 = pixmodel.model_image('gauss', self['dims'], self['cen'],
                                        psf_cov2,nsub=nsub)

            self.psf = (psf1 + b*psf2)/(1+b)
        else:
            psf_cov = self.psfpars['cov']
            cov = objcov + psf_cov
            self.image = \
                pixmodel.model_image('gauss', self['dims'], self['cen'],
                                     cov,nsub=nsub)

            self.psf = pixmodel.model_image('gauss', self['dims'], self['cen'],
                                            psf_cov,nsub=nsub)

class ConvolverTurbulence(ConvolverBase):
    """
    Convolve models with a turbulent psf
    """
    def __init__(self, objpars, psfpars, **keys):
        """
        Some model convolved with turbulent psf
        """
        super(ConvolverTurbulence,self).__init__(objpars,psfpars,**keys)

        self['psffac'] = TURB_PADDING
        if self.objpars['model'] == 'exp':
            # The broad exponential must feel the outer psf
            self['psffac'] *= 1.5

        # 6 seems good enough.  Can we do this in ConvolverGaussFFT?
        self['minres'] = 6


        # inherited
        self.set_cov_and_etrue()

        self.set_dims_and_expansion()

        # this implemented in this class
        self.make_images()

        # these inherited
        self.add_image0_stats()
        self.add_psf_stats()
        self.add_image_stats()

    def set_dims_and_expansion(self):
        """
        Need huge space around the psf because of the wings
        """
        psffac=self['psffac']
        objfac=self['objfac']

        imcov=self.objpars['cov']
        obj_sigma = cov2sigma(imcov)

        fwhm = self.psfpars['psf_fwhm']
        imsize = 2*sqrt( (psffac*fwhm)**2 + objfac**2*obj_sigma**2 )
        dims = array([imsize]*2,dtype='i8')

        if (dims[0] % 2) != 0:
            dims += 1
        self['dims'] = dims
        self['cen'] = dims/2

        sigma_obj = cov2sigma(imcov)

        # the idea of "sigma" has no meaning for this type of psf
        # be conservative and make it seem small, which will result
        # in extra expansion
        #sigma_psf = fwhm/TURB_SIGMA_FAC
        sigma_psf = fwhm/4.
        sigma_min = min(sigma_psf,sigma_obj)

        # do we need to expand before convolving?
        fac=1
        if sigma_min < self['minres']:
            # find the expansion that will get sigma > minres
            fac = int(self['minres']/sigma_min)

        #if fac < 16:
        #    fac=16.
        self['expand_fac'] = fac

        if fac > 1:
            # modify the dims to be odd, so the rebin will
            # give the right center
            # this will force the psf to be off center, but 
            # not much we can do!
            if (self['dims'][0] % 2) == 0:
                self['dims'] += 1
                self['cen'] = (self['dims']-1.)/2.
            self['edims'] = fac*self['dims']
            self['ecen'] = (self['edims']-1)/2
            self.objpars['ecov'] = self.objpars['cov']*fac**2
        else:
            self['edims'] = self['dims']
            self['ecen'] = self['cen']
            self.objpars['ecov'] = self.objpars['cov']

    def make_images(self):
        """
        Make an image in real space, go to fourier space and multiply by
        the psf, then fft back.

        If we are expanding, we create an expanded pre-psf image and convolve
        it.  The convolved image and psf are rebinned back.
        """

        fwhm = self.psfpars['psf_fwhm']
        if self['expand_fac'] > 1:
            #eimage0 = self.get_image0(expand=True, verify=True)
            eimage0 = self.get_image0(expand=True)
            efwhm = fwhm*self['expand_fac']
            eimage,epsf = convolve_turb(eimage0,efwhm,get_psf=True)

            image0 = rebin(eimage0, self['expand_fac'])
            image = rebin(eimage, self['expand_fac'])
            psf   = rebin(epsf, self['expand_fac'])
        else:
            image0 = self.get_image0()
            image,psf = convolve_turb(self.image0,fwhm,get_psf=True)

        self.image0 = image0
        self.image = image
        self.psf   = psf




# deprecated, but it is still used in the regauss
# sims, so leave it for now
class ConvolvedImageFFT(dict):
    '''

    This one always uses FFTs when the result cannot be computed analytically
    (gaussians) or forcegauss=True

    We can begin with a higher resolution image and sample back down to the
    requested resolution.

    The relevant parameter for expansion should be the ratio of the "sigma" of
    the smallest object to the pixel size.  For the convolutions we always want
    to be at sigma >> 1, say 10.  The images are verified to have moments and
    ellip within 0.001 of the true.

    Note the center is always placed exactly on a pixel and the expansion factors
    are odd so this remains true during the fft

    '''

    def __init__(self, objpars, psfpars, **keys):
        self.objpars = objpars
        self.psfpars = psfpars

        if self.objpars['model'] not in ['gauss','exp']:
            raise ValueError("only support gauss/exp objects")
        if self.psfpars['model'] not in ['gauss','dgauss','turb']:
            raise ValueError("only support gauss,dgauss,turb psfs")

        self['image_nsub'] = keys.get('image_nsub', 16)
        self['forcegauss'] = keys.get('forcegauss',False)
        self['debug']      = keys.get('debug',False)
        self['verbose']    = keys.get('verbose', False)

        # for calculations we will demand sigma > minres pixels
        # then sample back
        self['minres'] = keys.get('minres',10)

        if self['verbose']:
            wlog("  -> ConvolvedImage image nsub:",self['image_nsub'])

        self['allgauss'] = False
        if objpars['model'] == 'gauss' and psfpars['model'] in ['gauss','dgauss']:
            self['allgauss'] = True

        self.prep_cov()

        cov = self.objpars['cov']
        T=(cov[2]+cov[0])
        self['covtrue'] = self.objpars['cov']
        self['e1true'] = (cov[2]-cov[0])/T
        self['e2true'] = 2*cov[1]/T
        self['etrue'] = sqrt( self['e1true']**2 + self['e2true']**2 )

        self.get_dims()
        self.make_psf()
        self.make_image0()
        self.make_image()

    def prep_cov(self):
        '''
        Make sure all the covariances are arrays
        '''
        self.objpars['cov'] = array(self.objpars['cov'],dtype='f8')

        psfmodel = self.psfpars['model']
        if psfmodel == 'gauss':
            self.psfpars['cov'] = array(self.psfpars['cov'],dtype='f8')
        elif psfmodel == 'dgauss':
            self.psfpars['cov1'] = array(self.psfpars['cov1'],dtype='f8')
            self.psfpars['cov2'] = array(self.psfpars['cov2'],dtype='f8')
        else:
            raise ValueError("unknown model type: '%s'" % psfmodel)

    def make_psf(self):
        psfmodel = self.psfpars['model']
        if psfmodel == 'gauss':
            self.psf = self.get_gauss_psf()
        elif psfmodel == 'dgauss':
            psf = self.get_dgauss_psf()
            self.psf = psf
        else:
            raise ValueError("unknown model type: '%s'" % psfmodel)

        mom = stat.fmom(self.psf)
        cov_uw = mom['cov']
        cen_uw = mom['cen']

        res = admom.admom(self.psf,cen_uw[0],cen_uw[1], 
                          guess=(cov_uw[0]+cov_uw[2])/2)

        
        cov_admom = array([res['Irr'],res['Irc'],res['Icc']])
        cen_admom = array([res['wrow'], res['wcol']])

        self.psfpars['cov_uw'] = cov_uw
        self.psfpars['cen_uw'] = cen_uw
        self.psfpars['cov_admom'] = cov_admom
        self.psfpars['cen_admom'] = cen_admom

        self['cov_psf_uw'] = cov_uw
        self['cen_psf_uw'] = cen_uw
        self['cov_psf_admom'] = cov_admom
        self['cen_psf_admom'] = cen_admom

        if self['verbose'] > 1:
            wlog("PSF model")
            pprint(self.psfpars,stream=stderr)

        if self['debug']:
            plt=images.multiview(self.psf,levels=7, show=False)
            plt.title='PSF image'
            plt.show()

        self.psfpars['cen'] = self['cen']


    def get_gauss_psf(self, expand=False, verify=False):
        pars=self.psfpars

        if expand:
            cen  = self['ecen']
            dims = self['edims']
            cov  = pars['cov']*self['expand_fac']**2
        else:
            cen  = self['cen']
            dims = self['dims']
            cov  = pars['cov']

        psf = pixmodel.model_image('gauss',dims,cen,cov,nsub=self['image_nsub'])

        if verify:
            if self['verbose']:
                wlog("    verifying gauss psf")
            self.verify_image(psf, cov)
        return psf

    def get_dgauss_psf(self, expand=False, verify=False, both=False):

        pars=self.psfpars

        if expand:
            cen    = self['ecen']
            dims   = self['edims']
            cenrat = pars['cenrat']
            cov1   = pars['cov1']*self['expand_fac']**2
            cov2   = pars['cov2']*self['expand_fac']**2
        else:
            cen    = self['cen']
            dims   = self['dims']
            cenrat = pars['cenrat']
            cov1   = pars['cov1']
            cov2   = pars['cov2']

        det1=conversions.cov2det(cov1)
        det2=conversions.cov2det(cov2)
        s2 = sqrt(det2/det1)

        im1 = pixmodel.model_image('gauss',dims,cen,cov1, nsub=self['image_nsub'])
        im2 = pixmodel.model_image('gauss',dims,cen,cov2, nsub=self['image_nsub'])

        if both:
            return im1,im2
        else:
            b = pars['cenrat']
            psf = im1 + b*im2
            psf /= (1+b)

            if verify:
                if self['verbose']:
                    wlog("    verifying dgauss psf")
                self.verify_image(im1, cov1)
                self.verify_image(im2, cov2)

            return psf

    def make_image0(self):
        self.image0 = self.get_image0()

        mom = stat.fmom(self.image0)
        cov_meas = mom['cov']
        cen_meas = mom['cen']
        #cov_meas = stat.second_moments(self.image0, cen)

        res = admom.admom(self.image0, cen_meas[0],cen_meas[1], 
                          guess=(cov_meas[0]+cov_meas[1])/2 )
        cov_meas_admom = array([res['Irr'],res['Irc'],res['Icc']])

        pars = self.objpars


        mom = stat.fmom(self.image0)
        cov_uw = mom['cov']
        cen_uw = mom['cen']
        res = admom.admom(self.image0, cen_uw[0],cen_uw[1], 
                          guess=(cov_uw[0]+cov_uw[2])/2)
        
        cov_admom = array([res['Irr'],res['Irc'],res['Icc']])
        cen_admom = array([res['wrow'], res['wcol']])

        pars['cov_uw'] = cov_uw
        pars['cen_uw'] = cen_uw
        pars['cov_admom'] = cov_admom
        pars['cen_admom'] = cen_admom
        self['cov_image0_uw'] = cov_uw
        self['cen_image0_uw'] = cen_uw
        self['cov_image0_admom'] = cov_admom
        self['cen_image0_admom'] = cen_admom


        if self['verbose'] > 1:
            wlog("image0 pars")
            pprint(self.objpars,stream=stderr)

        if self['debug']:
            plt=images.multiview(self.image0,levels=7, show=False)
            plt.title='image0'
            plt.show()

    def get_image0(self, expand=False, verify=False):

        pars = self.objpars
        if 'counts' not in pars:
            pars['counts'] = 1.0

        if expand:
            cen  = self['ecen']
            dims = self['edims']
            cov  = pars['cov']*self['expand_fac']**2
        else:
            cen  = self['cen']
            dims = self['dims']
            cov  = pars['cov']


        objmodel = pars['model']
        image0 = pixmodel.model_image(objmodel,dims,cen,cov,
                                      counts=pars['counts'], 
                                      nsub=self['image_nsub'])

        if verify:
            if self['verbose']:
                wlog("    verifying image0")
            self.verify_image(image0, cov)
        return image0


    def make_image(self):


        if self['verbose']:
            wlog("Convolving to final image")

        image0 = self.image0
        psf = self.psf

        if (self['allgauss']) and not self['forcegauss']:
            if self['verbose']:
                wlog("doing analytic convolution of guassians for gauss")
            image = self.get_analytic_conv()
        else:
            image = self.get_fft_conv()

        self.image=image

        mom_uw = stat.fmom(self.image)
        cov_uw = mom_uw['cov']
        cen_uw = mom_uw['cen']

        res = admom.admom(self.image, cen_uw[0], cen_uw[1], 
                          guess=(cov_uw[0]+cov_uw[2])/2)
        cov_admom = array([res['Irr'],res['Irc'],res['Icc']])
        cen_admom = array([res['wrow'], res['wcol']])

        self['cov_uw'] = cov_uw
        self['cen_uw'] = cen_uw
        self['cov_admom'] = cov_admom
        self['cen_admom'] = cen_admom

        if self['verbose'] > 1:
            wlog("convolved image pars")
            pprint(self,stream=stderr)

        if self['debug']:
            plt=images.multiview(self.image,levels=7, show=False)
            plt.title='convolved image'
            plt.show()



    def verify_image(self, image, cov, eps=2.e-3):
        '''

        Ensure that the *unweighted* moments are equal to input moments

        This is only useful for expanded images since unweighted moments don't
        include sub-pixel effects

        '''

        mom = stat.fmom(image)
        mcov = mom['cov']

        rowrel = abs(mcov[0]/cov[0]-1)
        colrel = abs(mcov[2]/cov[2]-1)

        pdiff = max(rowrel,colrel)
        if pdiff > eps:
            raise ValueError("row pdiff %f not within "
                             "tolerance %f" % (pdiff,eps))

        T = mcov[2] + mcov[0]
        e1 = (mcov[2]-mcov[1])/T
        e2 = 2*mcov[1]/T
        e = sqrt(e1**2 + e2**2)

        Ttrue = cov[2] + cov[0]
        e1true = (cov[2]-cov[1])/T
        e2true = 2*cov[1]/T
        etrue = sqrt(e1true**2 + e2true**2)

        erel = abs(e/etrue-1)
        if erel > eps:
            raise ValueError("moments pdiff %f not within "
                             "tolerance %f" % (erel,eps))
        
        if self['verbose'] > 1:
            wlog("        moment fdiff: %e" % pdiff)
            wlog("        ellip  fdiff:   %e" % erel)



    def get_fft_conv(self):
        if self['expand_fac'] > 1:

            eimage0 = self.get_image0(expand=True, verify=True)
            psfmodel = self.psfpars['model']
            if psfmodel == 'gauss':
                epsf = self.get_gauss_psf(expand=True, verify=True)

                eimage = scipy.signal.fftconvolve(eimage0, epsf, mode='same')
                image = rebin(eimage, self['expand_fac'])

            elif psfmodel == 'dgauss':

                epsf1,epsf2 = \
                    self.get_dgauss_psf(expand=True, verify=True, both=True)
                eimage1 = scipy.signal.fftconvolve(eimage0, epsf1, mode='same')
                eimage2 = scipy.signal.fftconvolve(eimage0, epsf2, mode='same')
                b=self.psfpars['cenrat']
                eimage = (eimage1 + b*eimage2)/(1+b)

            else:
                raise ValueError("unknown model type: '%s'" % psfmodel)

            #eimage = scipy.signal.fftconvolve(eimage0, epsf, mode='same')
            image = rebin(eimage, self['expand_fac'])
        else:
            image = scipy.signal.fftconvolve(self.image0, self.psf, mode='same')

        # make sure the center didn't shift
        mom0 = stat.moments(self.image0)
        mom = stat.moments(image)

        max_shift = max( abs(mom['cen'][0]-mom0['cen'][0]), 
                         abs(mom['cen'][1]-mom0['cen'][1]) )
        if (max_shift/mom['cen'][0]-1) > 0.0033:
            raise ValueError("Center rel shifted greater than "
                             "0.0033: %f" % max_shift)

        return image


    def get_analytic_conv(self):

        pars    = self.objpars
        psfpars = self.psfpars
        dims    = self['dims']
        cen     = self['cen']

        ocov=pars['cov']
        if self.psfpars['model'] == 'gauss':
            pcov=psfpars['cov']
            cov = ocov + pcov

            image = pixmodel.model_image('gauss',dims,cen,cov,
                                         counts=pars['counts'], 
                                         nsub=self['image_nsub'])
        else:
            pcov1 = psfpars['cov1']
            pcov2 = psfpars['cov2']
            cov1  = ocov + pcov1
            cov2  = ocov + pcov2

            im1 = pixmodel.model_image('gauss',dims,cen,cov1,
                                       counts=pars['counts'], 
                                       nsub=self['image_nsub'])
            im2 = pixmodel.model_image('gauss',dims,cen,cov2,
                                       counts=pars['counts'], 
                                       nsub=self['image_nsub'])
            #s2 = psfpars['s2']
            b = psfpars['cenrat']
            #image = im1 + b*s2*im2
            #image /= (1+b*s2)
            image = im1 + b*im2
            image /= (1+b)

        return image



    def get_dims(self):
        '''

        Get the dimensions of the image objects.

        Determine by how much we need to expand in order to get accurate
        convolutions.

        '''

        p = self.objpars
        pp = self.psfpars
        sigma_max = sqrt( max(p['cov'][0],p['cov'][2]) )
        sigma_min = sqrt( min(p['cov'][0],p['cov'][2]) )

        odd=True
        if pp['model'] == 'gauss':
            sigma_psf_min = sqrt( min(pp['cov'][0], pp['cov'][2]) )
            sigma_psf_max = sqrt( max(pp['cov'][0], pp['cov'][2]) )
        elif pp['model'] == 'dgauss':
            sigma_psf_min1 = sqrt( min(pp['cov1'][0], pp['cov1'][2]) )
            sigma_psf_max1 = sqrt( max(pp['cov1'][0], pp['cov1'][2]) )

            sigma_psf_min2 = sqrt( min(pp['cov2'][0], pp['cov2'][2]) )
            sigma_psf_max2 = sqrt( max(pp['cov2'][0], pp['cov2'][2]) )

            sigma_psf_min = min(sigma_psf_min1,sigma_psf_min2)
            sigma_psf_max = max(sigma_psf_max1,sigma_psf_max2)
        else:
            raise ValueError("only support gauss/dgauss psf")

        # padding to large radius
        # number of sigma
        nsig=4.5
        #if p['model'] == 'exp' and pp['model'] == 'gauss':
        if p['model'] == 'exp':
            nsig=7.0

        Texpect = 2*(sigma_max**2 + sigma_psf_max**2)
        dims,cen = self._get_dimcen(Texpect, nsig=nsig,odd=odd)

        # now see if we need to expand in order to sample the sharp
        # inner region, use the smallest dimension
        sigma_min = min(sigma_min, sigma_psf_min)

        if sigma_min > self['minres']:
            fac=1
        else:
            # find the odd integer expansion that will get sigma > minres
            fac = int(self['minres']/sigma_min)
            if (fac % 2) == 0:
                fac += 1

        self['dims'] = dims
        self['cen'] = cen
        self['expand_fac'] = fac

        if fac > 1:
            self['edims'] = fac*self['dims']
            self['ecen'] = array( [(self['edims'][0]-1.)/2.]*2 )
        else:
            self['edims'] = self['dims']
            self['ecen'] = self['cen']

        if self['verbose']:
            wlog("  -> minres:    ",self['minres'])
            wlog("  -> expand_fac:",self['expand_fac'])

    def _get_dimcen(self, T, nsig=4.5, odd=True):
        sigma = sqrt(T/2)
        imsize = int( numpy.ceil(2*nsig*sigma) )

        if odd:
            if (imsize % 2) == 0:
                imsize+=1
            cen = array( [(imsize-1)/2]*2 )
        else:
            if (imsize % 2) != 0:
                imsize+=1
            cen = array( [imsize/2.]*2 )
        dims = array( [imsize]*2 )

        return dims,cen



# deprecated
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


        self.image_nsub=keys.get('image_nsub', 16)
        self.fft_nsub = keys.get('fft_nsub',1)
        self.fconvint_nsub = keys.get('fconvint_nsub',4)

        if self['verbose']:
            wlog("  -> ConvolvedImage image nsub:",self.image_nsub)
            if self.conv == 'fft':
                wlog("  -> ConvolvedImage fft_nsub:",self.fft_nsub)


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
        cov_meas = stat.second_moments(self.psf, cen)
        res = admom.admom(self.psf, cen[0],cen[1], guess=(cov_meas[0]+cov_meas[2])/2)

        cov_meas_admom = array([res['Irr'],res['Irc'],res['Icc']])

        self.psfpars['cov_meas'] = cov_meas
        self.psfpars['cov_meas_admom'] = cov_meas_admom

        self['cov_psf'] = cov_meas
        self['cov_psf_admom'] = cov_meas_admom

        if self.verbose:
            wlog("PSF model")
            pprint(self.psfpars,stream=stderr)

        if self.debug:
            plt=images.multiview(self.psf,levels=7, show=False)
            plt.title='PSF image'
            plt.show()

    def make_gauss_psf(self):
        pars=self.psfpars
        pars['cov'] = array(pars['cov'])
        cov = pars['cov']

        T = 2*max(cov)
        # getdimcen returns length 2 arrays
        dims,cen = self.getdimcen(T)

        self.psf = pixmodel.model_image('gauss',dims,cen,cov,
                                        nsub=self.image_nsub)

        pars['dims'] = dims
        pars['cen'] = cen

        if self.conv == 'func':
            self.psf_func=analytic.Gauss(cov)
       
    def make_dgauss_psf(self):

        pars=self.psfpars
        pars['cov1'] = array(pars['cov1'])
        pars['cov2'] = array(pars['cov2'])

        cenrat = pars['cenrat']
        cov1 = pars['cov1']
        cov2 = pars['cov2']

        Tmax = 2*max( max(cov1), max(cov2) )

        dims,cen = self.getdimcen(Tmax)

        self.psf = pixmodel.double_gauss(dims,cen,cenrat,cov1,cov2,
                                         nsub=self.image_nsub)
        det1=conversions.cov2det(cov1)
        det2=conversions.cov2det(cov2)
        pars['s2'] = sqrt(det2/det1)

        pars['dims'] = dims
        pars['cen'] = cen

        if self.conv == 'func':
            self.psf_func=analytic.DoubleGauss(cenrat,cov1,cov2)


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
        pars['cov'] = array(pars['cov'])
        cov = pars['cov']

        if objmodel == 'gauss':
            # could probably be 3.5
            nsig = 4.5
        elif objmodel == 'exp':
            nsig = 7.0
        else:
            raise ValueError("Unsupported obj model: '%s'" % objmodel)

        # expected size
        Irr,Irc,Icc = cov

        # don't use admom here, we want the unweighted size!
        Irr_expect = Irr + psfpars['cov_meas'][0]
        Icc_expect = Icc + psfpars['cov_meas'][2]

        Texpect = 2*max(Irr_expect,Icc_expect)
        dims, cen = self.getdimcen(Texpect, nsig=nsig)

        if dims[0] < psfpars['dims'][0] and old == False:
            dims = psfpars['dims']
            cen = psfpars['cen']

        wlog("  dims:",dims,"cen:",cen)

        self.image0 = pixmodel.model_image(objmodel,dims,cen,cov,
                                           counts=pars['counts'], 
                                           nsub=self.image_nsub)

        pars['dims'] = dims
        pars['cen'] = cen
        cov_meas = stat.second_moments(self.image0, cen)
        res = admom.admom(self.image0, cen[0],cen[1], guess=(cov_meas[0]+cov_meas[1])/2 )
        cov_meas_admom = array([res['Irr'],res['Irc'],res['Icc']])
        pars['cov_meas'] = cov_meas
        pars['cov_meas_admom'] = cov_meas_admom
        self['cov_image0'] = cov_meas
        self['cov_image0_admom'] = cov_meas_admom

        if self.conv == 'func':
            # turns out we can use very low precision..
            if objmodel == 'exp':
                self.obj_func=analytic.Exp(cov)
            elif objmodel == 'gauss':
                self.obj_func=analytic.Gauss(cov)
            else:
                raise ValueError("only support objmodel gauss or exp for "
                                 "slow convolution")

        if self.verbose:
            wlog("image0 pars")
            pprint(self.objpars,stream=stderr)

        if self.debug:
            plt=images.multiview(self.image0,levels=7, show=False)
            plt.title='image0'
            plt.show()



    def make_image(self):

        pars=self.objpars
        psfpars=self.psfpars

        if self.verbose:
            wlog("Convolving to final image")

        dims = pars['dims']
        cen = pars['cen']
        image0 = self.image0
        psf = self.psf

        # we don't have to do a numerical convolution if both are gaussians
        bothgauss=False
        psfmodel = psfpars['model']
        objmodel = pars['model']
        if (objmodel == 'gauss' and psfmodel == 'gauss') and not self.forcegauss:
            wlog("doing analytic convolution of guassians for gauss")
            bothgauss=True
            ocov=pars['cov']
            pcov=psfpars['cov']
            cov = [ocov[0]+pcov[0],
                     ocov[1]+pcov[1],
                     ocov[2]+pcov[2]]

            image = pixmodel.model_image('gauss',dims,cen,cov,
                                       counts=pars['counts'], nsub=self.image_nsub)
        elif (objmodel == 'gauss' and psfmodel == 'dgauss') and not self.forcegauss:
            wlog("doing analytic convolution of guassians for dgauss")
            ocov=pars['cov']
            pcov1=psfpars['cov1']
            cov1 = [ocov[0]+pcov1[0],
                      ocov[1]+pcov1[1],
                      ocov[2]+pcov1[2]]

            im1 = pixmodel.model_image('gauss',dims,cen,cov1,
                                       counts=pars['counts'], nsub=self.image_nsub)

            pcov2=psfpars['cov2']
            cov2 = [ocov[0]+pcov2[0],
                      ocov[1]+pcov2[1],
                      ocov[2]+pcov2[2]]

            im2 = pixmodel.model_image('gauss',dims,cen,cov2,
                                       counts=pars['counts'], nsub=self.image_nsub)

            s2 = psfpars['s2']
            b = psfpars['cenrat']
            #image = im1 + b*s2*im2
            #image /= (1+b*s2)
            image = im1 + b*im2
            image /= (1+b)

        else:
            if not have_scipy:
                raise ImportError("Could not import scipy")

            if self.conv == 'fft':
                image = self.make_image_fft()
            elif self.conv == 'fconvint':
                if objmodel != 'exp' or psfmodel not in ['gauss','dgauss']:
                    raise ValueError('fconvfunc only works on exp-gauss or exp-dgauss')

                wlog("running fconvint")
                if psfmodel == 'gauss':
                    image = fconv.conv_exp_gauss(dims,cen,pars['cov'], psfpars['cov'],
                                                nsub=self.fconvint_nsub)
                elif psfmodel == 'dgauss':
                    s2 = psfpars['s2']
                    b = psfpars['cenrat']
                    im1= fconv.conv_exp_gauss(dims,cen,pars['cov'], psfpars['cov1'],
                                              nsub=self.fconvint_nsub)
                    im2= fconv.conv_exp_gauss(dims,cen,pars['cov'], psfpars['cov2'],
                                              nsub=self.fconvint_nsub)

                    #image = im1 + b*s2*im2
                    #image /= (1+b*s2)
                    image = im1 + b*im2
                    image /= (1+b)

            elif self.conv == 'fconv':
                wlog("running fconv")
                if psfmodel == 'gauss':
                    image = fconv.gaussconv(self.image0, psfpars['cov'])
                elif psfmodel == 'dgauss':
                    s2 = psfpars['s2']
                    b = psfpars['cenrat']
                    im1 = fconv.gaussconv(self.image0, psfpars['cov1'])
                    im2 = fconv.gaussconv(self.image0, psfpars['cov2'])

                    #image = im1 + b*s2*im2
                    #image /= (1+b*s2)
                    image = im1 + b*im2
                    image /= (1+b)

            elif self.conv == 'func': 
                psf_func=self.psf_func
                obj_func=self.obj_func
                orange = obj_func.range()
                prange = psf_func.range()
                #rng = min(orange[1],prange[1])
                rng = 2*max(orange[1],prange[1])
                intrange = (-rng,rng)
                tdim = int( numpy.ceil( 2*sqrt( orange[1]**2 + prange[1]**2 ) ) )
                wlog("intrange:",intrange,"needed dim:",tdim)
                c = FuncConvolver(obj_func, psf_func, intrange, epsrel=self.eps,epsabs=self.eps)
                image = c.make_image(dims, cen)
                image *= (pars['counts']/image.sum())

        self.image = image
        self['dims'] = dims
        self['cen'] = cen
        if bothgauss:
            cov_meas = array(pars['cov']) + array(psfpars['cov'])
            cov_meas_admom = cov_meas
        else:
            cov_meas = stat.second_moments(self.image, cen)
            res = admom.admom(self.image, cen[0],cen[1], guess=(cov_meas[0]+cov_meas[2])/2)
            cov_meas_admom = array([res['Irr'],res['Irc'],res['Icc']])

        self['cov'] = cov_meas
        self['cov_admom'] = cov_meas_admom

        if self.verbose:
            wlog("convolved image pars")
            pprint(self,stream=stderr)

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
        cov = pars['cov']*fft_nsub**2
        
        psfmodel = ppars['model']
        psfdims = ppars['dims']*fft_nsub
        psfcen_orig = ppars['cen']

        # this is an array operation
        cen = (dims-1.)/2.

        image0_boosted = pixmodel.model_image(objmodel,dims,cen,cov,
                                              counts=pars['counts'], 
                                              nsub=self.image_nsub)

        psfcen = (psfdims-1.)/2.
        wlog("running fft convolve")
        if psfmodel == 'gauss':
            psfcov = ppars['cov']*fft_nsub**2
            psf_boosted = pixmodel.model_image('gauss',psfdims,psfcen,psfcov, 
                                               nsub=self.image_nsub)

            wlog("  running fftconvolve")
            image_boosted = scipy.signal.fftconvolve(image0_boosted, psf_boosted, 
                                                     mode='same')
        elif psfmodel == 'dgauss':
            b = ppars['cenrat']
            s2 = ppars['s2']
            cov1 = ppars['cov1']*fft_nsub**2
            cov2 = ppars['cov2']*fft_nsub**2
            g1 = pixmodel.model_image('gauss',psfdims,psfcen,cov1, 
                                      nsub=self.image_nsub)
            g2 = pixmodel.model_image('gauss',psfdims,psfcen,cov2, 
                                      nsub=self.image_nsub)

            wlog("  running fftconvolve1")
            im1 = scipy.signal.fftconvolve(image0_boosted, g1, mode='same')
            wlog("  running fftconvolve2")
            im2 = scipy.signal.fftconvolve(image0_boosted, g2, mode='same')

            #image_boosted = im1 + b*s2*im2
            #image_boosted /= (1+b*s2)
            image_boosted = im1 + b*im2
            image_boosted /= (1+b)
        else:
            raise ValueError("unsupported psf model: '%s'" % psfmodel)

        if fft_nsub > 1:
            image = rebin(image_boosted, fft_nsub)
        else:
            image = image_boosted

        # make sure the center didn't shift
        mom0 = stat.moments(self.image0)
        mom = stat.moments(image)

        if self.verbose:
            wlog("  psf dims:",self.psf.shape)
            wlog("  psf cen: ",self.psfpars['cen'])
            wlog("  image0 dims:",self.image0.shape)
            wlog("  image0 cen:",self.objpars['cen'])
            wlog("  boosted image0 dims:",image0_boosted.shape)
            wlog("  boosted image0 cen:",cen)

            wlog("  fft created image dims:",image.shape)
            wlog("  fft created image cen:",mom['cen'])

        max_shift = max( abs(mom['cen'][0]-mom0['cen'][0]), abs(mom['cen'][1]-mom0['cen'][1]) )
        if (max_shift/mom['cen'][0]-1) > 0.0033:
            raise ValueError("Center rel shifted greater than 0.0033: %f" % max_shift)

        return image

    def mom2sigma(self, T):
        return sqrt(T/2)

    def getdimcen(self, T, nsig=4.5):
        sigma = sqrt(T/2)
        imsize = int( numpy.ceil(2*nsig*sigma) )

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
              'cov1':array([1.23**2, 0.0, 1.23**2]),
              'cov2':array([2.9**2, 0.0, 2.9**2])}

    objpars1={'model':'gauss',
              'cov':array([1.5**2, 0.0, 0.5**2])}

    n=20
    sizefac_vals = numpy.linspace(1.1,20.0)

    e1=numpy.zeros(n,dtype='f4')
    e2=numpy.zeros(n,dtype='f4')
    s=numpy.zeros(n,dtype='f4')

    for i in xrange(n):
        sizefac = sizefac_vals[i]
        sizefac2 = sizefac**2

        wlog("sizefac:",sizefac)

        psfpars = copy.deepcopy( psfpars1 )

        psfpars['cov1'] *= sizefac2
        psfpars['cov2'] *= sizefac2

        objpars = copy.deepcopy( objpars1 )

        objpars['cov'] *= sizefac2

        ci = ConvolvedImage(objpars,psfpars, conv=conv)

        T = ci['cov'][0] + ci['cov'][2]

        e1t = (ci['cov'][2]-ci['cov'][0])/T
        e2t =  2*ci['cov'][1]/T

        e1[i] = e1t
        e2[i] = e2t
        det = conversions.cov2det(ci['cov'])
        s[i] = det**0.25
        wlog("s:",s[i])


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
                    wlog("(%d,%d)" % (row,col))
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


    wlog("range1:",r1)
    wlog("range2:",r2)
    wlog("intrange:",intrange)
    wlog("dims:",dims)
    wlog("cen:",cen)
    wlog("Irr expected:",Irr1+Irr2)
    wlog("Irc expected:",Irc1+Irc2)
    wlog("Icc expected:",Icc1+Icc2)


    c = FuncConvolver(g1,g2, intrange, epsabs=epsabs,epsrel=epsrel)

    imfast = c.make_image_fast(dims, cen)
    Irrfast,Ircfast,Iccfast = stat.second_moments(imfast,cen)
    wlog("Irr fast:",Irrfast)
    wlog("Irc fast:",Ircfast)
    wlog("Icc fast:",Iccfast)

    im = c.make_image(dims, cen)

    Irracc,Ircacc,Iccacc = stat.second_moments(im,cen)


    wlog("Irr acc:",Irracc)
    wlog("Irc acc:",Ircacc)
    wlog("Icc acc:",Iccacc)


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


    wlog("dgrange:",dgrange)
    wlog("erange:",erange)
    wlog("intrange:",intrange)
    wlog("dims:",dims)
    wlog("cen:",cen)


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


def test_conv_exp_gauss():
    '''
    Generate hires objects, convolve them with fft, and
    compare to the conv_exp_gauss.  

    Then rebin the hires and compare to conv_exp_gauss created
    at the lower res
    '''

    fac=5

    cov=array([2.0,0.5,1.0])
    psf_cov = array([1.0,0.0,1.0])

    objpars_hires={'model':'exp',   'cov':cov*fac**2}
    psfpars_hires={'model':'gauss', 'cov':psf_cov*fac**2}

    ci_fft = ConvolvedImage(objpars_hires, psfpars_hires, conv='fft')
    ci_int = ConvolvedImage(objpars_hires, psfpars_hires, conv='fconvint', fconvint_nsub=1)

    images.compare_images(ci_fft.image, ci_int.image, label1='fft', label2='int')



def example_pars(objmodel='exp', psfmodel='gauss', s2=1.0, ellip=0.3, theta=10):
    if psfmodel == 'gauss':
        psf_sigma=1.4
        cov=[psf_sigma**2, 0.0, psf_sigma**2]
        psfpars={'model':psfmodel,'cov':cov}
    elif psfmodel == 'dgauss':
        sigma=1.4
        sigrat=2.3
        cenrat=0.09

        cov1=array([sigma**2, 0.0, sigma**2])
        cov2=cov1*sigrat**2
        psfpars={'model':psfmodel,'cov1':cov1,'cov2':cov2,
                 'cenrat':cenrat}

        psum = 1+cenrat
        cov11 = (cov1[0] + cov2[0]*cenrat)/psum
        cov22 = (cov1[2] + cov2[2]*cenrat)/psum
        psf_sigma = sqrt( (cov11+cov22)/2)
    elif psfmodel=='turb':
        fwhm=3.3
        psfpars={'model':psfmodel,'psf_fwhm':fwhm}
        psf_sigma = fwhm/1.4
    else:
        raise ValueError("unknown psf model '%s'" % psfmodel)

    # very approx
    obj_sigma = psf_sigma/sqrt(s2)
    cov=conversions.ellip2mom(2*obj_sigma**2,e=ellip,theta=theta)
    objpars = {'model':objmodel,'cov':cov}

    return objpars, psfpars

