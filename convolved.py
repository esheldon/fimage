from __future__ import print_function
from pprint import pprint
import numpy
from numpy import sqrt, exp, array, ceil, log2, pi, ogrid, zeros, where
from numpy.fft import fftshift

import images
from sys import stderr

from . import statistics as stat
from . import pixmodel
from . import fconv
from . import analytic
from . import conversions
from .conversions import mom2sigma, cov2sigma

from .noise import add_noise_uw, s2n_andres

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
TURB_SIGMA_FAC=1.68
TURB_PADDING=10.0

GAUSS_PADDING=5.0
EXP_PADDING=7.0
DEV_PADDING=15.0

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

    # padding for PSF
    kdims = dims.copy()
    kdims += 2.*4.*sigma

    # Always use 2**n-sized FFT
    kdims = 2**ceil(log2(kdims))
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

    The images dimensions should be square, and even so the psf is
    centered.
    """

    dims = array(image.shape)
    if dims[0] != dims[1]:
        raise ValueError("only square images for now")

    # add padding for PSF in real space
    # sigma is approximate
    kdims=dims.copy()
    kdims += 2*4*fwhm/TURB_SIGMA_FAC

    # Always use 2**n-sized FFT
    kdims = 2**ceil(log2(kdims))
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

        if self.objpars['model'] not in ['gauss','exp','dev']:
            raise ValueError("only support gauss/exp/dev objects")
        if self.psfpars['model'] not in ['gauss','dgauss','turb']:
            raise ValueError("only support gauss/dgauss/turb psf")

        if 'cov' in self.objpars:
            self.objpars['cov'] = array(self.objpars['cov'],dtype='f8')
        if 'cov' in self.psfpars:
            self.psfpars['cov'] = array(self.psfpars['cov'],dtype='f8')
 
        # we want to try to let the expansion factor do the trick
        # even for dev, we probably don't need a full nsub=16 here
        defsub=1
        self['image_nsub'] = keys.get('image_nsub', defsub)
        self['expand_fac_min'] = keys.get('expand_fac_min', 1)
        stderr.write("image_nsub: %d " % self['image_nsub'])

        # for calculations we will demand sigma > minres pixels
        # then sample back
        if self.objpars['model'] == 'dev':
            #minres_def = 24
            minres_def = 20
        else:
            minres_def = 12
        self['minres'] = keys.get('minres',minres_def)

        self.set_default_padding(**keys)
        self.image0=None
        self.image=None
        self.psf=None

    def set_default_padding(self, **keys):
        if 'psffac' in keys:
            self['psffac'] = keys['psffac']
        if 'objfac' in keys:
            self['objfac'] = keys['objfac']
        else:
            # Hint for how large to make the base image
            if self.objpars['model'] == 'exp':
                self['objfac']=EXP_PADDING
            elif self.objpars['model'] == 'dev':
                self['objfac']=DEV_PADDING
            else:
                self['objfac']=GAUSS_PADDING

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

        e1_uw = (cov_uw[2]-cov_uw[0])/(cov_uw[2]+cov_uw[0])
        e2_uw = 2*cov_uw[1]/(cov_uw[2]+cov_uw[0])
        self['e1_image0_uw'] = e1_uw
        self['e2_image0_uw'] = e2_uw
        self['e_image0_uw'] = sqrt(e1_uw**2 + e2_uw**2)

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
        self['a4_psf'] = res['a4']

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
        self['a4'] = res['a4']

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

    def write_fits(self, fits_file, extra_keys=None):
        """
        Write the images and metadata to a fits file

        The images are in separate extensions 'image','psf','image0' and the
        metadata are in a binary table 'table'

        parameters
        ----------
        fits_file: string
            Name of the file to write
        ci: child of ConvolverBase
        """
        import fitsio
        dt=[]
        for k,v in self.iteritems():
            if isinstance(v,int) or isinstance(v,long):
                dt.append( (k, 'i8') )
            elif isinstance(v,float):
                dt.append( (k, 'f8') )
            elif isinstance(v,numpy.ndarray):
                this_t = v.dtype.descr[0][1]
                this_n = v.size
                if this_n > 1:
                    this_dt = (k,this_t,this_n)
                else:
                    this_dt = (k,this_t)
                dt.append(this_dt)
            else:
                raise ValueError("unsupported type: %s" % type(v))
        table = numpy.zeros(1, dtype=dt)
        for k,v in self.iteritems():
            table[k][0] = v

        with fitsio.FITS(fits_file,mode='rw',clobber=True) as fitsobj:
            h={}
            # note not all items will be written, only basic types,
            # so this is not for feeding to the sim code.  The full
            # metadata are in the table
            for k,v in self.iteritems():
                h[k] = v
            if extra_keys:
                for k,v in extra_keys.iteritems():
                    h[k] = v

            fitsobj.write(self.image, header=h, extname='image')
            fitsobj.write(self.psf, extname='psf')
            fitsobj.write(self.image0, extname='image0')
            fitsobj.write(table, extname='table')


class NoisyConvolvedImage(dict):
    def __init__(self, ci, s2n, s2n_psf):
        self.ci = ci
        self.image = ci.image
        self.image0 = ci.image0
        self.psf = ci.psf

        for k,v in ci.iteritems():
            self[k] = v

        if s2n > 0:
            self.image_nonoise = ci.image
            self.image, self['skysig'] = add_noise_uw(ci.image, s2n)
            self['s2n_andres'] = s2n_andres(ci.image, self['skysig'])
        if s2n_psf > 0: 
            self.psf_nonoise = ci.psf
            self.psf, self['skysig_psf'] = add_noise_uw(ci.psf, s2n_psf)
            self['s2n_andres_psf'] = s2n_andres(ci.psf, self['skysig_psf'])

class TrimmedConvolvedImage(ConvolverBase):
    def __init__(self, ci, fluxfrac=0.999937):
        """
        "4-sigma" corresponds to 0.999937 of the flux
        """
        for k,v in ci.iteritems():
            self[k] = v

        self.ci = ci
        self.objpars=ci.objpars
        self.psfpars=ci.psfpars
        self.fluxfrac=fluxfrac

        self.trim()
        self.add_image0_stats()
        self.add_psf_stats()
        self.add_image_stats()

        self['cen'] = self['cen_uw']

    def trim(self):

        im = self.ci.image
        row,col=ogrid[0:im.shape[0], 
                      0:im.shape[1]]
        rm = array(row - self.ci['cen'][0], dtype='f8')
        cm = array(col - self.ci['cen'][1], dtype='f8')
        radm = sqrt(rm**2 + cm**2)

        radii = numpy.arange(1,im.shape[0]/2)
        cnts=numpy.zeros(radii.size)
        for ir,r in enumerate(radii):
            w=where(radm <= r)
            if w[0].size > 0:
                cnts[ir] = im[w].sum()

        cnts /= cnts.max()

        w,=where(cnts > self.fluxfrac)
        if w.size > 0:
            rad = radii[w[0]]
            rmin = self.ci['cen'][0]-rad
            rmax = self.ci['cen'][0]+rad
            cmin = self.ci['cen'][1]-rad
            cmax = self.ci['cen'][1]+rad

            if rmin < 0:
                rmin=0
            if rmax > (im.shape[0]-1):
                rmax = (im.shape[0]-1)

            if cmin < 0:
                cmin=0
            if cmax > (im.shape[1]-1):
                cmax = (im.shape[1]-1)

            self.image = self.ci.image[rmin:rmax, cmin:cmax]
            self.image0 = self.ci.image0[rmin:rmax, cmin:cmax]
            self.psf = self.ci.psf[rmin:rmax, cmin:cmax]

        else:
            raise ValueError("no radii found, that might be a bug!")
            self.image = self.ci.image[rmin:rmax, cmin:cmax]
            self.image0 = self.ci.image0[rmin:rmax, cmin:cmax]
            self.psf = self.ci.psf[rmin:rmax, cmin:cmax]
             
class ConvolverGaussFFT(ConvolverBase):
    """
    Convolve models with a gaussian or double gaussian psf
    """
    def __init__(self, objpars, psfpars, **keys):
        """
        Gaussian fft with some model
        """
        super(ConvolverGaussFFT,self).__init__(objpars,psfpars,**keys)
            
        if 'psffac' not in self:
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

        psffac = self['psffac']
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

        # assume ellip=0.8.  This will equalize the image
        # sizes for different ellipticities
        sigma_psf = cov2sigma(psf_cov,maxe=0.8)
        sigma_obj = cov2sigma(obj_cov,maxe=0.8)


        imsize = 2*sqrt( psffac**2*sigma_psf**2 + objfac**2*sigma_obj**2)
        dims = array([imsize]*2,dtype='i8')

        self['dims'] = dims
        if (dims[0] % 2) != 0:
            dims += 1

        self['cen'] = (dims-1.)/2.


        #
        # do we need to expand before convolving?
        #

        sigma_obj_min = sqrt(min(obj_cov[0],obj_cov[2]))
        sigma_min = min(sigma_psf,sigma_obj_min)

        fac = 1
        if sigma_min < self['minres']:
            # find the odd integer expansion that will get sigma > minres
            fac = int(self['minres']/sigma_min)

        fac_min = self['expand_fac_min']

        if fac < fac_min:
            fac=fac_min

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

        stderr.write("image_nsub(again): %d expand_fac: %d\n" % (nsub,fac))
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
 
        if 'psffac' not in self:
            self['psffac'] = GAUSS_PADDING
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

        
        fac = self['psffac']

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
        sigma = cov2sigma(cov,maxe=0.8)

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

        if 'psffac' not in self:
            self['psffac'] = TURB_PADDING
        #self['psffac'] = 3.
        if self.objpars['model'] in ['exp','dev']:
            # The broad exponential and dev must feel the outer psf
            self['psffac'] *= 1.5

        # 6 seems good enough.  Can we do this in ConvolverGaussFFT?
        #self['minres'] = 6


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

        # assume ellip=0.8.  This will equalize the image
        # sizes for different ellipticities
        sigma_obj = cov2sigma(imcov,maxe=0.8)

        fwhm = self.psfpars['psf_fwhm']
        imsize = 2*sqrt( (psffac*fwhm)**2 + objfac**2*sigma_obj**2 )
        dims = array([imsize]*2,dtype='i8')

        if (dims[0] % 2) != 0:
            dims += 1
        self['dims'] = dims
        self['cen'] = dims/2


        # the idea of "sigma" has no meaning for this type of psf
        # be conservative and make it seem small, which will result
        # in extra expansion

        sigma_obj_min = sqrt(min(imcov[0],imcov[2]))
        sigma_psf = fwhm/4.
        sigma_min = min(sigma_psf,sigma_obj_min)

        # do we need to expand before convolving?
        fac=1
        if sigma_min < self['minres']:
            # find the expansion that will get sigma > minres
            fac = int(self['minres']/sigma_min)

        fac_min = self['expand_fac_min']
        if fac < fac_min:
            fac=fac_min

        #if fac < 16:
        #    fac=16.
        self['expand_fac'] = fac
        wlog("  expand_fac_min:",fac_min,"expand_fac:",fac)

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




class ConvolvedImageFromFits(dict):
    """
    Read a convolved image stored in fits format.

    The images will be in .image, .psf, .image0.  The metadata will be copied
    into the self dictionary.

    send hid=1 or 2 for ring pairs
    """
    def __init__(self, fitsfile, hid=None):
        self.hid=hid
        self.fitsfile=fitsfile
        self.read_data()

    def read_data(self):
        import fitsio

        imname,psfname,im0name,tablename=self.get_names()
        wlog(imname,psfname,im0name,tablename)
        with fitsio.FITS(self.fitsfile) as fits:
            self.image = fits[imname][:,:]
            self.psf = fits[psfname][:,:]
            self.image0 = fits[im0name][:,:]
            self.table = fits[tablename][:]

            self.header=fits[imname].read_header()
            for k in self.header.keys():
                self[k.lower()] = self.header[k]

            for k in self.table.dtype.names:
                self[k.lower()] = self.table[k][0]

    def get_names(self):
        imname='image'
        psfname='psf'
        im0name='image0'
        tablename='table'

        hid=self.hid
        if hid is not None:
            imname += '%d' % hid
            psfname += '%d' % hid
            im0name = 'image%d_0' % hid
            tablename += '%d' % hid
        return imname,psfname,im0name,tablename
 
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

