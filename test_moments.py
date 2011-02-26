from __future__ import print_function
import numpy
from numpy import arange, array, ceil, zeros, sqrt, log10

from .pixmodel import model_image
from .stat import second_moments

class MomentTester:
    '''
    Test if we recover the input ellipticity and size.  
    
        t=MomentTester(model)
        t.run()
        t.plot()
    
    Since this does no sub-sampling, using it in the model_image will mean this
    get's it wrong. Thus it is turned of in run by default

    This is also a test of the fake image making tools...

    Good sampling:
        I find for good sampling, sigfac=7 for exp and 3.5
        for gauss work well.

        exp:
            6 gives 0.1%
            6.5 gives 0.03% 
            7 gives 0.01%  in e and moments
            7.5 no improvement for some reason
            8.0 gives 2.e-5

        gauss:
            gauss, 3.5 gives 0.01%...

    Poor sample:
        I find one can get huge errors in measurement of ellip and even
        *bigger* for size with poor resolution.


    '''

    def __init__(self, model):
        self.model=model

    def run(self, nsub=1, sigfac=None):


        #facvals = arange(1,21)
        facvals = numpy.logspace(log10(1), log10(20.0), 20)

        cov0=array([1.4, 0.0, 0.7])

        Ttrue=cov0[0] + cov0[2]
        e1true = (cov0[2]-cov0[0])/Ttrue
        e2true = 2*cov0[1]/Ttrue
        etrue = sqrt(e1true**2 + e2true**2)

        if sigfac is None:
            if self.model == 'exp':
                sigfac=7.
            else:
                sigfac=3.5

        dim = int( ceil(2*sigfac*max(cov0)))
        if (dim % 2) == 0:
            dim+=1

        dims0=array([dim]*2)

        e1=zeros(facvals.size, dtype='f4')
        e2=zeros(facvals.size, dtype='f4')
        sigma=zeros(facvals.size, dtype='f4')
        T=zeros(facvals.size, dtype='f4')

        for i in xrange(facvals.size):
            fac=facvals[i]
            print("fac:",fac)
            dims = dims0*fac
            cov = cov0*fac**2

            cen = [(dims[0]-1)/2.]*2

            im = model_image(self.model, dims, cen, cov, nsub=nsub)

            covmeas = second_moments(im)
            Tmeas = covmeas[0]+covmeas[2]
            e1[i] = (covmeas[2]-covmeas[0])/Tmeas
            e2[i] = 2*covmeas[1]/Tmeas
            T[i] = Tmeas

            # note factor or 3 for exp
            sigma[i] = sqrt( (cov[0] + cov[2])/2. )

            print(dims)

        
        e = sqrt(e1**2 + e2**2)

        res={'cov':cov0,
             'Ttrue':Ttrue,
             'e1true':e1true,
             'e2true':e2true,
             'etrue':etrue,
             'facvals':facvals,
             'sigma':sigma,
             'T':T,
             'e1':e1,
             'e2':e2,
             'e':e}

        self.res=res

    def plot(self, yrange=None):
        from biggles import FramedPlot, Points, Curve, PlotKey, Table,PlotLabel

        res=self.res

        etrue = res['etrue']
        e1true = res['e1true']
        e2true = res['e2true']

        facvals = res['facvals']
        sigma = res['sigma']
        e1=res['e1']
        e2=res['e2']
        e=res['e']

        T=res['T']
        Ttrue=res['Ttrue']*facvals**2


        e_pdiff = e/etrue-1
        T_pdiff = T/Ttrue-1
        

        compc= Curve(sigma, sigma*0)

        ce = Curve(sigma, e_pdiff, color='red')
        pe = Points(sigma, e_pdiff, type='filled circle')

        e_plt = FramedPlot()
        e_plt.add(ce,pe,compc)
        e_plt.xlabel = r'$\sigma$'
        e_plt.ylabel = r'$e/e_{true}-1$'

        cT = Curve(sigma, T_pdiff, color='red')
        pT = Points(sigma, T_pdiff, type='filled circle')

        T_plt = FramedPlot()
        T_plt.add(cT,pT,compc)
        T_plt.xlabel = r'$\sigma$'
        T_plt.ylabel = r'$T/T_{true}-1$'

        lab=PlotLabel(0.9,0.9,self.model,halign='right')
        T_plt.add(lab)

        if yrange is not None:
            e_plt.yrange=yrange
            T_plt.yrange=yrange
        
        tab=Table(2,1)

        tab[0,0] = T_plt
        tab[1,0] = e_plt


        tab.show()
