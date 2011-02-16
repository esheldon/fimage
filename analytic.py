import numpy
from numpy import sqrt, exp

class Gauss:
    def __init__(self, covar):

        Irr,Irc,Icc = covar

        self.det = Irr*Icc - Irc**2
        if self.det == 0:
            raise RuntimeError("Determinant is zero")

        self.Irr=Irr
        self.Irc=Irc
        self.Icc=Icc

        self.Wrr = self.Irr/self.det
        self.Wrc = self.Irc/self.det
        self.Wcc = self.Icc/self.det

    def range(self):
        '''
        Range of values if you were to integrate the function.

        Exponentials are more extended
        use [-2*7*sigma, 2*7*sigma]
        '''
        sigma = sqrt( max( self.Irr, self.Icc) )
        rng = 4.5*sigma
        return (-rng,rng)

    def __call__(self, row, col):

        rr = row**2*self.Wcc -2*row*col*self.Wrc + col**2*self.Wrr
        return exp(-0.5*rr)



class Exp:
    def __init__(self, covar):

        Irr,Irc,Icc = covar
        self.det = Irr*Icc - Irc**2
        if self.det == 0:
            raise RuntimeError("Determinant is zero")

        self.Irr=Irr
        self.Irc=Irc
        self.Icc=Icc

        self.Wrr = self.Irr/self.det
        self.Wrc = self.Irc/self.det
        self.Wcc = self.Icc/self.det

    def range(self):
        '''
        Range of values if you were to integrate the function.

        Exponentials are more extended
        use [-2*7*sigma, 2*7*sigma]
        '''
        sigma = sqrt( max( self.Irr, self.Icc) )
        rng = 7*sigma
        return (-rng,rng)

    def __call__(self, row, col):

        rr = row**2*self.Wcc -2*row*col*self.Wrc + col**2*self.Wrr
        r = sqrt(rr)
        return exp(-r)

class DoubleGauss:
    def __init__(self, 
                 cenrat,
                 covar1,
                 covar2):

        Irr1,Irc1,Icc1 = covar1
        Irr2,Irc2,Icc2 = covar2

        self.det1 = Irr1*Icc1-Irc1**2
        if self.det1 == 0:
            raise RuntimeError("Determinant of gauss1 is zero")
        self.det2 = Irr2*Icc2-Irc2**2
        if self.det2 == 0:
            raise RuntimeError("Determinant of gauss2 is zero")

        self.det1sqrtinv = 1/sqrt( self.det1 )
        self.det2sqrtinv = 1/sqrt( self.det2 )

        self.b = cenrat
        self.s2 = sqrt( self.det2/self.det1 )
        self.fac = self.b*self.s2

        self.Irr1=Irr1
        self.Irc1=Irc1
        self.Icc1=Icc1

        self.Irr2=Irr2
        self.Irc2=Irc2
        self.Icc2=Icc2

        self.Wrr1=Irr1/self.det1
        self.Wrc1=Irc1/self.det1
        self.Wcc1=Icc1/self.det1

        self.Wrr2=Irr2/self.det2
        self.Wrc2=Irc2/self.det2
        self.Wcc2=Icc2/self.det2

    def range(self):
        '''
        Range of values if you were to integrate the function.

        use [-2*4.5*sigma, 2*4.5*sigma]
        '''

        sigma1 = sqrt( max( self.Irr1, self.Icc1) )
        sigma2 = sqrt( max( self.Irr2, self.Icc2) )
        sigma = max(sigma1,sigma2)
        rng = 4.5*sigma
        return (-rng,rng)


    def __call__(self, row, col):
        row2=row*row
        col2=col*col
        rowcoltimes2=2*row*col

        rr1 = row2*self.Wcc1 -rowcoltimes2*self.Wrc1 + col2*self.Wrr1
        rr2 = row2*self.Wcc2 -rowcoltimes2*self.Wrc2 + col2*self.Wrr2

        g1=exp(-0.5*rr1)*self.det1sqrtinv
        g2=exp(-0.5*rr2)*self.det2sqrtinv

        return g1 + self.fac*g2
