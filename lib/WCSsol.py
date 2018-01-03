import numpy as np
from scipy.optimize import minimize
import copy

class WCSsol(object):

    def __init__(self, RADECcat, WCS, CDscale = 1.):  # CDscale is a factor that can help with the numerical solution
        
        self.RADECcat = RADECcat
        self.WCS = WCS
        self.CDscale = CDscale #1e-5 #0.27 / 60. / 60. / 4.
        
    def residuals(self):
        return np.array([self.WCS.RA() - self.RADECcat.x[self.RADECcat.sidx], self.WCS.DEC() - self.RADECcat.y[self.RADECcat.sidx]])

    def chi2(self):
        return np.sum(self.residuals()**2)

    def dchi2dCRVAL1(self):
        return 2. * np.sum(self.residuals() * np.vstack([self.WCS.dRAdCRVAL1(), self.WCS.dDECdCRVAL1()]))

    def dchi2dCRVAL2(self):
        return 2. * np.sum(self.residuals() * np.vstack([self.WCS.dRAdCRVAL2(), self.WCS.dDECdCRVAL2()]))

    def dchi2dCRPIX1(self):
        return 2. * np.sum(self.residuals() * np.vstack([self.WCS.dRAdCRPIX1(), self.WCS.dDECdCRPIX1()]))

    def dchi2dCRPIX2(self):
        return 2. * np.sum(self.residuals() * np.vstack([self.WCS.dRAdCRPIX2(), self.WCS.dDECdCRPIX2()]))

    def dchi2dCD11(self):
        return 2. * np.sum(self.residuals() * np.vstack([self.WCS.dRAdCD11(), self.WCS.dDECdCD11()]))

    def dchi2dCD12(self):
        return 2. * np.sum(self.residuals() * np.vstack([self.WCS.dRAdCD12(), self.WCS.dDECdCD12()]))

    def dchi2dCD21(self):
        return 2. * np.sum(self.residuals() * np.vstack([self.WCS.dRAdCD21(), self.WCS.dDECdCD21()]))

    def dchi2dCD22(self):
        return 2. * np.sum(self.residuals() * np.vstack([self.WCS.dRAdCD22(), self.WCS.dDECdCD22()]))

    # jacobian of scalar chi2 funcion  --------------------

    # chi2 as a function of CRPIX CRVAL parameters
    def chi2CRpar(self, x):
        WCS2 = copy.copy(self.WCS)
        WCS2.CRPIX1 = x[0]
        WCS2.CRPIX2 = x[1]
        WCS2.CRVAL1 = x[2]
        WCS2.CRVAL2 = x[3]
        saux = WCSsol(self.RADECcat, WCS2)
        chi2 = saux.chi2()
        del WCS2, saux

        return chi2

    # chi2 as a function of CD parameters
    def chi2CDpar(self, x):
        WCS2 = copy.copy(self.WCS)
        WCS2.CD11 = self.CDscale * x[0]
        WCS2.CD12 = self.CDscale * x[1]
        WCS2.CD21 = self.CDscale * x[2]
        WCS2.CD22 = self.CDscale * x[3]
        saux = WCSsol(self.RADECcat, WCS2)
        chi2 = saux.chi2()
        del WCS2, saux

        return chi2

    # chi2 as a function of all parameters
    def chi2par(self, x):
        WCS2 = copy.copy(self.WCS)
        WCS2.CRPIX1 = x[0]
        WCS2.CRPIX2 = x[1]
        WCS2.CRVAL1 = x[2]
        WCS2.CRVAL2 = x[3]
        WCS2.CD11 = self.CDscale * x[4]
        WCS2.CD12 = self.CDscale * x[5]
        WCS2.CD21 = self.CDscale * x[6]
        WCS2.CD22 = self.CDscale * x[7]
        saux = WCSsol(self.RADECcat, WCS2)
        chi2 = saux.chi2()
        del WCS2, saux

        return chi2

    # chi2 jacobian w.r.t CR parameters
    def chi2CRjac(self, x):
        WCS2 = copy.copy(self.WCS)
        WCS2.CRPIX1 = x[0]
        WCS2.CRPIX2 = x[1]
        WCS2.CRVAL1 = x[2]
        WCS2.CRVAL2 = x[3]
        WCS2.CD11 = self.CDscale * x[4]
        WCS2.CD12 = self.CDscale * x[5]
        WCS2.CD21 = self.CDscale * x[6]
        WCS2.CD22 = self.CDscale * x[7]
        saux = WCSsol(self.RADECcat, WCS2)
        jac = np.array([saux.dchi2dCRPIX1(), saux.dchi2dCRPIX2(), saux.dchi2dCRVAL1(), saux.dchi2dCRVAL2()])
        del WCS2, saux

        return jac

    # chi2 jacobian w.r.t CD parameters
    def chi2CDjac(self, x):
        WCS2 = copy.copy(self.WCS)
        WCS2.CD11 = self.CDscale * x[0]
        WCS2.CD12 = self.CDscale * x[1]
        WCS2.CD21 = self.CDscale * x[2]
        WCS2.CD22 = self.CDscale * x[3]
        saux = WCSsol(self.RADECcat, WCS2)
        jac = np.array([saux.dchi2dCD11(), saux.dchi2dCD12(), saux.dchi2dCD21(), saux.dchi2dCD22()])
        del WCS2, saux

        return jac

    # chi2 jacobian w.r.t all parameters
    def chi2jac(self, x):
        WCS2 = copy.copy(self.WCS)
        WCS2.CRPIX1 = x[0]
        WCS2.CRPIX2 = x[1]
        WCS2.CRVAL1 = x[2]
        WCS2.CRVAL2 = x[3]
        WCS2.CD11 = self.CDscale * x[4]
        WCS2.CD12 = self.CDscale * x[5]
        WCS2.CD21 = self.CDscale * x[6]
        WCS2.CD22 = self.CDscale * x[7]
        saux = WCSsol(self.RADECcat, WCS2)
        jac = np.array([saux.dchi2dCRPIX1(), saux.dchi2dCRPIX2(), saux.dchi2dCRVAL1(), saux.dchi2dCRVAL2(), self.CDscale * saux.dchi2dCD11(), self.CDscale * saux.dchi2dCD12(), self.CDscale * saux.dchi2dCD21(), self.CDscale * saux.dchi2dCD22()])
        del WCS2, saux

        return jac


    # test jacobian
    def testjacobian(self, delta = 1e-9):
          
        print "\nTesting Jacobian"
        testvars = ["CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CD11", "CD12", "CD21", "CD22"]
        for var in testvars:
            WCS2 = copy.deepcopy(self.WCS)
            exec("WCS2.%s = WCS2.%s * (1. + delta)" % (var, var))
            exec("dvar = WCS2.%s - self.WCS.%s" % (var, var))
            sol2 = WCSsol(self.RADECcat, WCS2)
            exec("print \" dchi2/d%s\", (sol2.chi2() - self.chi2()) / dvar / self.dchi2d%s()" % (var, var))
        print
        
    # find best WCS given pixel and RADEC catalogues
    def solve(self):
    
        x0 = np.array([self.WCS.CRPIX1, self.WCS.CRPIX2, self.WCS.CRVAL1, self.WCS.CRVAL2, self.WCS.CD11 / self.CDscale, self.WCS.CD12 / self.CDscale, self.WCS.CD21 / self.CDscale, self.WCS.CD22 / self.CDscale])
        print "   Running minimization routine..."
        gcsol = minimize(self.chi2par, x0, method = 'L-BFGS-B', jac = self.chi2jac)#, options = {'ftol': 1e-15, 'gtol': 1e-15, 'factr': 1e2, 'eps': 1e-10, 'disp': True})
        if gcsol.success:
            print "   Using new WCS solution (chi2: %e)" % gcsol.fun

            # store values in current solution
            (self.WCS.CRPIX1, self.WCS.CRPIX2, self.WCS.CRVAL1, self.WCS.CRVAL2,
             self.WCS.CD11, self.WCS.CD12, self.WCS.CD21, self.WCS.CD22) = gcsol.x
            self.WCS.CD11 = self.WCS.CD11 * self.CDscale
            self.WCS.CD12 = self.WCS.CD12 * self.CDscale
            self.WCS.CD21 = self.WCS.CD21 * self.CDscale
            self.WCS.CD22 = self.WCS.CD22 * self.CDscale

            # change best guess parameters of reference catalogue
            self.WCS.ijcat.CRPIXguess = np.array([self.WCS.CRPIX1, self.WCS.CRPIX2])
            self.WCS.ijcat.CRVALguess = np.array([self.WCS.CRVAL1, self.WCS.CRVAL2])
            self.WCS.ijcat.CDguess = np.array([[self.WCS.CD11, self.WCS.CD12], [self.WCS.CD21, self.WCS.CD22]])
                        
        else:
            print "FAILURE"
            print gcsol
        

