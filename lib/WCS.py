import numpy as np

deg2rad = np.pi / 180.
rad2deg = 180. / np.pi

class WCS(object):

    def __init__(self, ijcat, CRPIX, CRVAL, CD, PV):
        
        self.ijcat = ijcat
        self.CRPIX1 = CRPIX[0]
        self.CRPIX2 = CRPIX[1]
        self.CRVAL1 = CRVAL[0]
        self.CRVAL2 = CRVAL[1]
        self.CD11 = CD[0, 0]
        self.CD12 = CD[0, 1]
        self.CD21 = CD[1, 0]
        self.CD22 = CD[1, 1]
        self.PV = PV

    # ---------------------------------------

    def ij2RADEC(self, i, j):
        
        xval = self.CD11 * (i - self.CRPIX1) + self.CD12 * (j - self.CRPIX2) # deg 
        yval = self.CD21 * (i - self.CRPIX1) + self.CD22 * (j - self.CRPIX2) # deg
        rval = np.sqrt(xval**2 + yval**2) # deg
        xival = deg2rad * (self.PV[0, 0] + self.PV[0, 1] * xval + self.PV[0, 2] * yval + self.PV[0, 3] * rval + self.PV[0, 4] * xval**2 + self.PV[0, 5] * xval * yval + self.PV[0, 6] * yval**2 + self.PV[0, 7] * xval**3 + self.PV[0, 8] * xval**2 * yval + self.PV[0, 9] * xval * yval**2 + self.PV[0, 10] * yval**3)
        etaval = deg2rad * (self.PV[1, 0] + self.PV[1, 1] * yval + self.PV[1, 2] * xval + self.PV[1, 3] * rval + self.PV[1, 4] * yval**2 + self.PV[1, 5] * yval * xval + self.PV[1, 6] * xval**2 + self.PV[1, 7] * yval**3 + self.PV[1, 8] * yval**2 * xval + self.PV[1, 9] * yval * xval**2 + self.PV[1, 10] * xval**3) # rad
        fuval = xival / np.cos(self.CRVAL2 * deg2rad) # rad
        fvval = 1. - etaval * np.tan(self.CRVAL2 * deg2rad) # rad
        fwval = etaval + np.tan(self.CRVAL2 * deg2rad) # rad
        faval = np.arctan2(fuval, fvval) # rad
        RAval = self.CRVAL1 + faval * rad2deg # deg
        DECval = np.arctan2(fwval * np.cos(faval), fvval) * rad2deg # deg
        
        return (RAval, DECval)


    def x(self):


        return self.CD11 * (self.ijcat.x[self.ijcat.sidx] - self.CRPIX1) + self.CD12 * (self.ijcat.y[self.ijcat.sidx] - self.CRPIX2) # deg 
  
    def y(self):
        
        return self.CD21 * (self.ijcat.x[self.ijcat.sidx] - self.CRPIX1) + self.CD22 * (self.ijcat.y[self.ijcat.sidx] - self.CRPIX2) # deg
  
    # ---------------------------------------

    def r(self):
        
        return np.sqrt(self.x()**2 + self.y()**2) # deg

    def xi(self):

        return deg2rad * (self.PV[0, 0] + self.PV[0, 1] * self.x() + self.PV[0, 2] * self.y() + self.PV[0, 3] * self.r() + self.PV[0, 4] * self.x()**2 + self.PV[0, 5] * self.x() * self.y() + self.PV[0, 6] * self.y()**2 + self.PV[0, 7] * self.x()**3 + self.PV[0, 8] * self.x()**2 * self.y() + self.PV[0, 9] * self.x() * self.y()**2 + self.PV[0, 10] * self.y()**3)

    def eta(self):

        return deg2rad * (self.PV[1, 0] + self.PV[1, 1] * self.y() + self.PV[1, 2] * self.x() + self.PV[1, 3] * self.r() + self.PV[1, 4] * self.y()**2 + self.PV[1, 5] * self.y() * self.x() + self.PV[1, 6] * self.x()**2 + self.PV[1, 7] * self.y()**3 + self.PV[1, 8] * self.y()**2 * self.x() + self.PV[1, 9] * self.y() * self.x()**2 + self.PV[1, 10] * self.x()**3) # rad

    # ---------------------------------------

    def fu(self):
        return self.xi() / np.cos(self.CRVAL2 * deg2rad) # rad

    def fv(self):
        return 1. - self.eta() * np.tan(self.CRVAL2 * deg2rad) # rad

    def fw(self):
        return self.eta() + np.tan(self.CRVAL2 * deg2rad) # rad

    def fa(self):
        return np.arctan2(self.fu(), self.fv()) # rad

    def RA(self):
        return self.CRVAL1 + self.fa() * rad2deg # deg

    def DEC(self):
        return np.arctan2(self.fw() * np.cos(self.fa()), self.fv()) * rad2deg # deg

    # -------------------------------------------

    def drdx(self):
        
        return self.x() / self.r()

    def drdy(self):
        
        return self.y() / self.r()

    def dxidx(self):
    
        return deg2rad * (self.PV[0, 1] + self.PV[0, 3] * self.drdx() + self.PV[0, 4] * 2. * self.x() + self.PV[0, 5] * self.y() + self.PV[0, 7] * 3. * self.x()**2 + self.PV[0, 8] * 2. * self.x() * self.y() + self.PV[0, 9] * self.y()**2)
    
    def dxidy(self):
        
        return deg2rad * (self.PV[0, 2] + self.PV[0, 3] * self.drdy() + self.PV[0, 5] * self.x() + self.PV[0, 6] * 2. * self.y() + self.PV[0, 8] * self.x()**2 + self.PV[0, 9] * self.x() * 2. * self.y() + self.PV[0, 10] * 3. * self.y()**2)
    
    def detadx(self):
    
        return deg2rad * (self.PV[1, 2] + self.PV[1, 3] * self.drdx() + self.PV[1, 5] * self.y() + self.PV[1, 6] * 2. * self.x() + self.PV[1, 8] * self.y()**2 + self.PV[1, 9] * self.y() * 2. * self.x() + self.PV[1, 10] * 3. * self.x()**2)
    
    def detady(self):
    
        return deg2rad * (self.PV[1, 1] + self.PV[1, 3] * self.drdy() + self.PV[1, 4] * 2. * self.y() + self.PV[1, 5] * self.x() + self.PV[1, 7] * 3. * self.y()**2 + self.PV[1, 8] * 2. * self.y() * self.x() + self.PV[1, 9] * self.x()**2)

    # -----------------------------------------
    
    def dfudCRVAL2(self):
        return self.xi() * np.tan(self.CRVAL2 * deg2rad) / np.cos(self.CRVAL2 * deg2rad) * deg2rad
    
    def dfudCRPIX1(self):
        return -(self.dxidx() * self.CD11 + self.dxidy() * self.CD21) / np.cos(self.CRVAL2 * deg2rad)
    
    def dfudCRPIX2(self):
        return -(self.dxidx() * self.CD12 + self.dxidy() * self.CD22) / np.cos(self.CRVAL2 * deg2rad)
    
    def dfudCD11(self):
        return (self.dxidx() * (self.ijcat.x[self.ijcat.sidx] - self.CRPIX1)) / np.cos(self.CRVAL2 * deg2rad)
    
    def dfudCD12(self):
        return (self.dxidx() * (self.ijcat.y[self.ijcat.sidx] - self.CRPIX2)) / np.cos(self.CRVAL2 * deg2rad)
    
    def dfudCD21(self):
        return (self.dxidy() * (self.ijcat.x[self.ijcat.sidx] - self.CRPIX1)) / np.cos(self.CRVAL2 * deg2rad)
    
    def dfudCD22(self):
        return (self.dxidy() * (self.ijcat.y[self.ijcat.sidx] - self.CRPIX2)) / np.cos(self.CRVAL2 * deg2rad)
    
    # ------------------------------------------
    
    def dfvdCRVAL2(self):
        return -self.eta() * (1. + np.tan(self.CRVAL2 * deg2rad)**2) * deg2rad
    
    def dfvdCRPIX1(self):
        return (self.detadx() * self.CD11 + self.detady() * self.CD21) * np.tan(self.CRVAL2 * deg2rad)
    
    def dfvdCRPIX2(self):
        return (self.detadx() * self.CD12 + self.detady() * self.CD22) * np.tan(self.CRVAL2 * deg2rad)
    
    def dfvdCD11(self):
        return -(self.detadx() * (self.ijcat.x[self.ijcat.sidx] - self.CRPIX1)) * np.tan(self.CRVAL2 * deg2rad)
    
    def dfvdCD12(self):
        return -(self.detadx() * (self.ijcat.y[self.ijcat.sidx] - self.CRPIX2)) * np.tan(self.CRVAL2 * deg2rad)
    
    def dfvdCD21(self):
        return -(self.detady() * (self.ijcat.x[self.ijcat.sidx] - self.CRPIX1)) * np.tan(self.CRVAL2 * deg2rad)
    
    def dfvdCD22(self):
        return -(self.detady() * (self.ijcat.y[self.ijcat.sidx] - self.CRPIX2)) * np.tan(self.CRVAL2 * deg2rad)
    
    # ------------------------------------------
    
    def dfwdCRVAL2(self):
        return (1. + np.tan(self.CRVAL2 * deg2rad)) * deg2rad
    
    def dfwdCRPIX1(self):
        return -(self.detadx() * self.CD11 + self.detady() * self.CD21)
    
    def dfwdCRPIX2(self):
        return -(self.detadx() * self.CD12 + self.detady() * self.CD22)
    
    def dfwdCD11(self):
        return self.detadx() * (self.ijcat.x[self.ijcat.sidx] - self.CRPIX1)
    
    def dfwdCD12(self):
        return self.detadx() * (self.ijcat.y[self.ijcat.sidx] - self.CRPIX2)
    
    def dfwdCD21(self):
        return self.detady() * (self.ijcat.x[self.ijcat.sidx] - self.CRPIX1)
    
    def dfwdCD22(self):
        return self.detady() * (self.ijcat.y[self.ijcat.sidx] - self.CRPIX2)
    
    # ----------------------------------------
    
    def dRAdCRVAL1(self):
        return 1.
    
    def dRAdCRVAL2(self):
        return rad2deg / (1. + (self.fu() / self.fv())**2) * (self.dfudCRVAL2() / self.fv() - self.fu() / self.fv()**2 * self.dfvdCRVAL2())
    
    def dRAdCRPIX1(self):
        return rad2deg / (1. + (self.fu() / self.fv())**2) * (self.dfudCRPIX1() / self.fv() - self.fu() / self.fv()**2 * self.dfvdCRPIX1())
    
    def dRAdCRPIX2(self):
        return rad2deg / (1. + (self.fu() / self.fv())**2) * (self.dfudCRPIX2() / self.fv() - self.fu() / self.fv()**2 * self.dfvdCRPIX2())
    
    def dRAdCD11(self):
        return rad2deg / (1. + (self.fu() / self.fv())**2) * (self.dfudCD11() / self.fv() - self.fu() / self.fv()**2 * self.dfvdCD11())
    
    def dRAdCD12(self):
        return rad2deg / (1. + (self.fu() / self.fv())**2) * (self.dfudCD12() / self.fv() - self.fu() / self.fv()**2 * self.dfvdCD12())
    
    def dRAdCD21(self):
        return rad2deg / (1. + (self.fu() / self.fv())**2) * (self.dfudCD21() / self.fv() - self.fu() / self.fv()**2 * self.dfvdCD21())
    
    def dRAdCD22(self):
        return rad2deg / (1. + (self.fu() / self.fv())**2) * (self.dfudCD22() / self.fv() - self.fu() / self.fv()**2 * self.dfvdCD22())

    # -----------------------------------

    def dDECdCRVAL1(self):
        return 0.
    
    def dDECdCRVAL2(self):
        return rad2deg / (1. + (self.fw() * np.cos(self.fa()) / self.fv())**2) * \
            (self.dfwdCRVAL2() * np.cos(self.fa()) / self.fv() - \
             self.fw() * np.sin(self.fa()) * self.dRAdCRVAL2() / rad2deg / self.fv() - \
             self.fw() * np.cos(self.fa()) / self.fv()**2 * self.dfvdCRVAL2())
    
    def dDECdCRPIX1(self):
        return rad2deg / (1. + (self.fw() * np.cos(self.fa()) / self.fv())**2) * \
            (self.dfwdCRPIX1() * np.cos(self.fa()) / self.fv() - \
             self.fw() * np.sin(self.fa()) * self.dRAdCRPIX1() / rad2deg / self.fv() - \
             self.fw() * np.cos(self.fa()) / self.fv()**2 * self.dfvdCRPIX1())
    
    def dDECdCRPIX2(self):
        return rad2deg / (1. + (self.fw() * np.cos(self.fa()) / self.fv())**2) * \
            (self.dfwdCRPIX2() * np.cos(self.fa()) / self.fv() - \
             self.fw() * np.sin(self.fa()) * self.dRAdCRPIX2() / rad2deg / self.fv() - \
             self.fw() * np.cos(self.fa()) / self.fv()**2 * self.dfvdCRPIX2())
    
    def dDECdCD11(self):
        return rad2deg / (1. + (self.fw() * np.cos(self.fa()) / self.fv())**2) * \
            (self.dfwdCD11() * np.cos(self.fa()) / self.fv() - \
             self.fw() * np.sin(self.fa()) * self.dRAdCD11() / rad2deg / self.fv() - \
             self.fw() * np.cos(self.fa()) / self.fv()**2 * self.dfvdCD11())
    
    def dDECdCD12(self):
        return rad2deg / (1. + (self.fw() * np.cos(self.fa()) / self.fv())**2) * \
            (self.dfwdCD12() * np.cos(self.fa()) / self.fv() - \
             self.fw() * np.sin(self.fa()) * self.dRAdCD12() / rad2deg / self.fv() - \
             self.fw() * np.cos(self.fa()) / self.fv()**2 * self.dfvdCD12())
    
    def dDECdCD21(self):
        return rad2deg / (1. + (self.fw() * np.cos(self.fa()) / self.fv())**2) * \
            (self.dfwdCD21() * np.cos(self.fa()) / self.fv() - \
             self.fw() * np.sin(self.fa()) * self.dRAdCD21() / self.fv() - \
             self.fw() * np.cos(self.fa()) / self.fv()**2 * self.dfvdCD21())
    
    def dDECdCD22(self):
        return rad2deg / (1. + (self.fw() * np.cos(self.fa()) / self.fv())**2) * \
            (self.dfwdCD22() * np.cos(self.fa()) / self.fv() - \
             self.fw() * np.sin(self.fa()) * self.dRAdCD22() / self.fv() - \
             self.fw() * np.cos(self.fa()) / self.fv()**2 * self.dfvdCD22())

