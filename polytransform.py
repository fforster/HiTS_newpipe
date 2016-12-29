import numpy as np
from scipy import linalg as scipylinalg
import sys

class polytransform(object):
    
    def __init__(self, order):

        self.order = order

    def solve(self, x1, y1, x2, y2):

        # solve arbitrary order transformation between two coordinate systems
        # find best transformation relating all these points
        # need to write the system of equations (e.g. for cubic order):
        # x' = a1 + b11 x + b12 y + c11 x^2 + c12 x y + c13 y^2 + d11 x^3 + d12 x^2 y + d13 x y^2 + d14 y^3...
        # y' = a2 + b21 x + b22 y + c21 x^2 + c22 x y + c23 y^2 + d21 x^3 + d22 x^2 y + d23 x y^2 + d24 y^3......
        # X' = X beta
        # we use beta = (a1 b11 b12 c11 c12 c13 d11 d12 d13 d14 a2 b21 b22 c21 c22 c23 d21 d22 d23 d24)^T
        # then e.g. for order 3
        # X' = (x1...xn y1...yn)^T, X = ((1 x1 y1 x1*y1 x1^2 y1^2 x1^2*y1 x1*y1^2 x1^3 y1^3 0 0 0 0 0 0 0 0 0 0) ... (1 xn yn xn*yn xn^2 yn^2 xn^2*yn xn*yn^2 xn^3 yn^3 0 0 0 0 0 0 0 0 0) (0 0 0 0 0 0 0 0 0 0 1 x1 y1 x1*y1 x1^2 y1^2 x1^2*y1 x1*y1^2 x1^3 y1^3) ... (0 0 0 0 0 0 0 0 0 0 1 xn yn xn*yn xn^2 yn^2 xn^2*yn xn*yn^2 xn^3 yn^3)
        # the least squares errors is found that beta which is solution of the following linear system
        # (X^T X) beta = (X^T X')
        # below we use the notation X'->Y
        
        if self.order == 1:
            nptmin = 3
        elif self.order == 2:
            nptmin = 6
        elif self.order == 3:
            nptmin = 10
            
        npt = len(x1)
        if npt < nptmin:
            print "\n\nWARNING: Not enough stars to do order %i astrometric solution (%i)...\n\n" % (self.order, npt)
            sys.exit(15)
        Y = np.zeros(2 * npt)
        Y[0:npt] = x2
        Y[npt: 2 * npt] = y2
        X = np.zeros((2 * npt, 2 * nptmin))
        iterm = 0
        X[0: npt, iterm] = 1.
        iterm = iterm + 1
        X[0: npt, iterm] = x1
        iterm = iterm + 1
        X[0: npt, iterm] = y1
        iterm = iterm + 1
        if self.order > 1:
            X[0: npt, iterm] = x1 * x1
            iterm = iterm + 1
            X[0: npt, iterm] = x1 * y1
            iterm = iterm + 1
            X[0: npt, iterm] = y1 * y1
            iterm = iterm + 1
        if self.order > 2:
            X[0: npt, iterm] = x1 * x1 * x1
            iterm = iterm + 1
            X[0: npt, iterm] = x1 * x1 * y1
            iterm = iterm + 1
            X[0: npt, iterm] = x1 * y1 * y1
            iterm = iterm + 1
            X[0: npt, iterm] = y1 * y1 * y1
            iterm = iterm + 1
        for jterm in range(iterm):
            X[npt: 2 * npt, iterm + jterm] = X[0:npt, jterm]
            X[npt: 2 * npt, iterm + jterm] = X[0:npt, jterm]
            X[npt: 2 * npt, iterm + jterm] = X[0:npt, jterm]
            X[npt: 2 * npt, iterm + jterm] = X[0:npt, jterm]
            X[npt: 2 * npt, iterm + jterm] = X[0:npt, jterm]
            X[npt: 2 * npt, iterm + jterm] = X[0:npt, jterm]
            X[npt: 2 * npt, iterm + jterm] = X[0:npt, jterm]
            X[npt: 2 * npt, iterm + jterm] = X[0:npt, jterm]
            X[npt: 2 * npt, iterm + jterm] = X[0:npt, jterm]
            X[npt: 2 * npt, iterm + jterm] = X[0:npt, jterm]
        # solve
        mat = np.dot(X.transpose(), X)
        rhs = np.dot(X.transpose(), Y)
        try:
            print "   Solving order %i transformation (npt: %i)..." % (self.order, npt)
            if self.order == 1:
                (a1, b11, b12, a2, b21, b22) = scipylinalg.solve(mat, rhs)
                self.sol = np.array([a1, a2, b11, b12, b21, b22])
            elif self.order == 2:
                (a1, b11, b12, c11, c12, c13, a2, b21, b22, c21, c22, c23) = scipylinalg.solve(mat, rhs)
                self.sol = np.array([a1, a2, b11, b12, b21, b22, c11, c12, c13, c21, c22, c23])
            elif self.order == 3:
                (a1, b11, b12, c11, c12, c13, d11, d12, d13, d14, a2, b21, b22, c21, c22, c23, d21, d22, d23, d24) = scipylinalg.solve(mat, rhs)
                self.sol = np.array([a1, a2, b11, b12, b21, b22, c11, c12, c13, c21, c22, c23, d11, d12, d13, d14, d21, d22, d23, d24])
        except:
            print "\n\nWARNING: Error solving linear system when matching pixel coordinate systems\n\n"
            sys.exit(16)

    # apply transformation
    def apply(self, x1, y1):
        
        # this is slow, but I prefer fewer bugs than speed at the moment...
        
        x1t = self.sol[0] + self.sol[2] * x1 + self.sol[3] * y1
        y1t = self.sol[1] + self.sol[4] * x1 + self.sol[5] * y1
        if self.order > 1:
            x1t = x1t + self.sol[6] * x1 * x1 + self.sol[7] * x1 * y1 + self.sol[8] * y1 * y1
            y1t = y1t + self.sol[9] * x1 * x1 + self.sol[10] * x1 * y1 + self.sol[11] * y1 * y1
        if self.order > 2:
            x1t = x1t + self.sol[12] * x1 * x1 * x1 + self.sol[13] * x1 * x1 * y1 + self.sol[14] * x1 * y1 * y1 + self.sol[15] * y1 * y1 * y1
            y1t = y1t + self.sol[16] * x1 * x1 * x1 + self.sol[17] * x1 * x1 * y1 + self.sol[18] * x1 * y1 * y1 + self.sol[19] * y1 * y1 * y1
            
        return (x1t, y1t)

