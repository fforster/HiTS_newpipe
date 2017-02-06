from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from stamp import *

class kernel(object):

    def __init__(self, conf, nvar):

        self.nvar = nvar
        self.ivarf = np.loadtxt("%s/ivar_%i.dat" % (conf.etcdir, self.nvar), dtype = int) - 1  # 9, 57, 81, 289, 625
        self.nf = np.shape(self.ivarf)[0]

        print("   Kernel initialized with filter size %i and %i free parameters" % (self.nf, self.nvar))

    # build vector of i and j position for given equation number
    def doieq(self, npsf):

        self.ieq2i = np.zeros(npsf * npsf, dtype = int)
        self.ieq2j = np.zeros(npsf * npsf, dtype = int)
        for i in range(npsf):
            for j in range(npsf):
                ieq = i * npsf + j
                self.ieq2i[ieq] = i
                self.ieq2j[ieq] = j

    # solve kernel given stamp dimensions, pairs of extended stars and convolution of reference boolean
    def solve(self, npsf, pairs, convref):

        # Number of stars
        nstars = len(pairs)

        npsf2 = npsf * npsf
        nfh = int(self.nf / 2)
        self.doieq(npsf)

        # start building kernel
        X = np.zeros((nstars * npsf2, self.nvar))
        Y = np.zeros(nstars * npsf2)

        for i, pair in enumerate(pairs):

            # round values
            ipix = np.round(pair.stampproj.x)
            jpix = np.round(pair.stampproj.y)

            # fill X and Y
            for k in range(self.nf):
                for l in range(self.nf):
                    ivar = self.ivarf[k, l]
                    if ivar == -1:
                        continue
                    if convref:
                        X[i * npsf2: (i + 1) * npsf2, ivar] \
                            = X[i * npsf2: (i + 1) * npsf2, ivar] + pair.stampref.flux[self.ieq2i + k, self.ieq2j + l]
                    else:
                        X[i * npsf2: (i + 1) * npsf2, ivar] \
                            = X[i * npsf2: (i + 1) * npsf2, ivar] + pair.stampproj.flux[self.ieq2i + k, self.ieq2j + l]
            if convref:
                Y[i * npsf2: (i + 1) * npsf2] = pair.stampproj.flux[nfh: -(nfh + 1), nfh: -(nfh + 1)][self.ieq2i, self.ieq2j]
            else:
                Y[i * npsf2: (i + 1) * npsf2] = pair.stampref.flux[nfh: -(nfh + 1), nfh: -(nfh + 1)][self.ieq2i, self.ieq2j]

        # solve filter
        mat = np.dot(X.transpose(), X)
        rhs = np.dot(X.transpose(), Y)
        # L2
        solvars = scipylinalg.solve(mat, rhs)
    
        """
        # L1
        absresiduals = np.abs(Y - np.dot(X, solvars))
        sigmaPH = 10. * np.amax(absresiduals)
        solvars = PosHuber(X, Y, sigmaPH, solvars)
        """
        
        # recover filter
        solfilter = np.zeros((self.nf, self.nf))
        rfilter = np.zeros((self.nf, self.nf))
        for k in range(self.nf):
            for l in range(self.nf):
                ivar = int(self.ivarf[k, l])
                if ivar == -1:
                    solfilter[k, l] = 0
                    continue
                solfilter[k, l] = solvars[ivar]
                rfilter[k, l] = np.sqrt((k - self.nf / 2.)**2 + (l - self.nf / 2.)**2)

        return solfilter
#
#    # compute filter characteristics
#    kratio = np.sum(solvars) / np.sum(np.abs(solvars))
#    ksupport = np.sum(solfilter * rfilter) / np.sum(solfilter)
#    knorm = np.sum(solfilter)
#    knorm2 = knorm * knorm
#    knormsum2 = np.sum(solfilter**2)
#
#    # save train stars and filter
#    if savestars:
#        fileout = "%s/%s/%s/CALIBRATIONS/stars_%s_%s_%02i-%02i_%04i-%04i.npy" % (sharedir, field, CCD, field, CCD, filesci, fileref, ipixref, jpixref)
#        np.save(fileout, np.dstack([psf1s, psf2s]))
#        fileout = "%s/%s/%s/CALIBRATIONS/kernel_%s_%s_%02i-%02i_%04i-%04i.npy" % (sharedir, field, CCD, field, CCD, filesci, fileref, ipixref, jpixref)
#        np.save(fileout, solfilter)
#        
#    return nclose, solfilter, kratio, ksupport, knorm2, knormsum2
