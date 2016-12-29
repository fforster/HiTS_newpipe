from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import affine_transform

from kernel import *

# image stamp
class stamp(object):

    def __init__(self, field, CCD, x, y, z, e_z, r, RA, DEC, MJD, flux, var, dq, bg):

        self.field = field
        self.CCD = CCD
        self.x = x
        self.y = y
        self.z = z
        self.e_z = e_z
        self.r = r
        self.RA = RA
        self.DEC = DEC
        self.MJD = MJD
        self.flux = flux
        self.var = var
        self.dq = dq
        self.bg = bg

    # show stamp
    def show(self):

        (dx, dy) = np.shape(self.flux)
        fig, ax = plt.subplots(ncols = 2, nrows = 2, figsize = (14, 14))
        ax[0, 0].set_title("flux")
        ax[0, 0].imshow(self.flux, interpolation = 'nearest', origin = 'lower')
        ax[0, 1].set_title("var")
        ax[0, 1].imshow(self.var, interpolation = 'nearest', origin = 'lower')
        (xoff, yoff) = self.offset()
        ax[1, 0].set_title("flux + offset %4.2f %4.2f" % (xoff, yoff))
        ax[1, 0].imshow(self.flux, interpolation = 'nearest', origin = 'lower')
        
        ax[1, 1].set_title("var + offset %4.2f %4.2f" % (xoff, yoff))
        ax[1, 1].imshow(self.var, interpolation = 'nearest', origin = 'lower')
        if np.mod(dx, 2) == 0:
            ax[0, 0].scatter(dx / 2 + yoff, dx / 2 + xoff)
            ax[1, 0].scatter(dx / 2 + yoff, dx / 2 + xoff)
        for i in range(2):
            for j in range(2):
                ax[i, j].set_xlim(-0.5, dx - 0.5)
                ax[i, j].set_ylim(-0.5, dx - 0.5)
                if np.mod(dx, 2) == 0:
                    ax[i, j].axhline((dx) / 2., c = 'k')
                    ax[i, j].axvline((dx) / 2., c = 'k') 
                else:
                    ax[i, j].axhline((dx - 1) / 2., c = 'k')
                    ax[i, j].axvline((dx - 1) / 2., c = 'k')
        plt.show()

    # do optimal photometry given 1D or 2D psf, flux, var and 1D mask
    def optphot(self, psf):

        mask = (self.dq < 1).flatten()
        auxvar = (psf.flatten())[mask] / (np.abs(self.var.flatten()))[mask]
        optf = np.sum(auxvar * (self.flux.flatten())[mask])
        var_optf = np.sum(auxvar * (psf.flatten())[mask])
        optf = optf / var_optf
        var_optf = 1. / var_optf
        
        return optf, var_optf

    # offset image using Spline interpolation
    def offset(self):

        #xoff = -(self.y - int(self.y)) + 0.5
        #yoff = -(self.x - int(self.x)) + 0.5
        xoff = (self.y - int(self.y)) - 0.5
        yoff = (self.x - int(self.x)) - 0.5
        # this moves the image in the xoff, yoff vector
        self.flux = affine_transform(self.flux, np.array([[1, 0], [0, 1]]), offset=(-xoff, -yoff))
        self.var = affine_transform(self.var, np.array([[1, 0], [0, 1]]), offset=(-xoff, -yoff))
        self.dq = affine_transform(self.dq, np.array([[1, 0], [0, 1]]), offset=(-xoff, -yoff))
        self.bg = affine_transform(self.bg, np.array([[1, 0], [0, 1]]), offset=(-xoff, -yoff))

        return (xoff, yoff)
    
# pair of stamps
class stamppair(object):

    def __init__(self, stampref, stampproj):

        self.stampref = stampref
        self.stampproj = stampproj
        self.kerneltrain = False
        self.psftrain = False


    # stamp visualization
    def show(self):

        fig, ax = plt.subplots(ncols = 2, nrows = 4)
        ax[0, 0].set_title("%i %i %i" % (self.stampref.x, self.stampref.y, max(self.stampref.flux.flatten())))
        ax[0, 0].imshow(self.stampref.flux, interpolation = 'nearest')
        ax[1, 0].imshow(self.stampref.var, interpolation = 'nearest')
        ax[2, 0].imshow(self.stampref.dq, interpolation = 'nearest')
        ax[3, 0].imshow(self.stampref.bg, interpolation = 'nearest')
        ax[0, 1].set_title("%i %i %i" % (self.stampproj.x, self.stampproj.y, max(self.stampproj.flux.flatten())))
        ax[0, 1].imshow(self.stampproj.flux, interpolation = 'nearest')
        ax[1, 1].imshow(self.stampproj.var, interpolation = 'nearest')
        ax[2, 1].imshow(self.stampproj.dq, interpolation = 'nearest')
        ax[3, 1].imshow(self.stampproj.bg, interpolation = 'nearest')

        fig.subplots_adjust(wspace = 0, hspace = 0)
        plt.show()


# list of stamp pairs
class stamppairlist(object):

    # initialize
    def __init__(self, catref, catsci, npsf, nf):

        self.pairs = []
        self.catref = catref
        self.catsci = catsci
        self.npsf = npsf
        self.nf = nf

        # error types
        self.err1 = 0
        self.err2 = 0
        self.err3 = 0
        self.err4 = 0
        self.success = 0

        # determine the direction of the convolution
        self.rrefmedian = np.median(self.catref.r[self.catref.sidx])
        self.rscimedian = np.median(self.catsci.r[self.catsci.sidx])
        
        if self.rrefmedian <= self.rscimedian:
            self.convref = True
            rtest = self.catref.r[self.catref.sidx]
            print("   Convolution will be done to the reference image")
        else:
            self.convref = False
            rtest = self.catsci.r[self.catsci.sidx]
            print("   Convolution will be done to the science projected image")
        
        # check maximum and minimum radius (median +- 2 MAD)
        self.rmedian = np.median(rtest)
        self.rMAD = np.median(np.abs(rtest - self.rmedian))
        self.rmin = self.rmedian - self.rMAD
        self.rmax = self.rmedian + self.rMAD

        # for kernel training stars only
        Xstars, Ystars = np.meshgrid(np.array(range(self.npsf + self.nf)), np.array(range(self.npsf + self.nf)))
        self.rs2Dstars = np.array(np.sqrt((Xstars - (self.npsf + self.nf) / 2.)**2 + (Ystars - (self.npsf + self.nf) / 2.)**2)).flatten()


    # add stamp and determine whether it can be used for kernel and PSF training
    def addpair(self, stamppair):

        if self.convref:
            x = stamppair.stampref.x
            y = stamppair.stampref.y
            r = stamppair.stampref.r
            xcat = self.catref.x[self.catref.sidx]
            ycat = self.catref.y[self.catref.sidx]
        else:
            x = stamppair.stampproj.x
            y = stamppair.stampproj.y
            r = stamppair.stampproj.r
            xcat = self.catsci.x[self.catsci.sidx]
            ycat = self.catsci.y[self.catsci.sidx]
            
        # find closest star distance
        dmin = np.argmin(map(lambda xcat, ycat: np.sqrt((xcat[xcat != x] - x)**2 + (ycat[ycat != y] - y)**2), xcat, ycat))
        
        # filter by minimum distance, radius
        if dmin < 2. * (self.npsf + self.nf):
            self.err1 += 1
            return

        
        # filter by size
        if r < self.rmin or r > self.rmax:
            self.err2 += 1
            return

        # filter by maximum radius with large enough SNR
        psf1 = stamppair.stampref.flux
        psf2 = stamppair.stampproj.flux
        if np.sum(psf1.flatten() > 3. * np.std(psf1)) > 0:
            r1signalmax = np.max(self.rs2Dstars[psf1.flatten() > 3. * np.std(psf1)])
        else:
            r1signalmax = -1

        if np.sum(psf2.flatten() > 3. * np.std(psf2)) > 0:
            r2signalmax = np.max(self.rs2Dstars[psf2.flatten() > 3. * np.std(psf2)])
        else:
            r2signalmax = -1

        if r1signalmax > self.npsf / 4. or r2signalmax > self.npsf / 4. or r1signalmax < 0 or r2signalmax < 0:
            self.err3 += 1
            return

        # filter by radius of maximum signal
        maxSNRradius = 5
        if self.rs2Dstars[np.argmax(psf1)] > maxSNRradius or self.rs2Dstars[np.argmax(psf2)] > maxSNRradius:
            self.err4 += 1
            return

        self.success += 1
        # add if all previous tests are OK
        stamppair.kerneltrain = True
        self.pairs.append(stamppair)

    # check optimal photometry as a check for psf building
    def computepsf(self):

        # get first estimate of psf
        psftryref = np.array(map(lambda pair: pair.stampref.flux, self.pairs))
        psftryproj = np.array(map(lambda pair: pair.stampproj.flux, self.pairs))

        if self.convref:
            psftry = psftryproj
        else:
            psftry = psftryref
        psftry = np.sum(psftry, axis = 0)
        psftry = psftry / np.sum(psftry.flatten())

        # compute optimal photometry using the previous psf
        if self.convref:
            optflux, var_optflux = np.array(map(lambda pair: pair.stampproj.optphot(psftry), self.pairs)).transpose()
            zcomp = np.array(map(lambda pair: pair.stampproj.z, self.pairs))
        else:
            optflux, var_optflux = np.array(map(lambda pair: pair.stampref.optphot(psftry), self.pairs)).transpose()
            zcomp = np.array(map(lambda pair: pair.stampref.z, self.pairs))

        # mask galaxies
        maskflux = np.array(optflux / zcomp > 1, dtype = bool)
        
        # plot flux comparison
        plotfluxratio = False
        if plotfluxratio:
            fig, ax = plt.subplots(ncols = 2)
            ax[0].scatter(optflux[maskflux], zcomp[maskflux], lw = 0, c = 'r')
            ax[0].scatter(optflux[maskflux], zcomp[maskflux], lw = 0, c = 'b')
            ax[1].scatter(optflux, optflux / zcomp, lw = 0)
            ax[1].axhline(1.)
            ax[0].set_xscale('log')
            ax[0].set_yscale('log')
            ax[1].set_xscale('log')
            plt.show()

        # plot all stamps
        plotallstamps = False
        if plotallstamps:
            nplots = int(np.ceil((len(optflux + 1) / 10.)))
            fig, ax = plt.subplots(10, nplots, figsize = (10. * (nplots / 10.), 10.))
            for iplot in range(nplots):
                for j in range(10):
                    istar = iplot * 10 + j
                    if istar < len(optflux):
                        ax[j, iplot].axes.get_xaxis().set_visible(False)
                        ax[j, iplot].axes.get_yaxis().set_visible(False)
                        if self.convref:
                            ax[j, iplot].imshow(self.pairs[istar].stampref.flux, interpolation = 'nearest')
                        else:
                            ax[j, iplot].imshow(self.pairs[istar].stampproj.flux, interpolation = 'nearest')
                            
                        ax[j, iplot].text(0, 0, "%i" % (istar), fontsize = 7)
                        if not maskflux[istar]:
                            ax[j, iplot].spines['top'].set_color('red')
                            ax[j, iplot].spines['bottom'].set_color('red')
                            ax[j, iplot].spines['left'].set_color('red')
                            ax[j, iplot].spines['right'].set_color('red')
                            ax[j, iplot].plot([0, self.npsf + self.nf], [0, self.npsf + self.nf], 'k')
                            ax[j, iplot].plot([self.npsf + self.nf, 0], [0, self.npsf + self.nf], 'k')
                            ax[j, iplot].set_xlim(0, self.npsf + self.nf)
                            ax[j, iplot].set_ylim(0, self.npsf + self.nf)
            
            plt.show()


        # center psfs
        #for pair in self.pairs:
        #    pair.stampref.offset()
        #    pair.stampproj.offset()

        # get new estimate of psf
        psfproj = np.array(map(lambda pair: pair.stampproj.flux, self.pairs))
        psfref = np.array(map(lambda pair: pair.stampref.flux, self.pairs))
            
        # plot all psfs
        showallpsfs = False
        if showallpsfs:
            for pair, mask in zip(self.pairs, maskflux):
                if mask:
                    pair.stampproj.show()
                    pair.stampref.show()

        # average and normalize
        psfref = np.sum(psfref[maskflux], axis = 0)
        psfref = psfref / np.sum(psfref.flatten())
        psfproj = np.sum(psfproj[maskflux], axis = 0)
        psfproj = psfproj / np.sum(psfproj.flatten())

        # extract npsf sized stamps
        idx1 = int(self.nf / 2) + 1
        idx2 = self.npsf + self.nf - idx1 + 1
        psfref = psfref[idx1: idx2, idx1: idx2]
        psfproj = psfproj[idx1: idx2, idx1: idx2]
        if self.convref:
            psf = psfproj
        else:
            psf = psfref

        # save psf training status
        for pair in self.pairs:
            pair.psftrain = True

        # add final psf and show
        self.psfref = psfref
        self.psfproj = psfproj
        self.psf = psf

        showpsf = True
        if showpsf:
            fig, ax = plt.subplots(ncols = 2, figsize = (10, 5))
            ax[0].imshow(psfref, interpolation = 'nearest')
            ax[1].imshow(psfproj, interpolation = 'nearest')
            ax[0].axhline((self.npsf - 1) / 2.)
            ax[1].axhline((self.npsf - 1) / 2.)
            ax[0].axvline((self.npsf - 1) / 2.)
            ax[1].axvline((self.npsf - 1) / 2.)
            plt.show()

        return True

    # do PCA over stars used for psf training
    def doPCA(self):

        # select stars
        snrs = []
        for pair in self.pairs:

            if not pair.psftrain:
                continue
            
            if self.convref:
                snr = pair.stampproj.flux / np.sqrt(pair.stamppro.var)
            else:
                snr = pair.stampref.flux / np.sqrt(pair.stampref.var)

            snrs.append(snr.flatten())

        # do PCA
        from sklearn.decomposition import PCA
        nk = 7
        try:
            pca = PCA(n_components = nk)#min(np.shape(psfPCA)[0], npsf2))
            pca.fit(snrs)
            self.eigenval = pca.explained_variance_ratio_
            self.eigenvec = pca.components_
        except:
            print("\n\n Failed to do SVD decomposition\n\n")
            sys.exit(40)

        # plot PCA
        doplotPCA = False
        if doplotPCA:
            print("   Plotting PCA eigenvectors")
            # plot eigenvalues
            fig, ax = plt.subplots()
            ax.plot(np.cumsum(self.eigenval))# / np.sum(eigenval))
            ax.plot(np.array([nk, nk]), np.array([0, 1]), 'k')
            ax.set_ylim(0, 1)
            ax.set_xlabel("Number of components")
            ax.set_ylabel("Fraction of the total variance")
            plt.show()
            # plot images of eigenvectors
            nyeigen = 1
            fig, ax = plt.subplots(ncols = nk, figsize = (19, 11))
            for jPCA in range(nk):
                try:
                    ax[jPCA].imshow(self.eigenvec[jPCA].reshape(self.npsf + self.nf, self.npsf + self.nf), interpolation = 'nearest')
                except:
                    print("   -----------> Cannot plot eigenvector")
                    continue
            plt.show()

        return True

    # plot summary stats 
    def plotstats(self):

        fig, ax = plt.subplots(ncols = 3, figsize = (14, 10))
        ax[0].errorbar(map(lambda pair: pair.stampref.z, self.pairs),
                       map(lambda pair: pair.stampproj.z, self.pairs),
                       xerr = map(lambda pair: pair.stampref.e_z, self.pairs),
                       yerr = map(lambda pair: pair.stampproj.e_z, self.pairs),
                       marker = '.', c = 'r', lw = 0, elinewidth = 1)
        ax[1].scatter(map(lambda pair: pair.stampref.r, self.pairs),
                      map(lambda pair: pair.stampproj.r, self.pairs),
                      c = 'b', lw = 0)
            
        ax[2].hist(map(lambda pair: pair.stampref.r, self.pairs))
        
        ax[0].set_xlabel('ref flux')
        ax[0].set_ylabel('sci flux')
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[1].set_xlabel('ref radius')
        ax[1].set_ylabel('sci radius')
        ax[2].set_xlabel('ref radius')
        ax[2].set_ylabel('N')
        plt.show()


    # train the kernel
    def trainkernel(self, kconv):

        for pair in self.pairs:

            print(pair.stampproj.flux)
