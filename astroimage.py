from __future__ import print_function
from __future__ import division

import numpy as np
import pyfits as fits
import os
from astropy.table import Table

from catalogue import *
from stamp import *
from kernel import *

#from projection import projection
#projection.set_num_threads(4)

class HiTSconfig(object):
    
    def __init__(self, **kwargs):

        # directories
        self.refdir = kwargs["refdir"]
        self.indir = kwargs["indir"]
        self.outdir = kwargs["outdir"]
        self.sharedir = kwargs["sharedir"]
        self.webdir = kwargs["webdir"]
        self.etcdir = kwargs["etcdir"]
        # sextractor parameters
        if "backsize" in kwargs.keys():
            self.backsize = int(kwargs["backsize"])
        else:
            self.backsize = 64
        # verbosity
        if "verbose" in kwargs.keys():
            self.verbose = kwargs["verbose"]
        else:
            self.verbose = False
        
class HiTSimage(object):
    
    def __init__(self, HiTSconf, field, CCD, epoch):
        
        # configuration
        self.conf = HiTSconf

        # observations
        self.field = field
        self.CCD = CCD
        self.epoch = epoch
        self.prefix = "%s/%s/%s/%s_%s_%02i" % (HiTSconf.refdir, self.field, self.CCD, self.field, self.CCD, self.epoch)
        
        # input files
        if field[:8] == "Blind14A" or field[:8] == "Blind15A":
            self.imagefile = "%s_image.fits" % self.prefix
            self.useweights = 'internal'

        elif field[:8] == "Blind13A":
            self.imagefile = "%s_image.fits.fz" % self.prefix
            self.usewights = 'external'

        # sextractor and background file
        self.backgroundfile = self.imagefile.replace(".fits", "_%s_background%i.fits" % (self.useweights, self.conf.backsize))

    # this must be run before loadfits
    def loadsextractor(self, **kwargs):

        dobg = False
        if "dobg" in kwargs.keys():
            dobg = kwargs["dobg"]
    
        # create sextractor catalogue
        if self.conf.verbose:
            print("Creating sextractor catalogue")
        self.sexfile = "%s/%s/%s/%s_%s_%02i_image.fits-catalogue_wtmap_backsize%i.dat" % (self.conf.sharedir, self.field, self.CCD, self.field, self.CCD, self.epoch, self.conf.backsize)
        if not os.path.exists(self.sexfile):
            
            if self.conf.verbose:
                sexverbose = "FULL"
            else:
                sexverbose = "QUIET"
            #if dobg:
            #    bgstring = "-CHECKIMAGE_TYPE BACKGROUND -CHECKIMAGE_NAME %s" % self.backgroundfile
            #else:
            #    bgstring = ""
            command = "HITSPIPE=%s; export HITSPIPE=$HITSPIPE; sextractor -c %s/default.sex %s -CATALOG_NAME %s -WEIGHT_TYPE BACKGROUND -BACK_SIZE %i -CHECKIMAGE_TYPE BACKGROUND -CHECKIMAGE_NAME %s -VERBOSE_TYPE %s" % (self.conf.etcdir[:-4], self.conf.etcdir, self.imagefile, self.sexfile, self.conf.backsize, self.backgroundfile, sexverbose)
            #command = "HITSPIPE=%s; export HITSPIPE=$HITSPIPE; sex -c %s/default.sex %s -CATALOG_NAME %s -WEIGHT_TYPE BACKGROUND -BACK_SIZE %i %s -VERBOSE_TYPE %s" % (self.conf.etcdir[:-4], self.conf.etcdir, self.crblasterfile, self.sexfile, self.conf.backsize, bgstring, sexverbose)

            if self.conf.verbose:
                print(command)
            os.system(command)
        
        # load sextractor catalogue
        if self.conf.verbose:
            print("Loading sextractor catalogue")
        (x, y, z, e_z, r, flag) = np.loadtxt(self.sexfile, usecols = (1, 2, 5, 6, 8, 9)).transpose()
        self.pixcat = catalogue(x = x, y = y, z = z, e_z = e_z, r = r, flag = flag, xlabel = "ipix", ylabel = "jpix", zlabel = "flux", rlabel = "radius", xunit = "pix", yunit = "pix", zunit = "ADU", runit = "pix")
        self.pixcat.readheader(self.imagefile)

    # load GAIA catalogue for astrometry (PANSTARRS in the future for photometric calibrations)
    def loadGAIA(self):

        if self.conf.verbose:
            print("Loading GAIA catalogue file")
        # GAIA file
        self.GAIAfile = "%s/%s/%s/CALIBRATIONS/GAIA_%s_%s.vot" % (self.conf.sharedir, self.field, self.CCD, self.field, self.CCD)

        if os.path.exists(self.GAIAfile):
            GAIA = Table.read(self.GAIAfile, format = 'votable')
            RA = np.array(GAIA['ra']) / 15.
            DEC = np.array(GAIA['dec'])
            g = np.array(GAIA['phot_g_mean_mag'])
            name = np.array(GAIA['source_id'])
            self.RADECmagGAIA = catalogue(x = RA, y = DEC, z = g, xlabel = "RA", ylabel = "DEC", zlabel = "g mag", xunit = "hr", yunit = "deg", zunit = "mag")

        else:
            print("WARNING: cannot find GAIA file %s" % self.GAIAfile)

    # load PanSTARRS catalogue for photometric calibration
    def loadPanSTARRS(self):

        if self.conf.verbose:
            print("Loading PanSTARRS catalogue file")
        # GAIA file
        self.PanSTARRSfile = "%s/%s/%s/CALIBRATIONS/PS1_%s_%s.vot" % (self.conf.sharedir, self.field, self.CCD, self.field, self.CCD)

        if os.path.exists(self.PanSTARRSfile):
            PS1 = Table.read(self.PanSTARRSfile, format = 'votable')
            RA = np.array(PS1['raMean']) / 15.
            DEC = np.array(PS1['decMean'])
            g = np.array(PS1['gMeanKronMag'])
            name = np.array(PS1['objID'])
            self.RADECmagPanSTARRS = catalogue(x = RA, y = DEC, z = g, xlabel = "RA", ylabel = "DEC", zlabel = "g mag", xunit = "hr", yunit = "deg", zunit = "mag")
        else:
            print("WARNING: cannot find PanSTARRS file %s" % self.GAIAfile)

            
    # solve WCS given sextractor and USNO catalogue
    def solveWCS(self):
        
        if not hasattr(self, "pixcat") or not hasattr(self, "RADECmagGAIA"):
            print("WARNING: image must have attributes pixcat and RADECmagGAIA")
        else:
            self.pixcat.matchRADEC(RADECmagcat = self.RADECmagGAIA, solveWCS = True, solveZP = False, doplot = False)

        # save WCS
        self.WCS = self.pixcat.WCSsol.WCS


    # solve WCS given sextractor and USNO catalogue
    def solveZP(self):

        
        if not hasattr(self, "pixcat") or not hasattr(self, "RADECmagPanSTARRS"):
            print("WARNING: image must have attributes pixcat and RADECmagPanSTARRS")
        else:
            self.pixcat.matchRADEC(RADECmagcat = self.RADECmagPanSTARRS, solveWCS = True, solveZP = True, doplot = True)

        ## save WCS
        #self.ZP = self.pixcat.WCSsol.ZP


    # load the main data components
    def loadfits(self):
        
        if self.conf.verbose:
            print("Loading fits files")

        fitsdata = fits.open(self.imagefile)

        # header
        if self.conf.verbose:
            print("   Header...")
        self.header = fitsdata[0].header

        # original data
        if self.conf.verbose:
            print("   Observed flux with sky...")
        self.obsflux = fitsdata[0].data
        
        # data quality mask
        if self.conf.verbose:
            print("   Data quality mask...")
        self.dq = fitsdata[1].data

        # inverse variance
        if self.conf.verbose:
            print("   Variance...")
        self.var = 1. / fitsdata[2].data
        
        # background
        if self.conf.verbose:
            print("   Background... %s" % self.backgroundfile)
        self.bg = fits.open(self.backgroundfile)[1].data
        print(np.shape(self.bg))
        
        # sky subtracted flux
        if self.conf.verbose:
            print("sky subtracted data...")
        self.flux = np.array(self.obsflux) - np.array(self.bg)

        # units
        self.units = 'ADU'

    ## update WCS for crblastered fits file
    #def updateWCS(self):
    #    
    #    for key in ["CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CD1_1", "CD1_2", "CD2_1", "CD2_2"]:
    #        exec("self.header[key] = self.WCS.%s" % key.replace("_", ""))
    #    hdu = fits.PrimaryHDU(data = self.data, header = self.header)
    #    hdulist = fits.HDUList([hdu])
    #    hdulist.writeto(self.crblasterfile, clobber = True)

    # do relative pixel solution
    def solvepixeltransformation(self, image2):

        self.pixcat.matchpix(pixcat = image2.pixcat, nx = self.header["NAXIS1"], ny = self.header["NAXIS2"])

    # project image
    def project(self, image2, **kwargs):

        alanczos = kwargs["alanczos"]
        save = kwargs["save"]
        
        if self.conf.verbose:
            print("Projecting image...")

        # find relative pixel solution from image2 to self
        if self.conf.verbose:
            print("   Solving pixel transformation...")
        image2.solvepixeltransformation(self)

        # Lanczsos resampling
        if self.conf.verbose:
            print("   Lanczos projection..")

        # prepare for projection
        order = image2.pixcat.pol.order
        sol_astrometry = image2.pixcat.pol.sol

        # get dimensions and project
        (ny, nx) = np.shape(self.flux)
        self.nx = nx
        self.ny = ny

        # projection file
        self.projectedfile = "%s_image_grid%02i_lanczos%i.fits" % (self.prefix, image2.epoch, alanczos)

        # do projection if necessary
        print(self.projectedfile)
        if not os.path.exists(self.projectedfile):

            #projection.lanczos(alanczos, nx, ny, order, sol_astrometry, self.flux.transpose(), self.var.transpose(), self.dq.transpose(), self.bg.transpose())
            # get projected image and variance
            self.fluxproj = projection.imageout[0:nx, 0:ny]
            self.varproj = projection.varimageout[0:nx, 0:ny]
            self.dqproj = projection.dqout[0:nx, 0:ny]
            self.bgproj = projection.bgout[0:nx, 0:ny]
            self.fluxproj = np.array(self.fluxproj.transpose())
            self.varproj = np.array(self.varproj.transpose())
            self.dqproj = np.array(self.dqproj.transpose())
            self.bgproj = np.array(self.bgproj.transpose())
            # save projected image
            print(self.projectedfile)
            hduflux = fits.PrimaryHDU(self.fluxproj, header = self.header)
            # inherit WCS
            hduflux.header["CRVAL1"] = image2.header["CRVAL1"]
            hduflux.header["CRVAL2"] = image2.header["CRVAL2"]
            hduflux.header["CRPIX1"] = image2.header["CRVAL2"]
            hduflux.header["CRPIX2"] = image2.header["CRVAL2"]
            hduflux.header["CD1_1"] = image2.header["CD1_1"]
            hduflux.header["CD1_2"] = image2.header["CD1_2"]
            hduflux.header["CD2_2"] = image2.header["CD2_2"]
            hduflux.header["CD2_2"] = image2.header["CD2_2"]
            # hdulist
            hdulist = fits.HDUList([hduflux])
            hdulist.append(fits.ImageHDU(self.dqproj))
            hdulist.append(fits.ImageHDU(self.varproj))
            hdulist.append(fits.ImageHDU(self.bgproj))
            hdulist.writeto(self.projectedfile, clobber = True)
        else:
            data = fits.open(self.projectedfile)
            self.fluxproj = data[0].data
            self.dqproj = data[1].data
            self.varproj = data[2].data
            self.bgproj = data[3].data

    # find isolated sources after pixel matching
    def findisolatedsources(self, imageref, npsf, nf):

        if np.mod(npsf, 2) == 0:
            print ("Number of PSF pixels must be odd")
            sys.exit()
        if np.mod(nf, 2) == 0 and nf != 0:
            print ("Number of filter pixels must be odd or zero")
            sys.exit()
            
        # derived quantities
        npsfh = int(npsf / 2)
        dn = int((npsf + nf) / 2)

        # stamppairlist
        stars = stamppairlist(imageref.pixcat, self.pixcat, npsf, nf)
        
        # loop among matching stars
        for idxsci, idxref in zip(self.pixcat.sidx, imageref.pixcat.sidx):

            # science source properties
            xsci = self.pixcat.x[idxsci]
            ysci = self.pixcat.y[idxsci]
            zsci = self.pixcat.z[idxsci]
            e_zsci = self.pixcat.e_z[idxsci]
            rsci = self.pixcat.r[idxsci]

            # reference source properties
            xref = imageref.pixcat.x[idxref]
            yref = imageref.pixcat.y[idxref]
            zref = imageref.pixcat.z[idxref]
            e_zref = imageref.pixcat.e_z[idxref]
            rref = imageref.pixcat.r[idxref]

            # central pixel positions
            i = int(yref) - 1
            j = int(xref) - 1
            
            
            # find stars within definition range
            #if i > dn and i < self.nx - dn and j > dn and j < self.ny - dn:
            if i > npsf and i < self.nx - npsf and j > npsf and j < self.ny - npsf:

                # image limits
                imin = i - dn
                imax = i + dn
                jmin = j - dn
                jmax = j + dn
                if nf == 0:
                    imax = imax + 1
                    jmax = jmax + 1
                # reference stamp
                stampref = stamp(self. field, self.CCD, xref, yref, zref, e_zref, rref, 0, 0, 0, imageref.flux[imin:imax, jmin:jmax], imageref.var[imin:imax, jmin:jmax], imageref.dq[imin:imax, jmin:jmax], imageref.bg[imin:imax, jmin:jmax])
                # projected stamp
                stampproj = stamp(self.field, self.CCD, xsci, ysci, zsci, e_zsci, rsci, 0, 0, 0, self.fluxproj[imin:imax, jmin:jmax], self.varproj[imin:imax, jmin:jmax], self.dqproj[imin:imax, jmin:jmax], self.bgproj[imin:imax, jmin:jmax])

                # no strange pixels inside star
                if max(stampref.dq.flatten()) < 1 and max(stampproj.dq.flatten()) < 1:
                    pair = stamppair(stampref, stampproj)
                    stars.addpair(pair)

        #print(stars.err1, stars.err2, stars.err3, stars.err4, stars.success)

        # see stars properties
        plotstats = False
        if plotstats:
            stars.plotstats()

        # get psf
        if not stars.computepsf(plotfluxratio = False, plotallstamps = False, plotpsfs = False):
            print("WARNING: cannot compute psf")
            sys.exit()
        
        # do star PCA
        if not stars.doPCA(plotPCA = False):
            print("WARNING: cannot perform PCA over training stars")
            sys.exit()

        # return stars
        return stars

    # convolve best image of a pair of images
    def convolve(self, image2, **kwargs):
        
        # create test stamp
        npsf = kwargs["npsf"]
        nvar = kwargs["nvar"]

        if self.conf.verbose:
            print("Convolving image...")

        # create kernel model
        kconv = kernel(conf, nvar)

        # find common sources for kernel building which have no bad pixels
        stars = self.findisolatedsources(image2, npsf = npsf, nf = kconv.nf)

        # train kernel based on trainkernel stars
        stars.trainkernel(kconv, plotpsfs = True)

        # apply convolution kernel to image
        
        

        return True

        
if __name__ == "__main__":

    # HiTS configuration
    namedir = "/home/fforster/Work/HiTS_newpipe/SNHiTS15J"
    datadir = "%s/DATA" % namedir
    conf = HiTSconfig(refdir = datadir, indir = datadir, outdir = datadir, sharedir = "%s/SHARED" % namedir, webdir = "%s/WEB" % namedir, etcdir = "/home/fforster/Work/HiTS/etc", verbose = True, backsize = 256)

    # HiTS field
    field = "Blind15A_25"
    CCD = "S14"
    epochref = 2
    epochsci = 3

    # load HiTS reference and solve WCS
    print("\nReference")
    imageref = HiTSimage(conf, field, CCD, epochref)
    imageref.loadsextractor()
    imageref.loadGAIA()
    imageref.solveWCS()
    imageref.loadPanSTARRS()
    imageref.solveZP()
    imageref.loadfits()
    sys.exit()

    
    #  load HiTS science and solve WCS
    print("\nScience")
    imagesci = HiTSimage(conf, field, CCD, epochsci)
    imagesci.loadsextractor()
    imagesci.loadGAIA()
    imagesci.solveWCS()
    imagesci.loadPanSTARRS()
    imagesci.solveZP()
    imagesci.loadfits()
    #imagesci.updateWCS()

    # project image
    print("\nProjection")
    imagesci.project(imageref, alanczos = 2, save = True)
    
    # convolve image
    print("\nConvolution")
    imagesci.convolve(imageref, npsf = 21, nvar = 81)
    
    # do difference
    print("\nDifference")

    # do photometry
    print("\nPhotometry")

    
