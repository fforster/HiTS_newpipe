import numpy as np
import matplotlib.pyplot as plt
import pyfits as fits
from WCS import *
from WCSsol import *
from polytransform import *
import os

deg2rad = np.pi / 180.
rad2deg = 180. / np.pi

class catalogue(object):

    def __init__(self, **kwargs):
        
        self.x = np.array(kwargs["x"], dtype = float)
        self.y = np.array(kwargs["y"], dtype = float)
        self.z = np.array(kwargs["z"], dtype = float)
        if "e_z" in kwargs.keys():
            self.e_z = np.array(kwargs["e_z"], dtype = float)
        self.xlabel = kwargs["xlabel"]
        self.ylabel = kwargs["ylabel"]
        self.zlabel = kwargs["zlabel"]
        self.xunit = kwargs["xunit"]
        self.yunit = kwargs["yunit"]
        self.zunit = kwargs["zunit"]
        if "r" in kwargs.keys():
            self.r = np.array(kwargs["r"], dtype = float)
            self.rlabel = kwargs["rlabel"]
            self.runit = kwargs["runit"]
        if "flag" in kwargs.keys():
            self.flag = kwargs["flag"]
        if "CRPIXguess" in kwargs.keys():
            self.CRPIXguess = kwargs["CRPIXguess"]
        if "CRVALguess" in kwargs.keys():
            self.CRVALguess = kwargs["CRVALguess"]
        if "CDguess" in kwargs.keys():
            self.CDguess = kwargs["CDguess"]
        if "PV" in kwargs.keys():
            self.PV = kwargs["PV"]

        self.indices = np.array(range(len(self.x)), dtype = int)
        self.sidx = self.indices # selected and sorted indices

    def readheader(self, filename):

            # get initial solution from reference header

            header = fits.open(filename)[0].header
            nPV1 = 2
            nPV2 = 11
            PV = np.zeros((nPV1, nPV2))
            for i in range(nPV1):
                for j in range(nPV2):
                    PV[i, j] = float(header["PV%i_%i" % (i + 1, j)])
            CD = np.zeros((2, 2))
            for i in range(2):
                for j in range(2):
                    CD[i, j] = float(header["CD%i_%i" % (i + 1, j + 1)])
            CRPIX = np.zeros(2)
            CRVAL = np.zeros(2)
            for i in range(2):
                CRPIX[i] = float(header["CRPIX%i" % (i + 1)])
                CRVAL[i] = float(header["CRVAL%i" % (i + 1)])

            # save values
            self.CRPIXguess = CRPIX
            self.CRVALguess = CRVAL
            self.CDguess = CD
            self.PV = PV
    

    def matchRADEC(self, **kwargs):
        
        RADECmagcat = kwargs["RADECmagcat"]
        solveWCS = True
        solveZP = True
        doplot = False
        outdir = '.'
        outname = 'test'
        verbose = False
        if "solveWCS" in kwargs.keys():
            solveWCS = kwargs["solveWCS"]
        if "solveZP" in kwargs.keys():
            solveZP = kwargs["solveZP"]
        if "doplot" in kwargs.keys():
            doplot = kwargs["doplot"]
        if "outdir" in kwargs.keys():
            outdir = kwargs["outdir"]
        if "outname" in kwargs.keys():
            outname = kwargs["outname"]
        if "verbose" in kwargs.keys():
            verbose = kwargs["verbose"]
        
        print "   Matching sets of stars to RA DEC catalogue, checking catalogue requirements"
        if self.xunit != 'pix' or self.yunit != 'pix' or RADECmagcat.xlabel != 'RA' or RADECmagcat.ylabel != 'DEC' or RADECmagcat.zunit != 'mag':# or not self.hasattr("CRPIXguess") or not self.hasattr("CRVALguess") or not self.hasattr("CDguess") or not self.hasattr("PV"):
            print "Inconsistent attributes to do coordinate matching"
            return False

        # transform RA to degrees
        if RADECmagcat.xunit == 'hr':
            RADECmagcat.x = RADECmagcat.x * 15.
            RADECmagcat.xunit = 'deg'
    
        # sort celestial catalogue stars by flux
        idxsorted = np.argsort(RADECmagcat.z)
        RADECmagcat.x = RADECmagcat.x[idxsorted]
        RADECmagcat.y = RADECmagcat.y[idxsorted]
        RADECmagcat.z = RADECmagcat.z[idxsorted]

        # sort pixel catalogue by flux
        idxsorted = np.argsort(self.z)[::-1]
        self.x = self.x[idxsorted]
        self.y = self.y[idxsorted]
        self.z = self.z[idxsorted]
        self.r = self.r[idxsorted]
        self.flag = self.flag[idxsorted]

        # select brightest isolated stars from sextractor catalogue and store mask
        npix = 100
        nstars = 60
        maskpixcat = (self.flag == 0) & (self.r < np.percentile(self.r, 90)) & (self.z > np.percentile(self.z, 90))
        nstarspix = min(np.sum(maskpixcat), nstars)
        pixcatdistmin = np.array(map(lambda i, j: np.min(np.sqrt((self.x[maskpixcat][:nstarspix][(self.x[maskpixcat][:nstarspix] != i) & (self.y[maskpixcat][:nstarspix] != j)] - i)**2 + (self.y[maskpixcat][:nstarspix][(self.x[maskpixcat][:nstarspix] != i) & (self.y[maskpixcat][:nstarspix] != j)] - j)**2)), self.x[maskpixcat][:nstarspix], self.y[maskpixcat][:nstarspix]))
        maskpixcatdist = (pixcatdistmin > npix)
        self.sidx = self.indices[maskpixcat][:nstarspix][maskpixcatdist]

        # create WCS object and evaluate RA and DEC
        sol = WCS(self, self.CRPIXguess, self.CRVALguess, self.CDguess, self.PV)
        (RA, DEC) = (sol.RA(), sol.DEC())

        # select brightest isolated stars from celestial catalogue and store indices
        maskRADECcat = (RADECmagcat.x > min(RA)) & (RADECmagcat.x < max(RA)) & (RADECmagcat.y > min(DEC)) & (RADECmagcat.y < max(DEC))
        nstarsRADEC = min(np.sum(maskRADECcat), nstars)
        distmin = np.array(map(lambda i, j: np.min(np.sqrt((RADECmagcat.x[maskRADECcat][:nstarsRADEC][(RADECmagcat.x[maskRADECcat][:nstarsRADEC] != i) & (RADECmagcat.y[maskRADECcat][:nstarsRADEC] != j)] - i)**2 + (RADECmagcat.y[maskRADECcat][:nstarsRADEC][(RADECmagcat.x[maskRADECcat][:nstarsRADEC] != i) & (RADECmagcat.y[maskRADECcat][:nstarsRADEC] != j)] - j)**2)), RADECmagcat.x[maskRADECcat][:nstarsRADEC], RADECmagcat.y[maskRADECcat][:nstarsRADEC]))
        maskRADECcatdist = (distmin > 100. * 0.27 / 60. / 60.)
        RADECmagcat.sidx = RADECmagcat.indices[maskRADECcat][:nstarsRADEC]

        # REMOVE
        fig, ax = plt.subplots()
        ax.scatter(RADECmagcat.x[maskRADECcat], RADECmagcat.y[maskRADECcat], c = 'b', alpha = 0.5, marker = '.')
        ax.scatter(RA, DEC, c = 'r', alpha = 0.5, marker = 'o', s = 50)
        plt.show()

        # plot both
        if doplot:
            print("Plotting...")
            try:
                fig, ax = plt.subplots()
                ax.scatter(RA, DEC, alpha = 0.6, marker = 'd', c = pixcatdistmin[maskpixcatdist], lw = 0, s = 50)
                ax.scatter(RADECmagcat.x[maskRADECcat][:nstarsRADEC][maskRADECcatdist], RADECmagcat.y[maskRADECcat][:nstarsRADEC][maskRADECcatdist], alpha = 0.6, s = 100, c = distmin[maskRADECcatdist])
                ax.set_xlim(min(RA), max(RA))
                ax.set_ylim(min(DEC), max(DEC))
                ax.set_title("Bright, isolated stars")
                ax.set_xlabel("RA [deg]")
                ax.set_ylabel("DEC [deg]")
                plt.savefig("%s/%s_stars.png" % (outdir, outname))
            except:
                print "Problem plotting stars..."

        # find matching star indices and compute offsets
        idxmatch = map(lambda x, y: np.argmin((RA - x)**2 + (DEC - y)**2), RADECmagcat.x[maskRADECcat][:nstarsRADEC][maskRADECcatdist], RADECmagcat.y[maskRADECcat][:nstarsRADEC][maskRADECcatdist])
        deltaRA = RADECmagcat.x[maskRADECcat][:nstarsRADEC][maskRADECcatdist] - RA[idxmatch]
        deltaDEC = RADECmagcat.y[maskRADECcat][:nstarsRADEC][maskRADECcatdist] - DEC[idxmatch]
        
        # find most common offset
        nclose = np.zeros(np.shape(deltaRA))
        for i in range(len(nclose)):
            dist = np.sqrt((deltaRA - deltaRA[i])**2 + (deltaDEC - deltaDEC[i])**2)
            nclose[i] = np.sum(dist < 5. * 0.27 / 60. / 60.)
        dRA = deltaRA[np.argmax(nclose)]
        dDEC = deltaDEC[np.argmax(nclose)]

        # plot offsets
        if doplot:
            fig, ax = plt.subplots()
            ax.scatter(deltaRA[nclose > 1], deltaDEC[nclose > 1], c = nclose[nclose > 1], lw = 0)
            ax.scatter(deltaRA, deltaDEC, c = nclose, lw = 0, marker = '*')
            ax.set_xlabel("delta RA [deg]")
            ax.set_ylabel("delta DEC [deg]")
            ax.set_title("Raw offsets")
            plt.savefig("%s/%s_stars_delta.png" % (outdir, outname))
            
        # correct and recalculate distance to closest star in other catalogue
        self.sidx = self.indices[maskpixcat][:nstarspix]
        sol = WCS(self, self.CRPIXguess, self.CRVALguess, self.CDguess, self.PV)
        (RA, DEC) = (sol.RA(), sol.DEC())

        # plot corrected positions
        if doplot:
            fig, ax = plt.subplots()
            ax.scatter(RA + dRA, DEC + dDEC)
            ax.scatter(RADECmagcat.x[maskRADECcat][:nstarsRADEC], RADECmagcat.y[maskRADECcat][:nstarsRADEC], alpha = 0.6, s = 100, c = 'r')
            ax.set_xlabel("RA [deg]")
            ax.set_ylabel("DEC [deg]")
            ax.set_title("Star positions after offset")
            plt.savefig("%s/%s_stars_corrected.png" % (outdir, outname))
      
        # find index of closest match
        idxmatch = map(lambda x, y: np.argmin((RA + dRA - x)**2 + (DEC + dDEC - y)**2), RADECmagcat.x[maskRADECcat][:nstarsRADEC], RADECmagcat.y[maskRADECcat][:nstarsRADEC])
        distmatch = np.sqrt((RA[idxmatch] + dRA - RADECmagcat.x[maskRADECcat][:nstarsRADEC])**2 + (DEC[idxmatch] + dDEC - RADECmagcat.y[maskRADECcat][:nstarsRADEC])**2)
        maskdist = (distmatch < 5. * 0.27 / 60. / 60.)
        ipixcatmatch = self.x[maskpixcat][:nstarspix][idxmatch][maskdist]
        jpixcatmatch = self.y[maskpixcat][:nstarspix][idxmatch][maskdist]
        self.sidx = self.indices[maskpixcat][:nstarspix][idxmatch][maskdist]

        RAmatch = RADECmagcat.x[maskRADECcat][:nstarsRADEC][maskdist]
        DECmatch = RADECmagcat.y[maskRADECcat][:nstarsRADEC][maskdist]
        RADECmagcat.sidx = RADECmagcat.indices[maskRADECcat][:nstarsRADEC][maskdist]

        # create final WCSsol
        sol = WCS(self, self.CRPIXguess, self.CRVALguess, self.CDguess, self.PV)
        (RA, DEC) = (sol.RA(), sol.DEC())

        # plot matched stars
        if doplot:
            fig, ax = plt.subplots()
            map(lambda x, y, z, a: ax.plot([x, y], [z, a], c = 'k'), RA, RADECmagcat.x[RADECmagcat.sidx], DEC, RADECmagcat.y[RADECmagcat.sidx])
            ax.set_xlabel("RA [deg]")
            ax.set_ylabel("DEC [deg]")
            plt.savefig("%s/%s_stars_match.png" % (outdir, outname))

        # apply first order correction
        self.CRVALguess = self.CRVALguess + np.array([dRA, dDEC])

        # solve WCS
        if solveWCS:

            # find astrometric solution
            self.WCSsol = WCSsol(RADECmagcat, WCS(self, self.CRPIXguess, self.CRVALguess, self.CDguess, self.PV))
            self.WCSsol.solve()

            if doplot:
            
                fig, ax = plt.subplots()
                ax.scatter(3600. * (self.WCSsol.WCS.RA() - RADECmagcat.x[RADECmagcat.sidx]), 3600. * (self.WCSsol.WCS.DEC() - RADECmagcat.y[RADECmagcat.sidx]))
                ax.axvline(0)
                ax.axhline(0)
                ax.set_xlabel("Delta RA [arcsec]")
                ax.set_ylabel("Delta DEC [arcsec]")
                plt.savefig("%s/%s_WCS_delta.png" % (outdir, outname))

        # solve ZP
        if solveZP:

            if doplot:
            
                fig, ax = plt.subplots()
                print "sum:", np.shape(RADECmagcat.sidx)
                print RADECmagcat.z[RADECmagcat.sidx]
                ax.scatter(RADECmagcat.z[RADECmagcat.sidx], RADECmagcat.z[RADECmagcat.sidx])
                plt.savefig("%s/%s_ZP.png" % (outdir, outname))
            print "Testing"
            

    # select N brightest objects well inside the image edges and that are not in crowded regions
    def select(self, x, y, z, tolx, toly, xmax, xmin, ymax, ymin, error, N):
    
        # mask points too close to the edges
        maskout = (np.abs(x - (xmax + xmin) / 2.) <= np.abs(xmax - xmin) / 2. - 2. * tolx) & (np.abs(y - (ymax + ymin) / 2.) <= np.abs(ymax - ymin) / 2. - 2. * toly)
        
        # find indices of brightest objects in descending order
        idxflux = np.argsort(z[maskout])[::-1]
    
        # select only objects well inside, sorted by flux in descending order
        xsel = x[maskout][idxflux]
        ysel = y[maskout][idxflux]
        zsel = z[maskout][idxflux]
        
        # remove points in crowded regions
        isolated = np.ones(len(xsel), dtype = bool)
        for i in range(len(xsel)):
            if not isolated[i]:
                continue
            dist = np.sqrt((xsel - xsel[i])**2 + (ysel - ysel[i])**2)
            dist[i] = error
            idxmin = np.argmin(dist)
            if dist[idxmin] < error:
                isolated[i] = False
                isolated[idxmin] = False
        
        # maximum number requested
        N = min(N, np.sum(isolated))
    
        return xsel[isolated][0:N], ysel[isolated][0:N]
        
    # distance between two sets
    def xydistance(self, x1, x2, y1, y2, delta):
    
        nsources = 0
        total = 0
    
        for isource in range(len(x2)):
            
            distx = (x2[isource] - x1)
            disty = (y2[isource] - y1)
            dist = distx * distx + disty * disty
            idx = np.argmin(dist)
    
            if dist[idx] < delta**2:
                nsources += 1
                total += dist[idx]
        
        return nsources, total

    # function that tries to find first rough astrometric solution by brute force
    def bruteforceastro(self, x1, x2, y1, y2, deltaxmin, deltaxmax, deltaymin, deltaymax, delta):

        ibest = 0
        jbest = 0
        nbest = 0
    
        for i in np.arange(deltaxmin, deltaxmax, delta):
            
            for j in np.arange(deltaymin, deltaymax, delta):
    
                (nsources, dist) = self.xydistance(x1, x2 + i, y1, y2 + j, delta)
                if nsources >= nbest:
                    ibest = i
                    jbest = j
                    nbest = nsources
                    #print ibest, jbest, nsources, dist
        
        return ibest, jbest

    # match two pixel based catalogues from the same CCD by their pixel coordinates
    def matchpix(self, **kwargs):
        
        pixcat = kwargs["pixcat"]
        nx = kwargs["nx"]
        ny = kwargs["ny"]
        bruteforce = False
        doplot = False
        outdir = '.'
        outname = 'test'
        verbose = False
        if "bruteforce" in kwargs.keys():
            bruteforce = kwargs["bruteforce"]
        if "doplot" in kwargs.keys():
            doplot = kwargs["doplot"]
        if "outdir" in kwargs.keys():
            outdir = kwargs["outdir"]
        if "outname" in kwargs.keys():
            outname = kwargs["outname"]
        if "verbose" in kwargs.keys():
            verbose = kwargs["verbose"]
            
        if verbose:
            print "Matching sets of stars to pixel catalogue, checking catalogue requirements"

        if self.xunit != 'pix' or self.yunit != 'pix' or pixcat.xunit != 'pix' or pixcat.yunit != 'pix': 
            print "Inconsistent attributes to do coordinate matching"
            return False

        nstarmin = 100.
        rcrowd = np.sqrt(nx * ny / np.sqrt(len(self.x) * len(pixcat.x)) / np.pi)
        flux1min = np.percentile(self.z, 100. * (1. - nstarmin / len(self.z)))
        flux1max = min(1e6, np.percentile(self.z, 100))
        flux2min = np.percentile(pixcat.z, 100. * (1. - nstarmin / len(pixcat.z)))
        flux2max = min(1e6, np.percentile(pixcat.z, 100))

        if verbose:
            print "Flux cuts based on expected number of stars:"
            print "   flux1min %f, flux1max %f, flux2min %f, flux2max %f" % (flux1min, flux1max, flux2min, flux2max)
            
            print "%i and %i stars before flux cut" % (len(self.x), len(self.x)) 
    
        mask = (self.z > flux1min) & (self.z < flux1max)
        x1 = self.x[mask]; y1 = self.y[mask]
        r1 = self.r[mask]
        z1 = self.z[mask]; e_z1 = self.e_z[mask]
        
        mask = (pixcat.z > flux2min) & (pixcat.z < flux2max)
        x2 = pixcat.x[mask]; y2 = pixcat.y[mask]
        r2 = pixcat.r[mask]
        z2 = pixcat.z[mask]; e_z2 = pixcat.e_z[mask]
    
        if verbose:
            print "%i and %i stars after flux cut" % (len(x1), len(x2)) 
    
        # select the brightest stars only from each set until having a similar number of elements
        idxz1 = np.argsort(z1)[::-1]
        idxz2 = np.argsort(z2)[::-1]
        if len(z1) > 1.2 * len(z2):
            naux = int(1.2 * len(z2))
            x1 = x1[idxz1[:naux]]
            y1 = y1[idxz1[:naux]]
            r1 = r1[idxz1[:naux]]
            e_z1 = e_z1[idxz1[:naux]]
            z1 = z1[idxz1[:naux]]
        elif len(z2) > 1.2 * len(z1):
            naux = int(1.2 * len(z1))
            x2 = x2[idxz2[:naux]]
            y2 = y2[idxz2[:naux]]
            r2 = r2[idxz2[:naux]]
            e_z2 = e_z2[idxz2[:naux]]
            z2 = z2[idxz2[:naux]]
    
        if doplot:
            if verbose:
                print "%i and %i stars after normalization cut" % (len(x1), len(x2)) 
            fig, ax = plt.subplots()#nrows = 2, figsize = (21, 14))
            ax.scatter(y1, x1, marker = 'o', c = 'r', s = 10, alpha = 0.5, edgecolors = 'none')
            ax.scatter(y2, x2, marker = '*', c = 'b', s = 10, alpha = 0.5, edgecolors = 'none')
            ax.axvline(30)
            ax.axvline(ny - 30)
            ax.axhline(30)
            ax.axhline(nx - 30)
            ax.set_ylim(0, nx)
            ax.set_xlim(0, ny)
            plt.savefig("%s/%s_00.png" % (outdir, outname))
     
        # first select only sources not in crowded regions and far from the edges
        tolx = 100; toly = 100;
        (x1s, y1s) = self.select(x1, y1, z1, tolx, toly, 0, nx, 0, ny, rcrowd, nstarmin)
        (x2s, y2s) = self.select(x2, y2, z2, tolx / 2., toly / 2., 0, nx, 0, ny, rcrowd, nstarmin)

        if doplot:
            if verbose:
                print "%i and %i stars selected by select routine" % (len(x1s), len(x2s)) 
            fig, ax = plt.subplots()
            ax.scatter(y1s, x1s, marker = 'o', edgecolors = 'none', c = 'r', s = 10, alpha = 0.5)
            ax.scatter(y2s, x2s, marker = '*', edgecolors = 'none', c = 'b', s = 10, alpha = 0.5)
            ax.set_title("First selection", fontsize = 8)
            ax.axvline(30)
            ax.axvline(ny - 30)
            ax.axhline(30)
            ax.axhline(nx - 30)
            ax.set_ylim(0, nx)
            ax.set_xlim(0, ny)
            plt.savefig("%s/%s_01.png" % (outdir, outname))

        if len(x1s) == 0 or len(x2s) == 0:
            print "   ---> WARNING, no matching stars found."
            return False

        
        if bruteforce:
    
            # Use brute force approach to find first guess

            ibest = 0
            jbest = 0
            
            if verbose:
                print "Refining solution to 25 pixels..."
            (ibest, jbest) = self.bruteforceastro(x1, x2, y1, y2, ibest - 500, ibest + 500, jbest - 500, jbest + 500, 25)
            
            if verbose:
                print ibest, jbest
            
            if verbose:
                print "Refining solution to 5 pixels..."
            (ibest, jbest) = self.bruteforceastro(x1, x2, y1, y2, ibest - 25, ibest + 25, jbest - 25, jbest + 25, 5)
            
            if verbose:
                print ibest, jbest
            
            if verbose:
                print "Refining solution to 2 pixels..."
            (ibest, jbest) = self.bruteforceastro(x1, x2, y1, y2, ibest - 5, ibest + 5, jbest - 5, jbest + 5, 2)
            
            if verbose:
                print ibest, jbest
    
            deltax = ibest
            deltay = jbest
    
        else:
    
            # find median separation in x and y
            sepx = []
            sepy = []
            for i in range(len(x1s)):
                dist = np.sqrt((x1s[i] - x2s)**2 + (y1s[i] - y2s)**2)
                idxmin = np.argmin(dist)
                sepx.append(x1s[i] - x2s[idxmin])
                sepy.append(y1s[i] - y2s[idxmin])
            sepx = np.array(sepx)
            sepy = np.array(sepy)

            # find the highest concentration
            nsep = np.zeros(len(sepx), dtype = int)
            for i in range(len(sepx)):
                nsep[i] = np.sum(np.sqrt((sepx - sepx[i])**2 + (sepy - sepy[i])**2 < 5))
            idxnsep = np.argmax(nsep)
            sepxcomp = sepx[idxnsep]
            sepycomp = sepy[idxnsep]
            mask = (nsep >= max(nsep) - 1)
            if np.sum(mask) > 1:
                sepxcomp = np.median(sepx[mask])
                sepycomp = np.median(sepy[mask])
    
            if doplot:
                fig, ax = plt.subplots(figsize = (12, 6))
                ax.scatter(sepy, sepx, marker = '.', edgecolors = 'none', c = 'b', s = 5)
                ax.scatter(sepycomp, sepxcomp, marker = 'o', facecolors = 'none', s = 100)
                ax.set_title("Minimum separations", fontsize = 8)
                plt.savefig("%s/%s_02.png" % (outdir, outname))
        
            # 0th order correction
            deltax = sepx[idxnsep] #np.median(sepx)
            deltay = sepy[idxnsep] #np.median(sepy)
    
        # mask stars on the edges and apply correction
        mask1 = (x1 > tolx) & (x1 < nx - tolx) & (y1 > toly) & (y1 < ny - toly)
        mask2 = (x2 > tolx) & (x2 < nx - tolx) & (y2 > toly) & (y2 < ny - toly)
        x1 = x1[mask1]
        y1 = y1[mask1]
        z1 = z1[mask1]
        x2 = x2[mask2]
        y2 = y2[mask2]
        z2 = z2[mask2]
        x2 = x2 + deltax
        y2 = y2 + deltay
    
        if doplot:
            if verbose:
                print "Stars after median correction: %i" % len(x2)
            fig, ax = plt.subplots(figsize = (12, 6))
            ax.scatter(y1, x1, marker = 'o', edgecolors = 'none', c = 'r', s = 10, alpha = 0.5)
            ax.scatter(y2, x2, marker = '*', edgecolors = 'none', c = 'b', s = 10, alpha = 0.5)
            ax.set_title("Median distance correction", fontsize = 8)
            plt.savefig("%s/%s_03.png" % (outdir, outname))
    
        # select pairs of points that are farther than error / 2 to a source in the second image after median correction
        sel1 = np.zeros(len(x1), dtype = bool)
        sel2 = np.zeros(len(x2), dtype = bool)
        idx1 = []
        idx2 = []
        for i in range(len(x1)):
            if sel1[i]:
                continue
            dist = np.sqrt((x1[i] - x2)**2 + (y1[i] - y2)**2)
            idxmin = np.argmin(dist)
            if sel2[idxmin]:
                continue
            if dist[idxmin] < rcrowd / 2.:
                idx1.append(i)
                idx2.append(idxmin)
                sel1[i] = True
                sel2[idxmin] = True
    
        if idx1 == []:
            return None
    
        idx1 = np.array(idx1)
        idx2 = np.array(idx2)
    
        x1 = x1[idx1]
        y1 = y1[idx1]
        r1 = r1[idx1]
        z1 = z1[idx1]
        e_z1 = e_z1[idx1]
        x2 = x2[idx2]
        y2 = y2[idx2]
        r2 = r2[idx2]
        z2 = z2[idx2]
        e_z2 = e_z2[idx2]
    
        if doplot:
            if verbose:
                print "Stars after distance matching: %i" % len(x1)
            fig, ax = plt.subplots(figsize = (12, 6))
            ax.scatter(y1, x1, marker = 'o', edgecolors = 'none', c = 'r', s = 10, alpha = 0.5)
            ax.scatter(y2, x2, marker = '*', edgecolors = 'none', c = 'b', s = 10, alpha = 0.5)
            ax.set_title("Distance matching", fontsize = 8)
            plt.savefig("%s/%s_04.png" % (outdir, outname))
    
            fig, ax = plt.subplots()
            ax.scatter(y1 - y2, x1 - x2, marker = 'o', edgecolors = 'none', c = 'r', s = 10, alpha = 0.5)
            ax.axvline(0)
            ax.axhline(0)
            ax.set_title("Differences after matching", fontsize = 8)
            plt.savefig("%s/%s_05.png" % (outdir, outname))
    
    
        # find new highest concentration
        nsep = np.zeros(len(x1))
        for i in range(len(x1)):
            nsep[i] = np.sum((np.sqrt((x1[i] - x2[i])**2 + (y1[i] - y2[i])**2) < 5))
        idxnsep = np.argmax(nsep)
        deltasepxcomp = np.median((x1 - x2)[nsep == max(nsep)])
        deltasepycomp = np.median((y1 - y2)[nsep == max(nsep)])
    
        # select matched sources, removing outliers
        dist = np.sqrt((x1 - x2 - deltasepxcomp)**2 + (y1 - y2 - deltasepycomp)**2)
        distmask = (dist < 2.)  # 5 * pixscale
        order = 2
        if order == 1:
            nptmin = 3
        elif order == 2:
            nptmin = 6
        elif order == 3:
            nptmin = 10
        if np.sum(distmask) < nptmin:
            distmask = (dist < 10.)
        
        x1 = x1[distmask]
        y1 = y1[distmask]
        r1 = r1[distmask]
        z1 = z1[distmask]
        e_z1 = e_z1[distmask]
        x2 = x2[distmask] - deltax
        y2 = y2[distmask] - deltay
        r2 = r2[distmask]
        z2 = z2[distmask]
        e_z2 = e_z2[distmask]
    
        if doplot:
            if verbose:
                print "Stars after distance filtering: %i" % np.sum(distmask)
    
            fig, ax = plt.subplots(figsize = (12, 6))
            ax.scatter(y1, x1, marker = 'o', edgecolors = 'none', c = 'r', s = 10, alpha = 0.5)
            ax.scatter(y2, x2, marker = '*', edgecolors = 'none', c = 'b', s = 10, alpha = 0.5)
            ax.set_title("Positions after matching and filtering", fontsize = 8)
            plt.savefig("%s/%s_06.png" % (outdir, outname))
    
            fig, ax = plt.subplots()
            ax.scatter(y1 - y2, x1 - x2, marker = '.', edgecolors = 'none', c = 'b', s = 10, alpha = 0.5)
            ax.scatter(deltasepycomp + deltay, deltasepxcomp + deltax, marker = 'o', s = 100, facecolors = 'none')
            ax.set_title("Differences after matching and filtering", fontsize = 8)
            plt.savefig("%s/%s_07.png" % (outdir, outname))
    
        if verbose:
            print "Number of star coincidences: ", len(x1), len(y1)
        # find best transformation relating all these points
        self.pol = polytransform(2)
        self.pol.solve(x1, y1, x2, y2)

        # find rms
        (x1t, y1t) = self.pol.apply(x1, y1)
        rms = np.sqrt(np.sum((x1t - x2)**2 + (y1t - y2)**2) / len(x1))
        if verbose:
            print "rms: ", rms

        if doplot:
            if verbose:
                print "Stars after polynomial fit"
            fig, ax = plt.subplots()
            ax.scatter(y1t - y1, x1t - x1, marker = '.', edgecolors = 'none', c = 'b', s = 10, alpha = 0.5)
            ax.set_title("Differences after order %i polynomial fit (rms: %f pix)" % (self.pol.order, rms), fontsize = 8)
            plt.savefig("%s/%s_08.png" % (outdir, outname))

        # apply transformation to all points and find final mask
        (x1t, y1t) = self.pol.apply(self.x, self.y)

        # find final masks
        idxmatch = np.array(map(lambda i, j: np.argmin((pixcat.x - i)**2 + (pixcat.y - j)**2), x1t, y1t))
        distmin = np.sqrt((pixcat.x[idxmatch] - x1t)**2 + (pixcat.y[idxmatch] - y1t)**2)
        if verbose:
            print "MAD", np.median(np.abs(np.median(distmin) - distmin))
        deltaedge = 50
        SNRlim = 8
        maskfinal = (distmin - np.median(distmin) < min(2., np.median(np.abs(np.median(distmin) - distmin)))) & (pixcat.z[idxmatch] / pixcat.e_z[idxmatch] > SNRlim) & (self.z / self.e_z > SNRlim) & (self.x > deltaedge) & (self.x < nx - deltaedge) & (self.y > deltaedge) & (self.y < ny - deltaedge)
        self.sidx = self.indices[maskfinal]
        pixcat.sidx = idxmatch[self.sidx]

        if doplot:
            if verbose:
                print "New selection after polynomial fit"
            fig, ax = plt.subplots()
            ax.scatter(pixcat.x[pixcat.sidx] - self.x[self.sidx], pixcat.y[pixcat.sidx] - self.y[self.sidx], marker = '.', edgecolors = 'none', c = 'b', s = 10, alpha = 0.5)
            ax.set_title("Differences after order %i polynomial fit (rms: %f pix)" % (self.pol.order, rms), fontsize = 8)
            plt.savefig("%s/%s_09.png" % (outdir, outname))

        # find final solution
        self.pol.solve(self.x[self.sidx], self.y[self.sidx], pixcat.x[pixcat.sidx], pixcat.y[pixcat.sidx])

        # find rms
        (x1t, y1t) = self.pol.apply(self.x[self.sidx], self.y[self.sidx])
        delta = np.sqrt((x1t - pixcat.x[pixcat.sidx])**2 + (y1t - pixcat.y[pixcat.sidx])**2)
        rms = np.sqrt(np.sum((x1t - pixcat.x[pixcat.sidx])**2 + (y1t - pixcat.y[pixcat.sidx])**2) / len(x1t))
        if verbose:
            print "rms: ", rms
            
        if doplot:
            if verbose:
                print "New selection after polynomial fit"
            fig, ax = plt.subplots()
            map(lambda x1, y1, x2, y2, c: ax.plot([x1, y1], [x2, y2], c = 'b', alpha = 0.5), pixcat.x[pixcat.sidx], self.x[self.sidx], pixcat.y[pixcat.sidx], self.y[self.sidx], delta)
            ax.set_title("Differences after order %i polynomial fit (rms: %f pix)" % (self.pol.order, rms), fontsize = 8)
            plt.savefig("%s/%s_10.png" % (outdir, outname))

            fig, ax = plt.subplots()
            ax.scatter(pixcat.x[pixcat.sidx] - x1t, pixcat.y[pixcat.sidx] - y1t, marker = '.', edgecolors = 'none', c = 'b', s = 10, alpha = 0.5)
            ax.set_title("Differences after order %i polynomial fit (rms: %f pix)" % (self.pol.order, rms), fontsize = 8)
            plt.savefig("%s/%s_11.png" % (outdir, outname))

        # do one more filter
        self.sidx = self.sidx[delta < 0.7]
        pixcat.sidx = pixcat.sidx[delta < 0.7]

        # find final solution
        self.pol.solve(self.x[self.sidx], self.y[self.sidx], pixcat.x[pixcat.sidx], pixcat.y[pixcat.sidx])

        # find rms
        (x1t, y1t) = self.pol.apply(self.x[self.sidx], self.y[self.sidx])
        delta = np.sqrt((x1t - pixcat.x[pixcat.sidx])**2 + (y1t - pixcat.y[pixcat.sidx])**2)
        rms = np.sqrt(np.sum((x1t - pixcat.x[pixcat.sidx])**2 + (y1t - pixcat.y[pixcat.sidx])**2) / len(x1t))
        if verbose:
            print "rms: ", rms

        if doplot:
            if verbose:
                print "Final selection after polynomial fit"
            fig, ax = plt.subplots()
            map(lambda x1, y1, x2, y2, c: ax.plot([x1, y1], [x2, y2], c = 'b', alpha = 0.5), pixcat.x[pixcat.sidx], self.x[self.sidx], pixcat.y[pixcat.sidx], self.y[self.sidx], delta)
            ax.set_title("Differences after order %i polynomial fit (rms: %f pix)" % (self.pol.order, rms), fontsize = 8)
            ax.set_xlim(0, nx)
            ax.set_ylim(0, ny)
            plt.savefig("%s/%s_final_pos.png" % (outdir, outname))

            fig, ax = plt.subplots()
            ax.scatter(pixcat.x[pixcat.sidx] - x1t, pixcat.y[pixcat.sidx] - y1t, marker = '.', edgecolors = 'none', c = 'b', s = 10, alpha = 0.5)
            ax.set_title("Differences after order %i polynomial fit (rms: %f pix)" % (self.pol.order, rms), fontsize = 8)
            plt.savefig("%s/%s_final_res.png" % (outdir, outname))

            
if __name__ == "__main__":

    print "Testing"

    catdir = "/home/fforster/Work/HiTS/devel/SNHiTS15J"
    field = "Blind15A_25"
    CCD = "S14"
    epoch = 2

    # load sextractor catalogue and get file header before trying to match
    (x, y, z, e_z, r, flag) = np.loadtxt("%s/%s_%s_%02i_image_crblaster.fits-catalogue_wtmap_backsize64.dat" % (catdir, field, CCD, epoch), usecols = (1, 2, 5, 6, 8, 9)).transpose()
    pixcat = catalogue(x = x, y = y, z = z, e_z = e_z, r = r, flag = flag, xlabel = "ipix", ylabel = "jpix", zlabel = "flux", rlabel = "radius", xunit = "pix", yunit = "pix", zunit = "ADU", runit = "pix")
    pixcat.readheader("../SNHiTS15J/%s_%s_%02i_image.fits" % (field, CCD, epoch))
    
    # load USNO data and create catalogue
    fileUSNO = "%s/USNO_%s_%s_%02i.npy" % (catdir, field, CCD, 2)
    if os.path.exists(fileUSNO):
        (x, y, name, B, R) = np.load(fileUSNO)
        RADECmagcat = catalogue(x = x, y = y, z = B, xlabel = "RA", ylabel = "DEC", zlabel = "B mag", xunit = "hr", yunit = "deg", zunit = "mag")
        
        # match both catalogues
        pixcat.matchRADEC(RADECmagcat = RADECmagcat, solveWCS = True, doplot = True, outdir = "astrometry_test", outname = "test")

    ##TEST artificially set the USNO catalogue coordinates to the WCS derived solution
    #RADECmagcat.x[RADECmagcat.sidx] = pixcat.WCSsol.WCS.RA()
    #RADECmagcat.y[RADECmagcat.sidx] = pixcat.WCSsol.WCS.DEC()
  
    # solve astrometric solution for all epochsload 2nd sextractor catalogue
    rmspix = []
    rmsWCS = []
    pixscale = 0.27 #arcsec

    for epoch in range(40):

        # load sextractor catalogue
        fileepoch = "%s/%s_%s_%02i_image_crblaster.fits-catalogue_wtmap_backsize64.dat" % (catdir, field, CCD, epoch)
        if not os.path.exists(fileepoch):
            continue
        (x, y, z, e_z, r, flag) = np.loadtxt(fileepoch, usecols = (1, 2, 5, 6, 8, 9)).transpose()
        pixcat2 = catalogue(x = x, y = y, z = z, e_z = e_z, r = r, flag = flag, xlabel = "ipix", ylabel = "jpix", zlabel = "flux", rlabel = "radius", xunit = "pix", yunit = "pix", zunit = "ADU", runit = "pix")
        pixcat2.readheader("../SNHiTS15J/%s_%s_%02i_image.fits" % (field, CCD, epoch))
        
        # do relative pixel solution
        if epoch == 2:
            continue

        # match both catalogues and find WCS
        pixcat2.matchRADEC(RADECmagcat = RADECmagcat, solveWCS = True, doplot = True, outdir = "astrometry_test", outname = "%02i" % epoch)
        
        # plot residuals of both transformations
        fig, ax = plt.subplots()

        # plot residuals of pixel to RADEC transformation
        ax.scatter(3600 * (pixcat2.WCSsol.WCS.RA() - pixcat2.WCSsol.RADECcat.x[pixcat2.WCSsol.RADECcat.sidx]), 3600 * (pixcat2.WCSsol.WCS.DEC() - pixcat2.WCSsol.RADECcat.y[pixcat2.WCSsol.RADECcat.sidx]), c = 'k', lw = 0, marker = 'o', label = 'USNO', s = 40) 

        # match pixel coordinates
        pixcat.matchpix(pixcat = pixcat2, nx = 2048, ny = 4096, bruteforce = False, doplot = False, outdir = "pol_test", outname = "%02i" % epoch)
        pol = polytransform(3)
        pol.solve(pixcat.x[pixcat.sidx], pixcat.y[pixcat.sidx], pixcat2.x[pixcat2.sidx], pixcat2.y[pixcat2.sidx])
        (x1t, y1t) = pol.apply(pixcat.x[pixcat.sidx], pixcat.y[pixcat.sidx])
        rmspix.append(np.sqrt(np.sum((x1t - pixcat2.x[pixcat2.sidx])**2 + (y1t - pixcat2.y[pixcat2.sidx])**2) / len(x1t)) * pixscale)
        print "Epoch %i to epoch %i rms: %f arcsec" % (epoch, 2, rmspix[-1])

        # plot residuals between pixel solutions and between RADEC solutions
        pixcat.WCSsol.WCS.ijcat.sidx = pixcat.sidx
        pixcat2.WCSsol.WCS.ijcat.sidx = pixcat2.sidx
        rmsWCS.append(3600. * np.sqrt(np.sum((pixcat.WCSsol.WCS.RA() - pixcat2.WCSsol.WCS.RA())**2 + (pixcat.WCSsol.WCS.DEC() - pixcat2.WCSsol.WCS.DEC())**2) / len(pixcat.sidx)))
        print "WCS rms: %f arcsec" % rmsWCS[-1]
        ax.scatter(pixscale * (x1t - pixcat2.x[pixcat2.sidx]), pixscale * (y1t - pixcat2.y[pixcat2.sidx]), c = 'r', lw = 0, marker = '.', label = 'pix', alpha = 0.5) 
        ax.scatter(3600. * (pixcat.WCSsol.WCS.RA() - pixcat2.WCSsol.WCS.RA()), 3600. * (pixcat.WCSsol.WCS.DEC() - pixcat2.WCSsol.WCS.DEC()), c = 'b', lw = 0, marker = '.', label = 'WCS', alpha = 0.5) 
        ax.axvline(0)
        ax.axhline(0)
        ax.legend(loc = 2, fontsize = 8, framealpha = 0.5)
        ax.set_xlabel("Delta RA [arcsec]")
        ax.set_ylabel("Delta DEC [arcsec]")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        plt.savefig("astrometry_test/test_%02i.png" % epoch)
        
    # plot rms
    rmspix = np.array(rmspix)
    rmsWCS = np.array(rmsWCS)
    fig, ax = plt.subplots()
    ax.hist(rmspix, color = 'r', label = 'pix', alpha = 0.5)
    ax.hist(rmsWCS, color = 'b', label = 'WCS', alpha = 0.5)
    ax.set_xlabel("rms [arcsec]")
    ax.set_ylabel("N")
    ax.legend(loc = 2, fontsize = 8, framealpha = 0.5)
    plt.savefig("rms.png")
    
