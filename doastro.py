#!/usr/bin/python2.7

import os
import re # use regular patterns
import sys, getopt # system commands
import string # string functions4
import math
import numpy as np # numerical tools
from scipy import linalg as scipylinalg
from scipy import stats
from scipy.ndimage import filters
from scipy import signal as scipysignal
from pylab import *
#from mx.DateTime import * # date conversion
import datetime
import time
from multiprocessing import Array, Pool
from PosHuber import PosHuber
from sklearn import linear_model

import pickle
import json

from astropy.table import Table # Virtual Observatory Tables

def printtime(context):
    now = datetime.datetime.now()
    print "\n\n   BENCHMARK %s" % context, now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond
#    now = DateTime(now.year, now.month, now.day, now.hour, now.minute, now.second + 1e-6 * now.microsecond) 
#    print "   ---TIME MJD--- %s ---" % context, now.mjd
    print "\n"

######## WARNINGS ##################
#doastro.py
#1: badly formatted arguments or help
#2: "\n\nWARNING: reference and science file are the same, try different numbers.\n\n"
#3: "\n\nWARNING: Need filesci variable\n\n"
#4: "\n\nWARNING: Need filesci variable\n\n"
#5: "\n\nWARNING: File %s_image.fits.fz does not exist\n\n" % file1
#6: "\n\nWARNING: File %s_image.fits.fz does not exist\n\n" % file2
#7: "\n\nWARNING: File %s_image_crblaster.fits does not exist\n\n" % file1
#8: "\n\nWARNING: File %s_image_crblaster.fits does not exist\n\n" % file2
#9: "\n\nWARNING: File %s/domeflat/%s/domeflat_%s_master_mask.pkl does not exist\n\n" % (sharedir, CCD, CCD)
#10: "\n\nWARNING: File %s_wtmap.fits.fz does not exist\n\n" % file1
#11: "\n\nWARNING: File %s_wtmap.fits.fz does not exist\n\n" % file2
#12: "\n\nWARNING: Cannot find field 'MJD-OBS'\n\n"
#13: "\n\nWARNING: mask size is not the same as the image size\n\n"
#14: "WARNING: File %s does not exist, EXIT" % usnofile
#15: "\n\nWARNING: Not enough stars to do astrometric solution...\n\n"
#16: "\n\nWARNING: Error solving linear system when matching pixel coordinate systems\n\n"
#17: "\n\nWARNING: Cannot find stars to load\n\n"
#18: "\n\nWARNING: Cannot find astrometric solution in field %s, CCD %s, between epochs %i and %i.\n\n" % (field, CCD, fileref, filesci)
#19: "\n\nWARNING: Not enough stars or rms of astrometric solution too big (%i objects, rms: %f pixels)\n\n" % (len(x1sel), rms)
#20: "WARNING: No celestial astrometric solution found"
#21: "\n\nWARNING: Not enough reference stars to do PCA\n\n"
#22: "\n\nWARNING: it appears that the --dofilter option was used after --dosubtraction without the --candidates option. Run again including the --candidates option\n\n"
#look.py:
#23:  "ERROR with MPChecker coordinates"
#24: "Strange RA coordinates in Simbad object, please send output to francisco.forster@gmail.com"
#25: "Strange DEC coordinates in Simbad object, please send output to francisco.forster@gmail.com"
#26: \n\nWARNING: Cannot find psf file <file>
#27: \n\nWARNING: science image filter does not match reference image filter
#28: \n\nWARNING: Too many pixels to sort, probably a bad subtraction
#
# ###########################################################

# print the local time
printtime("start")

# fits files
import pyfits as fits

# date and time
import datetime

# lomb scargle periodogram
import lomb

# do webpage
from doweb import *

# public catalogues
from look import *

# astrometric solution mode
astrometry = 'pix2pix' #'WCS' # 'pix2pix'

# own fortran routines
from convolution import convolution
from optimalphotometry import optimalphotometry
if astrometry == 'pix2pix':
    from projection import projection
elif astrometry == 'WCS':
    from projectionWCS import projectionwcs

# default values
# ----------------------------------------------------------

# radians
deg2rad = np.pi / 180.
rad2deg = 180. / np.pi


# plotting and html generation options
doplottimeseriesflux = False
doplotnegatives = False
doplottimeseriesimages = True
doplotperiodogram = False
doplotcandidates = True
doplotPCA = True
dohtml = False

# use feature pairs
dorfcpair = False

# default file, CCD
field = 'Blind_03'
CCD = 'N4'

# default number of cores
ncores = 4

# default output directory
refdir =   "/home/fforster/Work/CMMPIPE/DATA_CMMPIPE"
indir =    "/home/fforster/Work/CMMPIPE/DATA_CMMPIPE"
outdir =   "/home/fforster/Work/CMMPIPE/DATA_CMMPIPE"
sharedir = "/home/fforster/Work/CMMPIPE/DATA_CMMPIPE"
webdir =   "/home/fforster/Work/CMMPIPE/DATA_CMMPIPE"

# data quality mask threshold
dqth = 128

# make image minus sky zero where the error is too large
dozeromask = False

# save filter training stars, default
savestars = True

# do catalalogue search, default
dolook = False

# do crblaster, default
docrblaster = False

# default crblaster location
crblaster = ""#/home/fforster/Work/CMMPIPE/COSMICRAYS/crblaster/crblaster"

# default reference file and file to process
fileref = 1
filesci = 0

# use weights, default
useweights = 'external'
e_read = 10. # readout error in case no weights are used (it seems to move between 5 and 10 in real weight maps)

# do subtraction
dodiff = False

# add stars artificially, default
doadd = False

# do candidate features
docandidates = False

# use random forest classifier (instead of arbitrary cuts), default
doML = False

# do candidates filtering, default
dofilter = False

# do candidates filtering, default
dorevisit = False

# verbosity, default
verbose = False

# projection transformation order
order = 2

# psf peak fraction above which to do optimal photometry 
fcut = 0.2

# save only positive, repeated candidates in json
jsononlypositiverepeated = True

# save SNR image
saveSNR = True

# save convolved image
saveconv = True

# save optimal photometry results
saveoptphot = False

# save fake candidate information even after filter
savefakes = False

# check SNR histogram and correct detection SNR limit if necessary
docheckSNRhisto = False

# add stars to reference
addstars2ref = False

# background estimation
backgroundtype = 'sextractor' # 'median'
backsize = 64 # window size used to estimate background
forcebackground = False # force background computation

# detection probability threshold
prob_threshold = 0.5 # 0.7
probmode = 'max_prob' # 'DL'

# use data quality mask
dodqmask = True

# first order astrometric solution
deltax = 0
deltay = 0


# command line options
try:
    opts, args = getopt.getopt(sys.argv[1:], "hf:c:r:p:n:R:I:O:S:W:b:w:LsacFfRv", ["help", "field=", "CCD=", "reference=", "science=", "ncores=", "refdir=", "indir=", "outdir=", "sharedir=", "webdir=", "crblaster=", "weights=", "look", "subtraction", "addstars", "candidates", "loadrandomforest", "filter", "revisit", "verbose"])
except getopt.GetoptError:
    print 'doastro.py --help'
    sys.exit(1)
for opt, arg in opts:
    if opt in ('-h', '--help'):
        print "\nCC by F.Forster\nDo image matching, source extraction, astrometric solution, projection, convolution, difference, candidate selection and statistics, candidate filtering"
        print 'Usage: doastro.py --field <field#> --CCD <CCD#> --reference <fileref> --science <filesci> --ncores <ncores#> --refdir <refdir> --outdir <outdir> --outdir <outdir> --sharedir <sharedir> --crblaster <crblasterloc> [--look] [--weights] --subtraction --candidates --filter --verbose]\n'
        print '-h --help: this message'
        print '-f <field#> --field <field#>: the field string (e.g. Blind_01)'
        print '-c <CCD#> --CCD <CCD#>: the CCD string (e.g. S20)'
        print '-r <fileref> --reference <fileref>: reference epoch for image projection and subtraction'
        print '-p <filesci> --science <filesci>: new epoch to process'
        print '-n <ncores#> --ncores <ncores#>: number of cores to use in projection and convolution routines'
        print '-R <refdir> --refdir <refdir>: directory where the reference files are stored'
        print '-R <indir> --indir <indir>: directory where the files to process are stored'
        print '-O <outdir> --outdir <outdir>: directory where to save processed fits files'
        print '-S <sharedir> --sharedir <sharedir>: directory where fast access products are stored'
        print '-W <webdir> --webdir <webdir>: directory where webpage will be created'
        print '-b <crblaster> --crblaster <loc>: how to call crblaster via command line'
        print '-L --look: do a detached catalogue search before crblaster'
        print '-w <weightmode> --weights <weightmode>: use no weights (none), custom made mask (own), determined by own copy of DECam community pipeline (internal), or provided by a external copy of the DCP (external)'
        print '-s --subtraction: do image subtraction (includes matching, astrometric solution, projection, convolution and difference)'
        print '-a --addstars: add stars artificially to new locations (used for training classifier)'
        print '-c --candidates: compute candidate features'
        print '-f --filter: filter candidates'
        print '-R --revisit: revisit all difference images of a candidate present in more than one image, using available data'
        print '\ne.g. python doastro.py --field Blind15A_25 --CCD N1 --reference 2 --science 4 --ncores 4 --refdir /home/apps/astro/DATA --indir /home/apps/astro/DATA --outdir /home/apps/astro/DATA --sharedir /home/apps/astro/SHARED --webdir /home/apps/astro/WEB --weights internal --look --subtraction --addstars --candidates --loadrandomforest --filter --revisit --verbose'
        sys.exit(1)
    elif opt in ('-f', '--field'):
        field = arg
    elif opt in ('-c', '--CCD'):
        CCD = arg
    elif opt in ('-r', '--reference'):
        fileref = int(arg)
    elif opt in ('-p', '--science'):
        filesci = int(arg)
    elif opt in ('-n', '--ncores'):
        ncores = int(arg)
    elif opt in ('-R', '--refdir'):
        refdir = arg
    elif opt in ('-I', '--indir'):
        indir = arg
    elif opt in ('-O', '--outdir'):
        outdir = arg
    elif opt in ('-S', '--sharedir'):
        sharedir = arg
    elif opt in ('-W', '--webdir'):
        webdir = arg
    elif opt in ('-b', '--crblaster'):
        crblaster = arg
        docrblaster = True
    elif opt in ('-w', '--weights'):
        useweights = arg
    elif opt in ('-L', '--look'):
        dolook = True
    elif opt in ('-s', '--subtraction'):
        dodiff = True
    elif opt in ('-a', '--addstars'):
        doadd = True
    elif opt in ('-c', '--candidates'):
        docandidates = True
    elif opt in ('-F', '--loadrandomforest'):
        doML = True
    elif opt in ('-f', '--filter'):
        dofilter = True
    elif opt in ('-R', '--revisit'):
        dorevisit = True
    elif opt in ('-v', '--verbose'):
        verbose = True

print "\ndoastro.py (CC F.Forster)\n\nfield: %s\nCCD: %s\nfileref: %i\nfilesci: %i\nncores: %i\nrefdir: %s\nindir: %s\noutdir: %s\nsharedir: %s\nwebdir: %s\ncrblaster: %s\ndolook: %s\nuseweights: %s\ndodiff: %s\ndoadd: %s\ndocandidates: %s\ndoML: %s\ndofilter: %s\ndorevisit: %s\nverbose: %s\n" % (field, CCD, fileref, filesci, ncores, refdir, indir, outdir, sharedir, webdir, crblaster, dolook, useweights, dodiff, doadd, docandidates, doML, dofilter, dorevisit, verbose)

# HITSDIR environmental variable
hitsdir = os.environ['HITSPIPE']

# disable all plots if training
if doadd:
    doplottimeseriesflux = False
    doplotnegatives = False
    doplottimeseriesimages = False
    doplotperiodogram = False
    doplotcandidates = False
    doplotPCA = False
    dohtml = False
    
    statsfile = open("%s/%s/%s/fakestars_%s_%s_%02i-%02i.txt" % (sharedir, field, CCD, field, CCD, filesci, fileref), 'w')


# check that comparison files are different
if fileref == filesci and fileref != 0:
    print "\n\nWARNING: reference and science file are the same, try different numbers.\n\n"
    sys.exit(2)

if not (dofilter or dorevisit) and filesci == 0:
    print "\n\nWARNING: Need filesci variable\n\n"
    sys.exit(3)
if (dodiff or docandidates) and filesci == 0:
    print "\n\nWARNING: Need filesci variable\n\n"
    sys.exit(4)

# check that comparison files exist
file1 = "%s/%s/%s/%s_%s_%02i" % (refdir, field, CCD, field, CCD, fileref)
file2 = "%s/%s/%s/%s_%s_%02i" % (indir, field, CCD, field, CCD, filesci)
file1cat = file1.replace(refdir, sharedir)
file2cat = file2.replace(indir, sharedir)

if filesci != 0:
    if docrblaster and not os.path.exists("%s_image.fits.fz" % file1):
        print "\n\nWARNING: File %s_image.fits.fz does not exist\n\n" % file1
        sys.exit(5)
    if docrblaster and not os.path.exists("%s_image.fits.fz" % file2):
        print "\n\nWARNING: File %s_image.fits.fz does not exist\n\n" % file2
        sys.exit(6)
    if not docrblaster and not os.path.exists("%s_image_crblaster.fits" % file1):
        print "\n\nWARNING: File %s_image_crblaster.fits does not exist\n\n" % file1
        sys.exit(7)
    if not docrblaster and not os.path.exists("%s_image_crblaster.fits" % file2):
        print "\n\nWARNING: File %s_image_crblaster.fits does not exist\n\n" % file2
        sys.exit(8)
    if useweights == 'own' and not os.path.exists("%s/domeflat/%s/domeflat_%s_master_mask.pkl" % (sharedir, CCD, CCD)):
        print "\n\nWARNING: File %s/domeflat/%s/domeflat_%s_master_mask.pkl does not exist\n\n" % (sharedir, CCD, CCD)
        sys.exit(9)
    if useweights == 'external' and not os.path.exists("%s_wtmap.fits.fz" % file1):
        print "\n\nWARNING: File %s_wtmap.fits.fz does not exist\n\n" % file1
        sys.exit(10)
    if useweights == 'external' and not os.path.exists("%s_wtmap.fits.fz" % file2):
        print "\n\nWARNING: File %s_wtmap.fits.fz does not exist\n\n" % file2
        sys.exit(11)
    if useweights == 'internal' and not os.path.exists("%s_image.fits" % file1):
        print "\n\nWARNING: File %s_image.fits does not exist\n\n" % file1
        sys.exit(10)
    if useweights == 'internal' and not os.path.exists("%s_image.fits" % file2):
        print "\n\nWARNING: File %s_image.fits does not exist\n\n" % file2
        sys.exit(11)

# extract filter
filtername = (fits.open("%s_image_crblaster.fits" % file1)[0].header)["FILTER"]
filtername = filtername[0]
if verbose:
    print "Filter: %s\n" % filtername


# create directories if they do not exist
if not os.path.exists("%s/%s/%s" % (outdir, field, CCD)):
    if verbose:
        print "Creating output directory..."
    os.makedirs("%s/%s/%s" % (outdir, field, CCD))
if not os.path.exists("%s/%s/%s/CALIBRATIONS" % (sharedir, field, CCD)):
    if verbose:
        print "Creating directory to save calibration data..."
    os.makedirs("%s/%s/%s/CALIBRATIONS" % (sharedir, field, CCD))
if not os.path.exists("%s/%s/%s/CANDIDATES" % (sharedir, field, CCD)):
    if verbose:
        print "Creating directory to save candidates data..."
    os.makedirs("%s/%s/%s/CANDIDATES" % (sharedir, field, CCD))
if not os.path.exists("%s/%s/%s/CALIBRATIONS" % (webdir, field, CCD)):
    if verbose:
        print "Creating directory to save calibration data..."
    os.makedirs("%s/%s/%s/CALIBRATIONS" % (webdir, field, CCD))
if not os.path.exists("%s/%s/%s/CANDIDATES" % (webdir, field, CCD)):
    if verbose:
        print "Creating directory to save candidate web information..."
    os.makedirs("%s/%s/%s/CANDIDATES" % (webdir, field, CCD))

# string used for sextractor catalogues
cataloguestring = ''
if useweights == 'external' or useweights == 'internal':
    cataloguestring = '_wtmap_backsize%i' % backsize
elif useweights == 'own':
    cataloguestring = '_own_backsize%i' % backsize


# small tick labels
matplotlib.rc('xtick', labelsize = 7) 
matplotlib.rc('ytick', labelsize = 7) 

# sextractor verbosity
if verbose:
    sexverbose = "FULL"
else:
    sexverbose = "QUIET"

# weight map limit (below this number we throw away the data)
varmax = 1e6

# signal to noise ratio limit (select candidates above this limit for inspection)
SNRlim = 5. #4.

# cross correlation limit (plot candidates above this limit)
cclim = 0.5

# cosmic ray limit (plot candidates below this limit)
CRlim = 30

# maximum simultaneous pixel value limit in reference images (plot candidates whose subtraction images are below this limit in at least one of them, i.e. not a bright stars)
imlim = 25000

# maximum difference in first PCA coefficient after taking absolute value
diffcoefflim = 0.1

# maximum value of symmetry index
symmidxlim = 1.

# maximum number of nearby candidates
ncandlim = 15

# maximum difference between first PCA coefficient and cross correlation
PCA0ccdifflim = 0.08

# number of principal components to consider
nk = 7

# number of field apart from images of candidates to be saved
ninfo = 12

# set number of cores
convolution.set_num_threads(ncores)
optimalphotometry.set_num_threads(ncores)
if astrometry == 'pix2pix':
    projection.set_num_threads(ncores)
elif astrometry == 'WCS':
    projectionwcs.set_num_threads(ncores)
    

# maximum value in input image (everything above this value is set to the background level)
nmax = 60000 # 50000
#nmax = 100000 # 50000

# resampling type
resampling = 'lanczos2'
if resampling[0:7] == 'lanczos':
    alanczos = int(resampling[-1])
else:
    alanczos = 1

# for adding and checking artificial stars
ijadds = np.mgrid[200:2000:200, 400:4000:400]
ijadds = np.dstack(ijadds)
ijadds = np.reshape(ijadds, (np.shape(ijadds)[0]**2, 2))

# load ivar array
nvar = 81
ivarf = np.loadtxt("%s/etc/ivar_%i.dat" % (hitsdir, nvar), dtype = int) - 1  # 9, 57, 81, 289, 625
nf = np.shape(ivarf)[0]

# open CCD numbers file
CCDn = {}
(CCDstring, CCDnumber) = np.loadtxt("%s/etc/CCDnumbers.dat" % hitsdir, dtype = str).transpose()
CCDnumber = np.array(CCDnumber, dtype = int)
for i in range(len(CCDstring)):
    CCDn[CCDstring[i]] = CCDnumber[i]

# open zero point file
(IDzero, filterzero, azero, e_azero, bzero, e_bzero, kzero, e_kzero) = np.loadtxt("%s/etc/zeropoints_%s.txt" % (hitsdir, filtername), dtype = str).transpose()
IDzero = np.array(IDzero, dtype = int)
azero = np.array(azero, dtype = float)
e_azero = np.array(e_azero, dtype = float)
bzero = np.array(bzero, dtype = float)
e_bzero = np.array(e_bzero, dtype = float)
kzero = np.array(kzero, dtype = float)
e_kzero = np.array(e_kzero, dtype = float)

# function to convert fluxes into magnitudes given fluxes and errors in ADU, the CCD number, the exposure time and the airmass of the observation
def ADU2mag(flux, e_flux, CCD, exptime, airmass):
    mag = np.ones(np.shape(flux)) * 30
    mag_1 = np.ones(np.shape(flux)) * 30
    mag_2 = np.ones(np.shape(flux)) * 30
    fluxp = flux + e_flux
    fluxm = flux - e_flux
    mflux = (flux > 0)
    mfluxp = (fluxp > 0)
    mfluxm = (fluxm > 0)
    mag[mflux] = np.array(-2.5 * np.log10(flux[mflux]) + 2.5 * np.log10(exptime) - azero[CCDn[CCD] - 1] - kzero[CCDn[CCD] - 1] * airmass)
    mag_1[mfluxp] = np.array(-2.5 * np.log10(fluxp[mfluxp]) + 2.5 * np.log10(exptime) - azero[CCDn[CCD] - 1] - kzero[CCDn[CCD] - 1] * airmass)
    mag_2[mfluxm] = np.array(-2.5 * np.log10(fluxm[mfluxm]) + 2.5 * np.log10(exptime) - azero[CCDn[CCD] - 1] - kzero[CCDn[CCD] - 1] * airmass)
    return (mag, mag - mag_1, mag_2 - mag)

# prepare ieq2i and ieq2j arrays
npsf = 21 #21 #31
npsf2 = npsf * npsf
ieq2i = np.zeros(npsf2, dtype = int)
ieq2j = np.zeros(npsf2, dtype = int)
for i in range(npsf):
    for j in range(npsf):
        ieq = i * npsf + j
        ieq2i[ieq] = i
        ieq2j[ieq] = j

# routine to select positions where to add stars
def findgalaxies(xgal, ygal, zgal, rgal, fgal, deltagal):

    plotgal = False

    # select only non stars and plot catalogue

    dxgal = np.max(xgal) - np.min(xgal)
    dygal = np.max(ygal) - np.min(ygal)
    nonedge = np.array((xgal < np.max(xgal) - dxgal / 10.) & (xgal > np.min(xgal) + dxgal / 10.) & (ygal < np.max(ygal) - dygal / 10.) & (ygal > np.min(ygal) + dygal / 10.))
    nonstars = np.array((fgal != 0) & (rgal > np.percentile(rgal, 30)) & nonedge)
    dimstars = np.array((zgal < np.percentile(zgal, 90)) & (rgal > np.percentile(rgal, 70)) & nonedge)
    if field[0:8] == 'Blind13A':
        nonstars = np.array(nonstars | dimstars)
    brightstars = np.array((zgal > np.percentile(zgal, 99)) & nonedge)
    
    if plotgal:
        fig, ax = plt.subplots(figsize = (15, 8))

        ax.scatter(xgal[nonstars], ygal[nonstars], marker = 'o', c = 'r', alpha = 0.5, edgecolors = 'none')
        ax.scatter(xgal, ygal, marker = '.', c = 'b', edgecolors = 'none', alpha = 0.5)

    # keep only brightest sources within rgal of 20 pixels and plot them

    sorted = np.argsort(zgal[nonstars])[::-1]
    xgalnew = xgal[nonstars][sorted]
    ygalnew = ygal[nonstars][sorted]
    zgalnew = zgal[nonstars][sorted]

    deltagal = 20

    xgalf = []
    ygalf = []
    mask = np.ones(np.shape(xgalnew), dtype = bool)
    
    for i in range(len(xgalnew)):

        # skip masked objects
        if not mask[i]:
            continue
        
        # mask targets that are too close to a bright star (make sure there is at least one bright star)
        if np.sum(brightstars) >= 1:
            dist2bright = np.min((xgalnew[i] - xgal[brightstars])**2 + (ygalnew[i] - ygal[brightstars])**2)
            if (dist2bright < deltagal**2):
                mask[i] = False
                continue
    
        # save object
        xgalf.append(xgalnew[i])
        ygalf.append(ygalnew[i])
        
        # compute distances from the object to the rest of the selected objects
        dist2 = (xgalnew[i] - xgalnew)**2 + (ygalnew[i] - ygalnew)**2
        
        # mask targets that are too close to our object
        mask[dist2 < 10. * deltagal**2] = False

    xgalf = np.array(xgalf)
    ygalf = np.array(ygalf)

    if plotgal:
        ax.scatter(xgalf, ygalf, marker = 'o', s = 200, alpha = 0.5)

        plt.savefig("/home/apps/astro/ARCHIVE/TEST/catalogue_%s_%s_%02i-%02i.png" % (field, CCD, filesci, fileref))

    return (xgalf, ygalf)


# get number of local maxima and the ratio between the maximum and the median of these maxima
# ------------------------------------------------------------------------------------------
def get_local_maxima(im, nn_radius):
    mask = (filters.maximum_filter(im, size = nn_radius, mode = 'constant', cval = 0.0) == im)
    max_array = im[mask]
    return np.sum(mask), np.median(max_array[np.isfinite(max_array)]) / np.median(im[np.isfinite(im)])


# compute Hu moments given an image
# ----------------------------------
def humoments(im, mask):
    immask = im[mask]
    m00 = np.sum(immask)
    hu = np.zeros(8)
    if m00 != 0:
        m00_inv2 = 1. / m00 / m00
        m00_inv25 = m00**-2.5
        # variance
        u20 = np.sum(immask * xx2D[mask]) * m00_inv2
        u11 = np.sum(immask * xy2D[mask]) * m00_inv2
        u02 = np.sum(immask * yy2D[mask]) * m00_inv2
        # skewness
        u30 = np.sum(immask * xxx2D[mask]) * m00_inv25
        u21 = np.sum(immask * xxy2D[mask]) * m00_inv25
        u12 = np.sum(immask * xyy2D[mask]) * m00_inv25
        u03 = np.sum(immask * yyy2D[mask]) * m00_inv25
        # eccentricity
        lambda1 = (u20 + u02) / 2. + sqrt(4. * u11**2 + (u20 - u02)**2) / 2.
        lambda2 = (u20 + u02) / 2. - sqrt(4. * u11**2 + (u20 - u02)**2) / 2. 
        ecc = np.sqrt(1. - lambda2 / lambda1)
        # Hu moments
        du20u02 = u20 - u02
        u21u03 = u21 + u03
        u03u21 = u03 + u21
        u03u212 = u03u21 * u03u21
        u21u032 = u21u03 * u21u03
        u30u12 = u30 + u12
        u30u122 = u30u12 * u30u12
        aux1 = u30 - 3. * u12
        aux2 = 3. * u21 - u03
        hu[0] = u20 + u02
        hu[1] = du20u02 * du20u02 + 4. * u11 * u11
#        hu[2] = aux1 * aux1 + aux2 * aux2
#        hu[3] = u30u12 * u30u12 + u21u032
        hu[2] = aux1 * u30u12 * (u30u122 - 3. * u21u032) + aux2 * u21u03 * (3. * u30u122 - u21u032)
        hu[3] = du20u02 * (u30u12 * u30u12 - u21u032) + 4. * u11 * u30u12 * u21u03
        hu[4] = aux2 * u30u12 * (u30u122 - 3. * u21u032) + aux1 * u21u03 * (3. * u30u122 - u21u032)
        hu[5] = u11 * (u30u122 - u03u21 * u03u21) - du20u02 * u30u12 * u03u21
        hu[6] = u11 * (u30u122 - u03u212) - du20u02 * u30u12 * u03u21
        hu[7] = ecc # added as hu moment
        
    return hu
        
# cosmic ray indicator
def CR(image, npsf): # from cosmics.py
  
    # resample image
    nresamp = 2
    newshape = (nresamp * image.shape[0], nresamp * image.shape[1])
    slices = [slice(0, old, float(old) / new) for old, new in zip(image.shape, newshape)]  # ((21, 84), (21, 84)) -> (0, 21, 0.25), (0, 21, 0.25)
    coordinates = np.mgrid[slices]
    indices = coordinates.astype('i')   #choose the biggest smaller integer index
    imagenew = image[tuple(indices)]

    # convolve with laplacian
    laplkernel = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
    conved = scipysignal.convolve2d(imagenew, laplkernel, mode = 'same', boundary = 'symm')
    conved = conved.clip(min = 0.0)
    
    # resample
    conved = conved.reshape(npsf, nresamp, npsf, nresamp)
    conved = conved.mean(axis = 3).mean(axis = 1)

    # clipped Laplacian
    Fim = (filters.median_filter(image, size = (10, 10)) - filters.median_filter(filters.median_filter(image, size = (10, 10)), size = (20, 20)))
    Fim = Fim.clip(np.abs(np.min(Fim)))
    
    # ratio
    return np.max(np.abs(conved / Fim))

# for kernel training stars only
Xstars, Ystars = np.meshgrid(np.array(range(npsf + nf)), np.array(range(npsf + nf)))
rs2Dstars = np.array(np.sqrt((Xstars - (npsf + nf - 1.) / 2.)**2 + (Ystars - (npsf + nf - 1.) / 2.)**2)).flatten()

# for contour plotting and for symmetry index
X, Y = np.meshgrid(np.array(range(npsf)), np.array(range(npsf)))
rs2D = np.array(np.sqrt((X - (npsf - 1.) / 2.)**2 + (Y - (npsf - 1.) / 2.)**2)).flatten()
xs2D = np.array(X - (npsf + nf - 1.) / 2.).flatten()
ys2D = np.array(Y - (npsf + nf - 1.) / 2.).flatten()
thetas2D = np.array(np.arctan2(Y - (npsf - 1.) / 2., X - (npsf - 1.) / 2.) + np.pi).flatten()
maskthetas = []
nthetas = 8
thetas = np.linspace(0, nthetas - 0.5, num = 2. * nthetas) / nthetas * 2. * np.pi
maskrs = (rs2D >= npsf / 2. * 0.5) & (rs2D <= npsf / 2.)
for i in range(len(thetas)):
  if i == 0:
    maskthetas.append((rs2D >= npsf / 2. / 2.) & (rs2D <= npsf / 2.) & ((thetas2D > thetas[2 * nthetas - 1]) | (thetas2D <= thetas[i + 1])))
  elif i == len(thetas) - 1:
    maskthetas.append((rs2D >= npsf / 2. / 2.) & (rs2D <= npsf / 2.) & (thetas2D > thetas[i - 1]))
  else:
    maskthetas.append((rs2D >= npsf / 2. / 2.) & (rs2D <= npsf / 2.) & (thetas2D > thetas[i - 1]) & (thetas2D <= thetas[i + 1]))
# for optimal photometry (do not change number 4, calibrated for good galaxy-star separation)
maskpsf = (np.array(rs2D < 4.)).flatten()

x2D = xs2D / npsf
y2D = ys2D / npsf
xy2D = x2D * y2D
xx2D = x2D * x2D
yy2D = y2D * y2D
xxx2D = xx2D * x2D
xxy2D = xx2D * y2D
xyy2D = x2D * yy2D
yyy2D = yy2D * y2D

# astrometric variables
NAXIS = np.zeros((2, 2))
CD = np.zeros((2, 2, 2))
nPV1 = 2
nPV2 = 11
PV = np.zeros((2, nPV1, nPV2))
CRVAL = np.zeros((2, 2))
CRPIX = np.zeros((2, 2))

# send no wait command to look for matches in other catalogues
if dolook:

    fileheader = "%s/LISTHEAD/%s_%s_%02i_listhead.txt" % (sharedir, field, CCD, fileref)
    if docrblaster and filesci != 0:
        command = "listhead %s/%s/%s/%s_%s_%02i_image.fits.fz | egrep 'NAXIS1|NAXIS2|CD1_1|CD1_2|CD2_1|CD2_2|CRPIX1|CRPIX2|CRVAL1|CRVAL2|nPV1|nPV2|PV1_0|PV1_0|PV1_1|PV1_2|PV1_3|PV1_4|PV1_5|PV1_6|PV1_7|PV1_8|PV1_9|PV1_10|PV1_11|PV2_0|PV2_1|PV2_2|PV2_3|PV2_4|PV2_5|PV2_6|PV2_7|PV2_8|PV2_9|PV2_10|PV2_11|MJD-OBS' | cut -f 1 -d '/' > %s" % (refdir, field, CCD, field, CCD, fileref, fileheader)
    else:
        command = "listhead %s/%s/%s/%s_%s_%02i_image_crblaster.fits | egrep 'NAXIS1|NAXIS2|CD1_1|CD1_2|CD2_1|CD2_2|CRPIX1|CRPIX2|CRVAL1|CRVAL2|nPV1|nPV2|PV1_0|PV1_0|PV1_1|PV1_2|PV1_3|PV1_4|PV1_5|PV1_6|PV1_7|PV1_8|PV1_9|PV1_10|PV1_11|PV2_0|PV2_1|PV2_2|PV2_3|PV2_4|PV2_5|PV2_6|PV2_7|PV2_8|PV2_9|PV2_10|PV2_11|MJD-OBS' | cut -f 1 -d '/' > %s" % (refdir, field, CCD, field, CCD, fileref, fileheader)

    if verbose:
        print command
    os.system(command)


    usnofile = "%s/%s/%s/CALIBRATIONS/USNO_%s_%s_%02i.npy" % (sharedir, field, CCD, field, CCD, fileref)
    if not os.path.exists(usnofile):
        nx = 2032
        ny = 4076
        # only USNO (wait)
        L = np.array(['python', 'dolook.py', field, CCD, fileref, '10', nx, ny, fileheader, sharedir, 'USNOESO', verbose])
        if verbose:
            print L
        os.spawnvpe(os.P_WAIT, 'python', L.tolist(), os.environ)

        # all but USNO (don't wait)
        L = np.array(['python', 'dolook.py', field, CCD, fileref, '10', nx, ny, fileheader, sharedir, 'all', verbose])
        if verbose:
            print L
        os.spawnvpe(os.P_NOWAIT, 'python', L.tolist(), os.environ)

# clean directory and run crblaster on reference image
if docrblaster and not os.path.exists("%s_image_crblaster.fits" % file1) and filesci != 0:
    
    command = "rm -rf %s_image_crblaster.fits" % file1
    if verbose:
        print "    %s\n" % command
    os.system(command)

    command = "mpirun -n %i %s 1 %s 1 %s_image.fits.fz %s_image_crblaster.fits" % (ncores, crblaster, ncores, file1, file1)
    if verbose:
        print "    %s\n" % command
    os.system(command)

# crblaster
# ---------------------------------------------------------
    
if dodiff and docrblaster and not os.path.exists("%s_image_crblaster.fits" % file2) and filesci != 0:
    command = "rm -rf %s_image_crblaster.fits" % file2
    if verbose:
        print "    %s\n" % command
    os.system(command)
    command = "mpirun -n %i %s 1 %s 1 %s_image.fits.fz %s_image_crblaster.fits" % (ncores, crblaster, ncores, file2, file2)
    if verbose:
        print "    %s\n" % command
    os.system(command)
        
# nice plotting
def HMS(hours, pos):
    if hours < 0:
        sign = '-'
    else:
        sign = ''
    hh = int(hours)
    mm = abs(int((hours - hh) * 60.))
    ss = (abs(hours - hh) - mm / 60.) * 3600
    hh = abs(hh)
    return "%s%02i:%02i:%02i" % (sign, hh, mm, ss)

        
# tangent projection coordinates
def xieta(x, y, PV): # all in degrees
  r = np.sqrt(x**2 + y**2)
  xicomp = PV[0, 0] + PV[0, 1] * x + PV[0, 2] * y + PV[0, 3] * r + PV[0, 4] * x**2 + PV[0, 5] * x * y + PV[0, 6] * y**2 + PV[0, 7] * x**3 + PV[0, 8] * x**2 * y + PV[0, 9] * x * y**2 + PV[0, 10] * y**3
  etacomp = PV[1, 0] + PV[1, 1] * y + PV[1, 2] * x + PV[1, 3] * r + PV[1, 4] * y**2 + PV[1, 5] * y * x + PV[1, 6] * x**2 + PV[1, 7] * y**3 + PV[1, 8] * y**2 * x + PV[1, 9] * y * x**2 + PV[1, 10] * x**3
  return (xicomp, etacomp)

# RA DEC given pixel coordinates
def RADEC(i, j, CD11, CD12, CD21, CD22, CRPIX1, CRPIX2, CRVAL1, CRVAL2, PV):

  # i, j to x, y
  x = CD11 * (i - CRPIX1) + CD12 * (j - CRPIX2) # deg 
  y = CD21 * (i - CRPIX1) + CD22 * (j - CRPIX2) # deg

  # x, y to xi, eta
  (xi, eta) = xieta(x, y, PV)

  # xi, eta to RA, DEC
  num1 = (xi * deg2rad) / np.cos(CRVAL2 * deg2rad) # rad
  den1 = 1. - (eta * deg2rad) * np.tan(CRVAL2 * deg2rad) # rad
  alphap = np.arctan2(num1, den1) # rad
  RA  = CRVAL1 + alphap * rad2deg # deg
  num2 = (eta * deg2rad + np.tan(CRVAL2 * deg2rad)) * np.cos(alphap) # rad
  DEC = np.arctan2(num2, den1) * rad2deg # deg

  return (RA / 15., DEC) # hr deg

# background estimation
def dobg(n, backgroundtype, imagefits, maskstars, a1):

    if backgroundtype == 'median':
    
        if maskstars:
                
            # read sources and mask them to estimate background
            (xsex, ysex, zsex) = np.loadtxt("%s_image_crblaster.fits-catalogue%s.dat" % (file1cat, cataloguestring), usecols = (1, 2, 5)).transpose()
            maskn = np.empty_like(n1, dtype = bool)
            maskn[:, :] = True
            maskn[0: int(2 * npsf), :] = False
            maskn[int(ny - 2 * npsf): ny, :] = False
            maskn[:, 0: int(2 * npsf)] = False
            maskn[:, int(nx - 2 * npsf): nx] = False
            for i in range(len(xsex)):
                if zsex[i] != 0:
                    rf = int(8. * np.log(np.abs(zsex[i])))
                    maskn[int(ysex[i] - rf): int(ysex[i] + rf), int(xsex[i] - rf): int(xsex[i] + rf)] = False
        
            # estimate median in 8 different regions and subtract it
            medianvals = np.empty((4, 2), dtype = float16)
            for i in range(4):
                medianvals[i, 0] = np.median(n[maskmedian[i, 0] & maskn])
                medianvals[i, 1] = np.median(n[maskmedian[i, 1] & maskn])
        
        else:
        
            # estimate median in 8 different regions and subtract it
            medianvals = np.empty((4, 2), dtype = float16)
            for i in range(4):
                medianvals[i, 0] = np.median(n[maskmedian[i, 0]])
                medianvals[i, 1] = np.median(n[maskmedian[i, 1]])
        
        # correct nans or infs
        medianmedian0 = np.median(medianvals[:, 0].flatten())
        medianmedian1 = np.median(medianvals[:, 1].flatten())
        for i in range(4):
            if not np.isfinite(medianvals[i, 0]):
                medianvals[i, 0] = medianmedian0
            if not np.isfinite(medianvals[i, 1]):
                medianvals[i, 1] = medianmedian1
        
        # fit two parabola as background
        X = np.zeros((4, 3), dtype = float32)
        Y = np.zeros(4, dtype = float32)
        X[:, 0] = 1.
        X[:, 1] = xmedian
        X[:, 2] = xmedian * xmedian
        Y = medianvals[:, 0]
        mat = np.dot(X.transpose(), X)
        rhs = np.dot(X.transpose(), Y)
        (bg11, bg12, bg13) = scipylinalg.solve(mat, rhs)
        Y = medianvals[:, 1]
        mat = np.dot(X.transpose(), X)
        rhs = np.dot(X.transpose(), Y)
        (bg21, bg22, bg23) = scipylinalg.solve(mat, rhs)
        
        # subtract parabola fit of background, take into account that the division between amplifiers has been shifted by -a1
        bg = np.zeros((ny, nx), dtype = float16)
        bg[:, max(0, -a1): nx / 2 - a1] = bg11 + bg12 * xidx[:, max(0, -a1): nx / 2 - a1] + bg13 * xidx[:, max(0, -a1): nx / 2 - a1]**2
        bg[:, nx / 2 - a1: min(nx, nx - a1)] = bg21 + bg22 * xidx[:, nx / 2 - a1: min(nx, nx - a1)] + bg23 * xidx[:, nx / 2 - a1: min(nx, nx - a1)]**2

    elif backgroundtype == 'sextractor':

        backgroundfits = imagefits.replace(".fits", "_%s_background%i.fits" % (useweights, backsize))
        print "\nBackground fits file", backgroundfits
        
        if not os.path.exists(backgroundfits) or forcebackground:
            command = "sex -c %s/etc/default.sex %s -CATALOG_NAME %s-catalogue%s.dat -WEIGHT_TYPE BACKGROUND -BACK_SIZE %i -CHECKIMAGE_TYPE BACKGROUND -CHECKIMAGE_NAME %s" % (hitsdir, imagefits, imagefits.replace(outdir, sharedir), cataloguestring, backsize, backgroundfits)
            os.system(command)
        bg = fits.open(backgroundfits)[0].data
    
    return bg

# open input fits files
if dodiff:
    
    if verbose:
        print "Cleaning candidate images..."
    command = "rm -rf %s/%s/%s/CANDIDATES/cand_%s_%s_%02it-%02i_grid%02i_%s*png" % (webdir, field, CCD, field, CCD, filesci, fileref, fileref, resampling)
    if verbose:
        print "  %s" % command
    os.system(command)
    command = "rm -rf %s/%s/%s/CANDIDATES/cand_%s_%s_%02i-%02it_grid%02i_%s*png" % (webdir, field, CCD, field, CCD, filesci, fileref, fileref, resampling)
    if verbose:
        print "  %s" % command
    os.system(command)
    print "\n"

    # Reference file
    # --------------------------------

    fits1 = fits.open("%s_image_crblaster.fits" % file1)
    n1 = fits1[0].data
    if verbose:
        print "n1 image properties:"
        print "   n1 dtype: ", n1.dtype, ", sum of non finite:", np.sum(np.invert(np.isfinite(n1))), ", shape:", np.shape(n1), "\n"
    if dodqmask:
        if useweights == 'external' or useweights == 'internal':
            if useweights == 'external':
                dqn1 = fits.open("%s_dqmask.fits.fz" % file1)[0].data
                n1orig = fits.open("%s_image.fits.fz" % file1)[0].data  # original image without crblaster
            elif useweights == 'internal':
                fitsall1 = fits.open("%s_image.fits" % file1)
                n1orig = fitsall1[0].data
                dqn1 = np.array(fitsall1[1].data, dtype = bool)
    if verbose:
        print "Data quality image 1 properties:"
        print "   dqn1 dtype: ", dqn1.dtype, ", sum of non finite:", np.sum(np.invert(np.isfinite(dqn1))), "\n"
               

    # actual shape
    ny = np.shape(n1)[0]
    nx = np.shape(n1)[1]

    # variables used for background estimation
    xidx = np.zeros((nx, ny), dtype = 'int32')
    yidx = np.zeros((ny, nx), dtype = 'int32')
    xidx[:] = np.arange(ny)
    xidx = xidx.transpose()
    yidx[:] = np.arange(nx)
    background1 = np.zeros((ny, nx), dtype = float16)
    background2 = np.zeros((ny, nx), dtype = float16)
    maskmedian = np.empty((4, 2), dtype = object)
    xmedian = np.zeros(4, dtype = int)
    for i in range(4):
        maskmedian[i, 0] = (yidx < nx / 2) & (xidx >= i * ny / 4) & (xidx < (i + 1) * ny / 4)
        maskmedian[i, 1] = (yidx >= nx / 2) & (xidx >= i * ny / 4) & (xidx < (i + 1) * ny / 4)
        xmedian[i] = (i + 0.5) * ny / 4

    if useweights == 'own':
        background1 = dobg(n1, 'median', None, False, 0)
    
    # data from reference file
    headerref = fits1[0].header
    try:
        MJDref = float(headerref['MJD-OBS'])
    except:
        print "\n\nWARNING: Cannot find field 'MJD-OBS'\n\n"
        sys.exit(12)
    MJDproc = MJDref
    try:
        gain = float(headerref['ARAWGAIN'])
    except:
        gain = 4.35
    if verbose:
        print "Environmental, instrumental variables:"
        print "   MJD reference: %f, gain: %f" % (MJDref, gain)
    try:
        exptime = float(headerref['EXPTIME'])
    except:
        print "\n\nWARNING: Cannot find field 'EXPTIME'\n\n"
        sys.exit(12)
    try:
        airmass = float(headerref['AIRMASS'])
    except:
        print "\n\nWARNING: Cannot find field 'AIRMASS'\n\n"
        sys.exit(12)
    if verbose:
        print "   Exposure time [sec]: %f, airmass: %f\n" % (exptime, airmass)

    # weight files
    if useweights == 'external' or useweights == 'internal':
        if useweights == 'external':
            varfits1 = fits.open("%s_wtmap.fits.fz" % file1)
            varn1 = varfits1[0].data
            if verbose:
                print "n1 variance properties:"
                print "   varn1 dtype: ", varn1.dtype, ", sum of non finite:", np.sum(np.invert(np.isfinite(varn1))), ", shape:", np.shape(varn1), "\n"
        elif useweights == 'internal':
            varn1 = fitsall1[2].data
            if verbose:
                print "n1 variance properties:"
                print "   varn1 dtype: ", varn1.dtype, ", sum of non finite:", np.sum(np.invert(np.isfinite(varn1)))

        if verbose:        
            print "Percentage of good pixels with variance less than maximum variance: %f\n" % (1. * np.sum(varn1 < varmax) / nx / ny)

        # convert weight map into variance map
        if verbose:
            print "Converting weight map into variance map..."
        varn1[varn1 < varmax] = 1. / varn1[varn1 < varmax]# + np.abs(n1[varn1 < varmax] / gain)  # variance

        # save reference variance map if it doesn't exist
        if not os.path.exists("%s_varmap.fits" % file1):
            if verbose:
                print "Saving %s_varmap.fits" % file1
            fits1[0].data = varn1
            fits1.writeto("%s_varmap.fits" % file1, clobber = True)
        else:
            print "%s_varmap.fits exists" % file1

    elif useweights == 'own':   # not really used anymore

        varn1 = np.array(pickle.load(open("%s/domeflat/%s/domeflat_%s_master_mask.pkl" % (sharedir, CCD, CCD), 'rb')), dtype = float)
        if np.shape(varn1) != np.shape(background1):
            print "\n\nWARNING: mask size is not the same as the image size\n\n"
            sys.exit(13)
        varn1 = varn1 * 1. / (2.80942971311 + 0.21173173807 * background1)
        # write file to fits file
        if not os.path.exists("%s_varmap.fits" % file1):
            fits1[0].data = varn1
            fits1[0].writeto("%s_varmap.fits" % file1)
        # add Poisson noise
        varn1[varn1 < varmax] = 1. / varn1[varn1 < varmax] + n1[varn1 < varmax] / gain  # variance

    else: # not really used anymore
        varn1 = 1. / (2.80942971311 + 0.21173173807 * mediann1) + n1 / gain

    # mask above maximum variance
    varn1[varn1 >= varmax] = varmax
    varn1[np.invert(np.isfinite(varn1))] = varmax
    if dodqmask:
        varn1[dqn1 > dqth] = 2. * varmax

    # date since reference
    date = 0

    # astrometric solution in reference frame
    CRVAL[0, 0] = headerref['CRVAL1']
    CRVAL[0, 1] = headerref['CRVAL2']
    NAXIS[0, 0] = headerref['NAXIS1']
    NAXIS[0, 1] = headerref['NAXIS2']
    CRPIX[0, 0] = headerref['CRPIX1']
    CRPIX[0, 1] = headerref['CRPIX2']
    CD[0, 0, 0]   = headerref['CD1_1']
    CD[0, 0, 1]   = headerref['CD1_2']
    CD[0, 1, 0]   = headerref['CD2_1']
    CD[0, 1, 1]   = headerref['CD2_2']
    # non-linear distortion parameters PV
    for i in range(nPV1):
        for j in range(nPV2):
            PV[0, i, j] = float(headerref['PV%i_%i' % (i + 1, j)])

    # Science image
    # ---------------------------------------
    if verbose:
        print "\nReading science image"
    fits2 = fits.open("%s_image_crblaster.fits" % file2)
    
    # astrometric solution in science image
    if astrometry == 'WCS':
        headersci = fits2[0].header
        CRVAL[1, 0] = headersci['CRVAL1']
        CRVAL[1, 1] = headersci['CRVAL2']
        NAXIS[1, 0] = headersci['NAXIS1']
        NAXIS[1, 1] = headersci['NAXIS2']
        CRPIX[1, 0] = headersci['CRPIX1']
        CRPIX[1, 1] = headersci['CRPIX2']
        CD[1, 0, 0]   = headersci['CD1_1']
        CD[1, 0, 1]   = headersci['CD1_2']
        CD[1, 1, 0]   = headersci['CD2_1']
        CD[1, 1, 1]   = headersci['CD2_2']
        # non-linear distortion parameters PV
        for i in range(nPV1):
            for j in range(nPV2):
                PV[1, i, j] = float(headersci['PV%i_%i' % (i + 1, j)])
                
        projectionwcs.setheader(NAXIS, CRPIX, CRVAL, CD, PV)

    # check that filter name matches the reference filtername
    if filtername != ((fits2[0].header)["FILTER"])[0]:
        print "\n\nWARNING: Science filter does not match reference filter. EXIT\n\n"
        sys.exit(27)

    MJDproc = float(fits2[0].header['MJD-OBS'])
    try:
        AIRMASSproc = float(fits2[0].header['AIRMASS'])
        EXPTIMEproc = float(fits2[0].header['EXPTIME'])
    except:
        "\n\nWARNING: Cannot find airmass or exposure time in header.\n\n"

    if verbose:
        print "   MJD science: %f" % MJDproc
        
    image = fits2[0].data        
    if dodqmask:
        if useweights == 'external' or useweights == 'internal':
            if useweights == 'external':
                dqn2 = fits.open("%s_dqmask.fits.fz" % file2)[0].data
            elif useweights == 'internal':
                fitsall2 = fits.open("%s_image.fits" % file2)
                dqn2 = np.array(fitsall2[1].data, dtype = bool)

    if useweights == 'own':
        backgroundimage = dobg(image, 'median', None, False, 0)

    if useweights == 'external' or useweights == 'internal':
        if useweights == 'external':
            varfits2 = fits.open("%s_wtmap.fits.fz" % file2)
            varimage = varfits2[0].data
        elif useweights == 'internal':
            varimage = fitsall2[2].data
        if verbose:
            print "Percentage of good pixels:", (1. * np.sum(varimage < varmax) / nx / ny)
        # convert to variance
        if verbose:
            print "Converting science weight map into variance map"
        varimage[varimage < varmax] = 1. / varimage[varimage < varmax]# + np.abs(image[varimage < varmax] / gain)  # variance

        if not os.path.exists("%s_varmap.fits" % file2):
            if verbose:
                print "Saving %s_varmap.fits" % file2
                fits1[0].data = varimage
            fits1.writeto("%s_varmap.fits" % file2, clobber = True)
        else:
            if verbose:
                print "%s_varmap.fits exists" % file2

    elif useweights == 'own':
        varimage = np.array(pickle.load(open("%s/domeflat/%s/domeflat_%s_master_mask.pkl" % (sharedir, CCD, CCD), 'rb')), dtype = float)
        varimage = varimage * 1. / (2.80942971311 + 0.21173173807 * backgroundimage)
        if not os.path.exists("%s_varmap.fits" % file2):
            fits2[0].data = varimage
            fits2.writeto("%s_varmap.fits" % file2)
        # add Poisson noise
        varimage[varimage < varmax] = 1. / varimage[varimage < varmax] + image[varimage < varmax] / gain  # variance
    else:
        varimage =  1. / (2.80942971311 + 0.21173173807 * mediann1) + image / gain

    varimage[varimage >= varmax] = varmax
    varimage[np.invert(np.isfinite(varimage))] = varmax
    if dodqmask:
        varimage[dqn2 > dqth] = varmax

    # shapes, partition image in 6 x 3
    if verbose:
        print "Image size and image size keywords:"
        print "   nx: %i, ny: %i, NAXIS1: %i, NAXIS2: %i" % (nx, ny, NAXIS[0, 0], NAXIS[0, 1])
    
    # create directory and look for USNO data if necessary
    usnofile = "%s/%s/%s/CALIBRATIONS/USNO_%s_%s_%02i.npy" % (sharedir, field, CCD, field, CCD, fileref)
    # USE GAIA
    usnofile = "%s/GAIA/VOT/%s_GAIA_%s.vot" % (sharedir, field, CCD)
    gaiafile = "%s/%s/%s/CALIBRATIONS/GAIA_%s_%s_%02i.npy" % (sharedir, field, CCD, field, CCD, fileref)
    if os.path.exists(usnofile):
        
        # GAIA quickfix
        USNO = Table.read(usnofile, format = 'votable')
        USNORA = np.array(USNO['ra']) / 15.
        USNODEC = np.array(USNO['dec'])
        USNO_B = np.array(USNO['phot_g_mean_mag'])
        USNO_R = np.array(USNO['phot_g_mean_mag'])
        USNONAME = np.array(USNO['source_id'])
        
        np.save(gaiafile, np.vstack([USNORA, USNODEC, USNONAME, USNO_B]))

        # original USNO
        #USNO = np.load(usnofile)
        #USNORA = np.array(USNO[0], dtype = float)
        #USNODEC = np.array(USNO[1], dtype = float)
        #USNONAME = USNO[2]
        #USNO_B = np.array(USNO[3], dtype = float)
        #USNO_R = np.array(USNO[4], dtype = float)
    else:
        print "WARNING: File %s does not exist, trying again in USNO and ESO sites..." % usnofile

        # only USNO (wait)
        L = np.array(['python', 'dolook.py', field, CCD, fileref, '10', nx, ny, fileheader, sharedir, 'USNOESO', verbose])
        if verbose:
            print L
        os.spawnvpe(os.P_WAIT, 'python', L.tolist(), os.environ)

        if not os.path.exists(usnofile):
            # only USNO (wait)
            L = np.array(['python', 'dolook.py', field, CCD, fileref, '10', nx, ny, fileheader, sharedir, 'USNO', verbose])
            if verbose:
                print L
            os.spawnvpe(os.P_WAIT, 'python', L.tolist(), os.environ)

            if not os.path.exists(usnofile):
                # only USNO (wait)
                L = np.array(['python', 'dolook.py', field, CCD, fileref, '10', nx, ny, fileheader, sharedir, 'USNOESO', verbose])
                if verbose:
                    print L
                os.spawnvpe(os.P_WAIT, 'python', L.tolist(), os.environ)

                print "WARNING: File %s does not exist, EXIT" % usnofile
                sys.exit(14)


else:
    nx = 2032
    ny = 4076
    print "Set nx= %i, ny= %i" % (nx, ny)
    gain = 4.35

# partitions
npartx = 3 # 3
nparty = 6 # 6
dxconv = 1. * (ny - nf) / nparty
dyconv = 1. * (nx - nf) / npartx

# number of pixels for kernel construction
dn = int((npsf + nf) / 2.)
npsfh = npsf / 2
nfh = nf / 2

if verbose:
    print "\n\nImage and kernel size related variables:"
    print "   nx: %i, ny: %i, npartx: %i, nparty: %i, dxconv: %i, dyconv: %i, dn: %i, npsf: %i, npsfh: %i, nf: %i, nfh: %i\n\n" % (nx, ny, npartx, nparty, dxconv, dyconv, dn, npsf, npsfh, nf, nfh)

# background file
backgroundfits = "%s_image_crblaster_%s_background%i.fits" % (file1.replace(refdir, sharedir), useweights, backsize)

if verbose:
    print "Reference catalogue file: %s_image_crblaster.fits-catalogue%s.dat" % (file1cat, cataloguestring)

# run extractor on reference image
if (not os.path.exists("%s_image_crblaster.fits-catalogue%s.dat" % (file1cat, cataloguestring)) or not os.path.exists(backgroundfits) or forcebackground) and filesci != 0:

    if useweights == 'external' or useweights == 'internal' or useweights == 'own':
        command = "sex -c %s/etc/default.sex %s_image_crblaster.fits -CATALOG_NAME %s_image_crblaster.fits-catalogue%s.dat -WEIGHT_TYPE BACKGROUND -BACK_SIZE %i -CHECKIMAGE_TYPE BACKGROUND -CHECKIMAGE_NAME %s -VERBOSE_TYPE %s" % (hitsdir, file1, file1cat, cataloguestring, backsize, backgroundfits, sexverbose)
    else:
        command = "sex -c %s/etc/default.sex %s_image_crblaster.fits -CATALOG_NAME %s_image_crblaster.fits-catalogue%s.dat -WEIGHT_TYPE BACKGROUND -BACK_SIZE %i -CHECKIMAGE_TYPE BACKGROUND -CHECKIMAGE_NAME %s -VERBOSE_TYPE %s" % (hitsdir, file1, file1cat, cataloguestring, backsize, backgroundfits, sexverbose)
    if verbose:
        print "    %s\n" % command
    os.system(command)

else:
    if verbose:
        print "Sextractor already run on %s" % file1

# initialize images to subtract
if dodiff:

    background1 = fits.open(backgroundfits)[0].data
    n1 = n1 - background1
    if useweights == 'internal':
        n1orig = n1orig - background1

    # mask bad pixels
    if dozeromask:
        n1[varn1 == varmax] = 0

    n2 = np.empty_like(n1)
    varn2 = np.empty_like(varn1)

# whether to transform first or 2nd image
conv1st = True


# select N brightest objects well inside the image edges and that are not in crowded regions
def select(x, y, z, tolx, toly, xmax, xmin, ymax, ymin, error, N):

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
        
# find arbitrary order (1, 2, or 3) transformation relating two sets of points
def findtransformation(order, x1, y1, x2, y2):
    
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
    
    if order == 1:
        nptmin = 3
    elif order == 2:
        nptmin = 6
    elif order == 3:
        nptmin = 10
        
    npt = len(x1)
    if npt < nptmin:
        print "\n\nWARNING: Not enough stars to do order %i astrometric solution (%i)...\n\n" % (order, npt)
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
    if order > 1:
        X[0: npt, iterm] = x1 * x1
        iterm = iterm + 1
        X[0: npt, iterm] = x1 * y1
        iterm = iterm + 1
        X[0: npt, iterm] = y1 * y1
        iterm = iterm + 1
    if order > 2:
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
        print "Solving order %i transformation (npt: %i)..." % (order, npt)
        if order == 1:
            (a1, b11, b12, a2, b21, b22) = scipylinalg.solve(mat, rhs)
            sol_astrometry = np.array([a1, a2, b11, b12, b21, b22])
        elif order == 2:
            (a1, b11, b12, c11, c12, c13, a2, b21, b22, c21, c22, c23) = scipylinalg.solve(mat, rhs)
            sol_astrometry = np.array([a1, a2, b11, b12, b21, b22, c11, c12, c13, c21, c22, c23])
        elif order == 3:
            (a1, b11, b12, c11, c12, c13, d11, d12, d13, d14, a2, b21, b22, c21, c22, c23, d21, d22, d23, d24) = scipylinalg.solve(mat, rhs)
            sol_astrometry = np.array([a1, a2, b11, b12, b21, b22, c11, c12, c13, c21, c22, c23, d11, d12, d13, d14, d21, d22, d23, d24])
    except:
        print "\n\nWARNING: Error solving linear system when matching pixel coordinate systems\n\n"
        sys.exit(16)

    return sol_astrometry

# apply previous transformation
def applytransformation(order, x1, y1, sol):

    # this is slow, but I prefer fewer bugs than speed at the moment...

    x1t = sol[0] + sol[2] * x1 + sol[3] * y1
    y1t = sol[1] + sol[4] * x1 + sol[5] * y1
    if order > 1:
        x1t = x1t + sol[6] * x1 * x1 + sol[7] * x1 * y1 + sol[8] * y1 * y1
        y1t = y1t + sol[9] * x1 * x1 + sol[10] * x1 * y1 + sol[11] * y1 * y1
    if order > 2:
        x1t = x1t + sol[12] * x1 * x1 * x1 + sol[13] * x1 * x1 * y1 + sol[14] * x1 * y1 * y1 + sol[15] * y1 * y1 * y1
        y1t = y1t + sol[16] * x1 * x1 * x1 + sol[17] * x1 * x1 * y1 + sol[18] * x1 * y1 * y1 + sol[19] * y1 * y1 * y1

    return (x1t, y1t)


# distance between two sets
def xydistance(x1, x2, y1, y2, delta):
    
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
def roughastro(x1, x2, y1, y2, deltaxmin, deltaxmax, deltaymin, deltaymax, delta):

    ibest = 0
    jbest = 0
    nbest = 0

    for i in arange(deltaxmin, deltaxmax, delta):
        
        for j in arange(deltaymin, deltaymax, delta):

            (nsources, dist) = xydistance(x1, x2 + i, y1, y2 + j, delta)
            if nsources >= nbest:
                ibest = i
                jbest = j
                nbest = nsources
                #print ibest, jbest, nsources, dist
    
    return ibest, jbest

    
# function that matches two sets of stars using constant shift, returning selected positions and shifts
# as well as linear transformation with coefficients and rms
def match(N, pixscale, order, x1, y1, z1, e_z1, r1, x2, y2, z2, e_z2, r2, tolx, toly, xmin, xmax, ymin, ymax, error1, error2, flux1min, flux1max, flux2min, flux2max):

    # ASTROMETRY
    # ---------------------------------------------------

    if e_z1 is None:
        e_z1 = np.ones(len(z1))
    if e_z2 is None:
        e_z2 = np.ones(len(z2))
    if r1 is None:
        r1 = np.ones(len(z1))
    if r2 is None:
        r2 = np.ones(len(z2))

    testastrometry = False
    if testastrometry:
        print "Flux cuts based on expected number of stars:"
        print "   flux1min %f, flux1max %f, flux2min %f, flux2max %f" % (flux1min, flux1max, flux2min, flux2max)

    if verbose:
        print "%i and %i stars before flux cut" % (len(x1), len(x2)) 

    x1 = x1[(z1 > flux1min) & (z1 < flux1max)]
    y1 = y1[(z1 > flux1min) & (z1 < flux1max)]
    r1 = r1[(z1 > flux1min) & (z1 < flux1max)]
    e_z1 = e_z1[(z1 > flux1min) & (z1 < flux1max)]
    z1 = z1[(z1 > flux1min) & (z1 < flux1max)]
    
    x2 = x2[(z2 > flux2min) & (z2 < flux2max)]
    y2 = y2[(z2 > flux2min) & (z2 < flux2max)]
    r2 = r2[(z2 > flux2min) & (z2 < flux2max)]
    e_z2 = e_z2[(z2 > flux2min) & (z2 < flux2max)]
    z2 = z2[(z2 > flux2min) & (z2 < flux2max)]


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

    if testastrometry:
        print "%i and %i stars after normalization cut" % (len(x1), len(x2)) 
        fig, ax = plt.subplots()#nrows = 2, figsize = (21, 14))
        ax.scatter(y1, x1, marker = 'o', c = 'r', s = 10, alpha = 0.5, edgecolors = 'none')
        ax.scatter(y2, x2, marker = '*', c = 'b', s = 10, alpha = 0.5, edgecolors = 'none')
        #        ax[1].imshow(n1.transpose(), interpolation = 'nearest', clim = (np.percentile(n1[::10, ::10].flatten(), 5), np.percentile(n1[::10, ::10].flatten(), 98)), cmap = 'gray', origin = 'lower')
        #        ax[1].scatter(y1, x1, marker = 'o', edgecolors = 'r', facecolor = 'none', s = 100, alpha = 0.5)
        if pixscale == 1:
            ax.axvline(30)
            ax.axvline(ny - 30)
            ax.axhline(30)
            ax.axhline(nx - 30)
            ax.set_ylim(0, nx)
            ax.set_xlim(0, ny)
#        ax[1].set_ylim(0, nx)
#        ax[1].set_xlim(0, ny)
        plt.savefig("%s/TESTING/fforster/astrometry/test_%s_0.png" % (webdir, pixscale == 1))
 
    # first select only sources not in crowded regions and far from the edges
    if pixscale == 1.:
        (x1s, y1s) = select(x1, y1, z1, tolx, toly, xmin, xmax, ymin, ymax, max(error1, error2), N)
        (x2s, y2s) = select(x2, y2, z2, tolx / 2., toly / 2., xmin, xmax, ymin, ymax, max(error1, error2), N)
    else:
        if len(x1) > len(x2):
            (x1s, y1s) = select(x1, y1, z1, tolx, toly, xmin, xmax, ymin, ymax, max(error1, error2), N)
            (x2s, y2s) = select(x2, y2, z2, tolx / 2., toly / 2., xmin, xmax, ymin, ymax, error2, len(x1s))
        else:
            (x2s, y2s) = select(x2, y2, z2, tolx, toly, xmin, xmax, ymin, ymax, max(error1, error2), N)
            (x1s, y1s) = select(x1, y1, z1, tolx / 2., toly / 2., xmin, xmax, ymin, ymax, error1, len(x2s))

    if testastrometry:
        print "%i and %i stars selected by select routine" % (len(x1s), len(x2s)) 
        fig, ax = plt.subplots()
        ax.scatter(y1s, x1s, marker = 'o', edgecolors = 'none', c = 'r', s = 10, alpha = 0.5)
        ax.scatter(y2s, x2s, marker = '*', edgecolors = 'none', c = 'b', s = 10, alpha = 0.5)
        ax.set_title("Selected", fontsize = 8)
        if pixscale == 1:
            ax.axvline(30)
            ax.axvline(ny - 30)
            ax.axhline(30)
            ax.axhline(nx - 30)
            ax.set_ylim(0, nx)
            ax.set_xlim(0, ny)
        plt.savefig("%s/TESTING/fforster/astrometry/test_%s_1.png" % (webdir, pixscale == 1))

    if len(x1s) == 0 or len(x2s) == 0:
        print "   ---> WARNING, no matching stars found."
        return None

    # Use brute force approach to find first guess
    
    dorough = True
    if dorough:

        ibest = 0
        jbest = 0
        
        print "Refining solution to 25 pixels..."
        (ibest, jbest) = roughastro(x1, x2, y1, y2, ibest - 500 * pixscale, ibest + 500 * pixscale, jbest - 500 * pixscale, jbest + 500 * pixscale, 25 * pixscale)
        
        print ibest, jbest
        
        print "Refining solution to 5 pixels..."
        (ibest, jbest) = roughastro(x1, x2, y1, y2, ibest - 25 * pixscale, ibest + 25 * pixscale, jbest - 25 * pixscale, jbest + 25 * pixscale, 5 * pixscale)
        
        print ibest, jbest
        
        print "Refining solution to 2 pixels..."
        (ibest, jbest) = roughastro(x1, x2, y1, y2, ibest - 5 * pixscale, ibest + 5 * pixscale, jbest - 5 * pixscale, jbest + 5 * pixscale, 2 * pixscale)
        
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

        # find the highest concentration
        nsep = np.zeros(len(sepx))
        for i in range(len(sepx)):
            nsep[i] = np.sum((np.sqrt((sepx - sepx[i])**2 + (sepy - sepy[i])**2) < 5. * pixscale))
        idxnsep = np.argmax(nsep)
        sepxcomp = sepx[idxnsep]
        sepycomp = sepy[idxnsep]
        #sepxcomp = np.median(sepx[nsep == max(nsep)])
        #sepycomp = np.median(sepy[nsep == max(nsep)])

        if testastrometry:
            fig, ax = plt.subplots(figsize = (12, 6))
            ax.scatter(sepy, sepx, marker = '.', edgecolors = 'none', c = 'b', s = 5)
            ax.scatter(sepycomp, sepycomp, marker = 'o', facecolors = 'none', s = 100)
            ax.set_title("Minimum separations", fontsize = 8)
            plt.savefig("%s/TESTING/fforster/astrometry/test_%s_2.png" % (webdir, pixscale == 1))
    
        # 0th order correction
        deltax = sepx[idxnsep] #np.median(sepx)
        deltay = sepy[idxnsep] #np.median(sepy)

    # mask stars on the edges and apply correction
    mask1 = (x1 > xmin + tolx) & (x1 < xmax - tolx) & (y1 > ymin + toly) & (y1 < ymax - toly)
    mask2 = (x2 > xmin + tolx) & (x2 < xmax - tolx) & (y2 > ymin + toly) & (y2 < ymax - toly)
    x1 = x1[mask1]
    y1 = y1[mask1]
    z1 = z1[mask1]
    x2 = x2[mask2]
    y2 = y2[mask2]
    z2 = z2[mask2]
    x2 = x2 + deltax
    y2 = y2 + deltay

    if testastrometry:
        print "Stars after median correction: %i" % len(x2)
        fig, ax = plt.subplots(figsize = (12, 6))
        ax.scatter(y1, x1, marker = 'o', edgecolors = 'none', c = 'r', s = 10, alpha = 0.5)
        ax.scatter(y2, x2, marker = '*', edgecolors = 'none', c = 'b', s = 10, alpha = 0.5)
        ax.set_title("Median distance correction", fontsize = 8)
        plt.savefig("%s/TESTING/fforster/astrometry/test_%s_3.png" % (webdir, pixscale == 1))

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
        if dist[idxmin] < max(error1, error2) / 2.:
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

    if testastrometry:
        print "Stars after distance matching: %i" % len(x1)
        fig, ax = plt.subplots(figsize = (12, 6))
        ax.scatter(y1, x1, marker = 'o', edgecolors = 'none', c = 'r', s = 10, alpha = 0.5)
        ax.scatter(y2, x2, marker = '*', edgecolors = 'none', c = 'b', s = 10, alpha = 0.5)
        ax.set_title("Distance matching", fontsize = 8)
        plt.savefig("%s/TESTING/fforster/astrometry/test_%s_4.png" % (webdir, pixscale == 1))

        fig, ax = plt.subplots()
        ax.scatter(y1 - y2, x1 - x2, marker = 'o', edgecolors = 'none', c = 'r', s = 10, alpha = 0.5)
        ax.set_title("Differences after matching", fontsize = 8)
        plt.savefig("%s/TESTING/fforster/astrometry/test_%s_5.png" % (webdir, pixscale == 1))


    # find new highest concentration
    nsep = np.zeros(len(x1))
    for i in range(len(x1)):
        nsep[i] = np.sum((np.sqrt((x1[i] - x2[i])**2 + (y1[i] - y2[i])**2) < 5. * pixscale))
    idxnsep = np.argmax(nsep)
    deltasepxcomp = np.median((x1 - x2)[nsep == max(nsep)])
    deltasepycomp = np.median((y1 - y2)[nsep == max(nsep)])

    # select matched sources, removing outliers
    dist = np.sqrt((x1 - x2 - deltasepxcomp)**2 + (y1 - y2 - deltasepycomp)**2)
    distmask = (dist < 5. * pixscale)  # 5 * pixscale
    if order == 1:
        nptmin = 3
    elif order == 2:
        nptmin = 6
    elif order == 3:
        nptmin = 10
    if np.sum(distmask) < nptmin:
        distmask = (dist < 10. * pixscale)
    
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

    if testastrometry:
        print "Stars after distance filtering: %i" % np.sum(distmask)

        fig, ax = plt.subplots(figsize = (12, 6))
        ax.scatter(y1, x1, marker = 'o', edgecolors = 'none', c = 'r', s = 10, alpha = 0.5)
        ax.scatter(y2, x2, marker = '*', edgecolors = 'none', c = 'b', s = 10, alpha = 0.5)
        ax.set_title("Positions after matching and filtering", fontsize = 8)
        plt.savefig("%s/TESTING/fforster/astrometry/test_%s_6.png" % (webdir, pixscale == 1))

        fig, ax = plt.subplots()
        ax.scatter(y1 - y2, x1 - x2, marker = '.', edgecolors = 'none', c = 'b', s = 10, alpha = 0.5)
        ax.scatter(deltasepycomp + deltay, deltasepxcomp + deltax, marker = 'o', s = 100, facecolors = 'none')
        ax.set_title("Differences after matching and filtering", fontsize = 8)
        plt.savefig("%s/TESTING/fforster/astrometry/test_%s_7.png" % (webdir, pixscale == 1))

    print "Number of star coincidences: ", len(x1), len(y1)
    # find best transformation relating all these points
    sol_astrometry = findtransformation(order, x1, y1, x2, y2)
    
    # compute new variables and return together with rms
    (x1t, y1t) = applytransformation(order, x1, y1, sol_astrometry)

    # root mean squared error
    rms = np.sqrt(np.sum((x1t - x2)**2 + (y1t - y2)**2) / len(x1))

    # PHOTOMETRY
    # ------------------------------------------------
    
    
     # find best constant relating the star fluxes (note that zero point in sextractor might be different to our zero point)
    if pixscale != 1.:
        maskflux = np.isfinite(z1) & np.isfinite(z2) & np.isfinite(e_z1) & np.isfinite(e_z2)
    else:
        ratioz = z1 / z2
        ratiozMAD = np.median(np.abs(np.median(ratioz) - ratioz))
        maskflux = np.isfinite(z1) & np.isfinite(z2) & np.isfinite(e_z1) & np.isfinite(e_z2) & (z1 < 5e5) & (z2 < 5e5) & (z1 > 500) \
                   & (z2 > 500) & (ratioz > np.median(ratioz) - 3. * ratiozMAD) & (ratioz < np.median(ratioz) + 3. * ratiozMAD)
    aflux = np.dot(z1[maskflux], z2[maskflux]) / np.dot(z1[maskflux], z1[maskflux])
    e_aflux = np.dot(z1[maskflux], e_z2[maskflux]) / np.dot(z1[maskflux], z1[maskflux])


    if pixscale != 1.:

        return (x1, y1, r1, z1, e_z1, x2, y2, r2, z2, e_z2, deltax, deltay, x1t, y1t, rms, aflux, e_aflux, sol_astrometry)

    else:

        print "pixscale: ", pixscale

#        maskflux = np.isfinite(z1) & np.isfinite(z2) & np.isfinite(e_z1) & np.isfinite(e_z2) & (z1 < 5e5) & (z2 < 5e5) & (z1 > 500) & (z2 > 500)
#        aflux = np.median(z1[maskflux] / z2[maskflux])

        diff = (np.log10(z1[maskflux]) - np.log10(aflux * z2[maskflux]))**2
        maskdiff= diff <= np.percentile(diff, 95)

        aflux = np.dot(z1[maskflux][maskdiff], z2[maskflux][maskdiff]) / np.dot(z1[maskflux][maskdiff], z1[maskflux][maskdiff])
        e_aflux = np.dot(z1[maskflux][maskdiff], e_z2[maskflux][maskdiff]) / np.dot(z1[maskflux][maskdiff], z1[maskflux][maskdiff])

        maskflux = np.isfinite(z1) & np.isfinite(z2) & np.isfinite(e_z1) & np.isfinite(e_z2) & (z1 < 5e5) & (z2 < 5e5)

        diff = (np.log10(z1[maskflux]) - np.log10(aflux * z2[maskflux]))**2
        maskdiff= diff <= np.percentile(diff, 95)
        
        return (x1, y1, r1, z1, e_z1, x2, y2, r2, z2, e_z2, deltax, deltay, x1t, y1t, rms, aflux, e_aflux, sol_astrometry)
        #return (x1[maskflux][maskdiff], y1[maskflux][maskdiff], r1[maskflux][maskdiff], z1[maskflux][maskdiff], e_z1[maskflux][maskdiff], x2[maskflux][maskdiff], y2[maskflux][maskdiff], r2[maskflux][maskdiff], z2[maskflux][maskdiff], e_z2[maskflux][maskdiff], deltax, deltay, x1t[maskflux][maskdiff], y1t[maskflux][maskdiff], rms, aflux, e_aflux, sol_astrometry)

# project image into new grid, uses radial weights or lanczos interpolation
def doproject(image, varimage, sol_astrometry, useweights):

    # estimate background to replace bad pixels with background level
    background = dobg(image, 'median', None, True, 0)
    image[image > nmax] = background[image > nmax]

    image = image.transpose()
    varimage = varimage.transpose()
    
    imagenew = np.zeros(np.shape(image))
    varimagenew = np.zeros(np.shape(image))

    if verbose:
        print "\nProjecting image..."

    print np.shape(n1), np.shape(dqn1), np.shape(varn1)
    print np.shape(image), np.shape(dqn2), np.shape(varimage)
    print np.shape(imagenew), np.shape(dqn2), np.shape(varimagenew)
    print alanczos, nx, ny, order, sol_astrometry, gain

    if resampling[0:7] == 'lanczos':
        if verbose:
            print "Entering fortran projection routine..."
        if astrometry == 'pix2pix':
            projection.lanczos(alanczos, nx, ny, order, sol_astrometry, image, varimage, gain)
        elif astrometry == 'WCS':
            projectionwcs.lanczos(alanczos, nx, ny, image, varimage)
    else:
        print "Resampling not recognized..."
        sys.exit(1)
    if verbose:
        print "Leaving fortran, image projected..."

    print "\n\nImage shapes:"
    print "   n1, dqn1, varn1:", np.shape(n1), np.shape(dqn1), np.shape(varn1)
    print "   image, dqn2, varimage:", np.shape(image), np.shape(dqn2), np.shape(varimage)

    if astrometry == 'pix2pix':
        imagenew = projection.imageout[0:nx, 0:ny]
        varimagenew = projection.varimageout[0:nx, 0:ny]
        imagenew = float32(imagenew.transpose())
        varimagenew = float32(varimagenew.transpose())
    elif astrometry == 'WCS':
        imagenew = projectionwcs.imageout.transpose()
        varimagenew = projectionwcs.varimageout.transpose()
        imagenew = float32(imagenew[0:ny, 0:nx])
        varimagenew = float32(varimagenew[0:ny, 0:nx])

    print "   imagenew, dqn2, varimagenew:", np.shape(imagenew), np.shape(dqn2), np.shape(varimagenew)


#    # update header
#    (fits1[0].header).update('MJD-OBS', "%s" % MJDproc)
#    try:
#        (fits1[0].header).update('AIRMASS', "%s" % AIRMASSproc)
#        (fits1[0].header).update('EXPTIME', "%s" % EXPTIMEproc)
    # update header

    fits1[0].header['MJD-OBS'] = "%s" % MJDproc
    try:
        fits1[0].header['AIRMASS'] = "%s" % AIRMASSproc
        fits1[0].header['EXPTIME'] = "%s" % EXPTIMEproc
    except:
        "\n\nWARNING: Cannot find airmass or exposure time in header.\n\n"

    # save new data
    fits1[0].data = imagenew
    fits1.writeto("%s_image_crblaster_grid%02i_%s.fits" % (file2.replace(indir, outdir), fileref, resampling), clobber = True)
    fits1[0].data = varimagenew
    fits1.writeto("%s_varmap_grid%02i_%s.fits" % (file2.replace(indir, outdir), fileref, resampling), clobber = True)

    return MJDproc, imagenew, varimagenew


# compute convolution kernel using stars
# when conv1st = True, compute kernel to transform 1 into 2
def dokernel(ipixref, jpixref, radius, fluxmin, fluxmax, x1, y1, r1, conv1st):
    
    if verbose:
        print "    dokernel, ipix: %i, jpix: %i, radius: %i, fluxmin: %i, fluxmax: %i, conv1st: %s" % (ipixref, jpixref, radius, fluxmin, fluxmax, conv1st)

    # select stars close to reference position
    distref = np.sqrt((x1 - ipixref)**2 + (y1 - jpixref)**2)
    maskclose = np.array((distref < radius) & (x1 > dn) & (x1 < nx - dn) & (y1 > dn) & (y1 < ny - dn))
    x1close = x1[maskclose]
    y1close = y1[maskclose]
    r1close = r1[maskclose]
    nclose = len(x1close)
    flux1 = []
    flux2 = []
    if verbose:
        print "          #stars after distance cuts:", nclose

    # filter stars according 
    for i in range(nclose):

        dnw = 4
        maxvar = np.max([np.max(varn1[int(round(y1close[i]) - dnw): int(round(y1close[i]) + dnw),
                                      int(round(x1close[i]) - dnw): int(round(x1close[i]) + dnw)]),
                         np.max(varn2[int(round(y1close[i]) - dnw): int(round(y1close[i]) + dnw),
                                      int(round(x1close[i]) - dnw): int(round(x1close[i]) + dnw)])])

        psf2 = n2[int(round(y1close[i]) - dn): int(round(y1close[i]) + dn), int(round(x1close[i]) - dn): int(round(x1close[i]) + dn)]
        psf1 = n1[int(round(y1close[i]) - dn): int(round(y1close[i]) + dn), int(round(x1close[i]) - dn): int(round(x1close[i]) + dn)]

        # find the radius of the maximum signal to noise ratio pixel in the image 
        # if there are not pixels which have more than three psf standard deviations, do not select
        if np.sum(psf1.flatten() > 3. * np.std(psf1)) > 0:
            r1signalmax = np.max(rs2Dstars[psf1.flatten() > 3. * np.std(psf1)])
        else:
            r1signalmax = -1

        if np.sum(psf2.flatten() > 3. * np.std(psf2)) > 0:
            r2signalmax = np.max(rs2Dstars[psf2.flatten() > 3. * np.std(psf2)])
        else:
            r2signalmax = -1

        # for a star to be selected:
        #  1. radius of brightest SNR pixel must be lower than given value in both images
        #  2. maximim radius of pixels which are above three times the psf standard deviation must be below npsf / 4
        #  3. there should be no cosmic rays
        #  4. maximum variance must be below varmax
        maxSNRradius = 5
        if rs2Dstars[np.argmax(psf1)] < maxSNRradius and rs2Dstars[np.argmax(psf2)] < maxSNRradius and r1signalmax < npsf / 4. and r2signalmax < npsf / 4. and CR(psf1, npsf + nf) < 30 and CR(psf2, npsf + nf) < 30 and r1signalmax > 0 and r2signalmax > 0 and maxvar < varmax:
            flux1.append(n1[int(y1close[i]), int(x1close[i])])
            flux2.append(n2[int(y1close[i]), int(x1close[i])])
        else:
            flux1.append(fluxmax)
            flux2.append(fluxmax)

    flux1 = np.array(flux1)
    flux2 = np.array(flux2)
    maskflux12 = (flux1 > fluxmin) & (flux2 > fluxmin) & (flux1 < fluxmax) & (flux2 < fluxmax)
    x1close = x1close[maskflux12] 
    y1close = y1close[maskflux12]
    r1close = r1close[maskflux12]
    flux = flux1[maskflux12]
    nclose = len(x1close)
    if verbose:
        print "          #stars after SNR, CRs, variance and flux cuts:", nclose
    
    while nclose == 0:
        radius += 150
        if verbose:
            print "          Trying new radius to select more stars: %i" % radius
        maskclose = np.array((distref < radius) & (x1 > dn) & (x1 <= nx - dn))
        x1close = x1[maskclose]
        y1close = y1[maskclose]
        r1close = r1[maskclose]
        nclose = len(x1close)
        if verbose:
            print "                #stars after distance cuts:", nclose

        flux1 = []
        flux2 = []
        for i in range(nclose):

            dnw = 4
            maxvar = np.max([np.max(varn1[int(round(y1close[i]) - dnw): int(round(y1close[i]) + dnw), int(round(x1close[i]) - dnw): int(round(x1close[i]) + dnw)]),
                            np.max(varn2[int(round(y1close[i]) - dnw): int(round(y1close[i]) + dnw), int(round(x1close[i]) - dnw): int(round(x1close[i]) + dnw)])])

            psf1 = n1[int(round(y1close[i]) - dn): int(round(y1close[i]) + dn), int(round(x1close[i]) - dn): int(round(x1close[i]) + dn)]
            psf2 = n2[int(round(y1close[i]) - dn): int(round(y1close[i]) + dn), int(round(x1close[i]) - dn): int(round(x1close[i]) + dn)]

            if np.sum(psf1.flatten() > 3. * np.std(psf1)) > 0:
                r1signalmax = np.max(rs2Dstars[psf1.flatten() > 3. * np.std(psf1)])
            else:
                r1signalmax = -1
            if np.sum(psf2.flatten() > 3. * np.std(psf2)) > 0:
                r2signalmax = np.max(rs2Dstars[psf2.flatten() > 3. * np.std(psf2)])
            else:
                r2signalmax = -1

            if rs2Dstars[np.argmax(psf1)] < 5 and rs2Dstars[np.argmax(psf2)] < 5 and r1signalmax < npsf / 4. and r2signalmax < npsf / 4. and CR(psf1, npsf + nf) < 30 and CR(psf2, npsf + nf) < 30 and r1signalmax > 0 and r2signalmax > 0 and maxvar < varmax:
                flux1.append(n1[int(y1close[i]), int(x1close[i])])
                flux2.append(n2[int(y1close[i]), int(x1close[i])])
            else:
                flux1.append(fluxmax)
                flux2.append(fluxmax)

        flux1 = np.array(flux1)
        flux2 = np.array(flux2)
        maskflux12 = (flux1 > fluxmin) & (flux1 < fluxmax) & (flux1 > fluxmin) & (flux1 < fluxmax)
        x1close = x1close[maskflux12]
        y1close = y1close[maskflux12]
        r1close = r1close[maskflux12]
        flux = flux1[maskflux12]
        nclose = len(x1close)

        if verbose:
            print "                #stars after SNR, CRs, variance and flux cuts:", nclose

        if radius > np.sqrt(nx * ny):
            print "\n\nWARNING: Cannot find stars to train the kernel\n\n"
            sys.exit(17)

    if verbose:
        print "    Selected %i stars" % (nclose)

    # start building kernel
    X = np.zeros((nclose * npsf2, nvar))
    Y = np.zeros(nclose * npsf2)
    RSS1 = np.zeros(nclose * npsf2)

    if savestars:
        psf1s = None
        psf2s = None

    # loop among sources to compute 1st filter
    for i in range(nclose):

        # round values
        ipix = np.round(x1close[i])
        jpix = np.round(y1close[i])

        # print central values
        #print ipix, jpix, n1[jpix, ipix], n2[jpix, ipix]

        # take stamp
        psf1 = n1[int(jpix - dn): int(jpix + dn), int(ipix - dn): int(ipix + dn)]
        psf2 = n2[int(jpix - dn): int(jpix + dn), int(ipix - dn): int(ipix + dn)]

        if savestars:
            if psf1s is None:
                psf1s = psf1
                psf2s = psf2
            else:
                psf1s = np.dstack([psf1s, psf1])
                psf2s = np.dstack([psf2s, psf2])
        
        # fill X and Y
        for k in range(nf):
            for l in range(nf):
                ivar = ivarf[k, l]
                if ivar == -1:
                    continue
                if conv1st:
                    X[i * npsf2: (i + 1) * npsf2, ivar] \
                        = X[i * npsf2: (i + 1) * npsf2, ivar] + psf1[ieq2i + k, ieq2j + l]
                else:
                    X[i * npsf2: (i + 1) * npsf2, ivar] \
                        = X[i * npsf2: (i + 1) * npsf2, ivar] + psf2[ieq2i + k, ieq2j + l]
        if conv1st:
            Y[i * npsf2: (i + 1) * npsf2] = psf2[nfh: -(nfh + 1), nfh: -(nfh + 1)][ieq2i, ieq2j]
        else:
            Y[i * npsf2: (i + 1) * npsf2] = psf1[nfh: -(nfh + 1), nfh: -(nfh + 1)][ieq2i, ieq2j]

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
    solfilter = np.zeros((nf, nf))
    rfilter = np.zeros((nf, nf))
    for k in range(nf):
        for l in range(nf):
            ivar = int(ivarf[k, l])
            if ivar == -1:
                solfilter[k, l] = 0
                continue
            solfilter[k, l] = solvars[ivar]
            rfilter[k, l] = np.sqrt((k - nf / 2.)**2 + (l - nf / 2.)**2)

    # compute filter characteristics
    kratio = np.sum(solvars) / np.sum(np.abs(solvars))
    ksupport = np.sum(solfilter * rfilter) / np.sum(solfilter)
    knorm = np.sum(solfilter)
    knorm2 = knorm * knorm
    knormsum2 = np.sum(solfilter**2)

    # save train stars and filter
    if savestars:
        fileout = "%s/%s/%s/CALIBRATIONS/stars_%s_%s_%02i-%02i_%04i-%04i.npy" % (sharedir, field, CCD, field, CCD, filesci, fileref, ipixref, jpixref)
        np.save(fileout, np.dstack([psf1s, psf2s]))
        fileout = "%s/%s/%s/CALIBRATIONS/kernel_%s_%s_%02i-%02i_%04i-%04i.npy" % (sharedir, field, CCD, field, CCD, filesci, fileref, ipixref, jpixref)
        np.save(fileout, solfilter)
        
    return nclose, solfilter, kratio, ksupport, knorm2, knormsum2


# do optimal photometry given 1D or 2D psf, flux, var and 1D mask
def getoptphot(psf, flux, var, mask):
    
    auxvar = (psf.flatten())[mask] / (np.abs(var.flatten()))[mask]
    optf = np.sum(auxvar * (flux.flatten())[mask])
    var_optf = np.sum(auxvar * (psf.flatten())[mask])
    optf = optf / var_optf
    var_optf = 1. / var_optf
    
    return optf, var_optf
    

        
#############################################################
###############        MAIN PROGRAM        ##################                            #
#############################################################




# Do projection, convolution and difference ##########
######################################################

printtime("dodiff")

if dodiff:

    # send no wait command to look for matches among Minor Planets
    # ------------------------------------------------------------
    
    doMP = True
    if dolook and doMP:
        fileheader = "%s/LISTHEAD/%s_%s_%02i_listhead.txt" % (sharedir, field, CCD, filesci)
        if docrblaster:
            command = "listhead %s/%s/%s/%s_%s_%02i_image.fits.fz | egrep 'NAXIS1|NAXIS2|CD1_1|CD1_2|CD2_1|CD2_2|CRPIX1|CRPIX2|CRVAL1|CRVAL2|nPV1|nPV2|PV1_0|PV1_0|PV1_1|PV1_2|PV1_3|PV1_4|PV1_5|PV1_6|PV1_7|PV1_8|PV1_9|PV1_10|PV1_11|PV2_0|PV2_1|PV2_2|PV2_3|PV2_4|PV2_5|PV2_6|PV2_7|PV2_8|PV2_9|PV2_10|PV2_11|MJD-OBS' | cut -f 1 -d '/' > %s" % (indir, field, CCD, field, CCD, filesci, fileheader)
        else:
            command = "listhead %s/%s/%s/%s_%s_%02i_image_crblaster.fits | egrep 'NAXIS1|NAXIS2|CD1_1|CD1_2|CD2_1|CD2_2|CRPIX1|CRPIX2|CRVAL1|CRVAL2|nPV1|nPV2|PV1_0|PV1_0|PV1_1|PV1_2|PV1_3|PV1_4|PV1_5|PV1_6|PV1_7|PV1_8|PV1_9|PV1_10|PV1_11|PV2_0|PV2_1|PV2_2|PV2_3|PV2_4|PV2_5|PV2_6|PV2_7|PV2_8|PV2_9|PV2_10|PV2_11|MJD-OBS' | cut -f 1 -d '/' > %s" % (indir, field, CCD, field, CCD, filesci, fileheader)

        if verbose:
            print command
        os.system(command)
        # only MP
        L = np.array(['python', 'dolook.py', field, CCD, filesci, '10', nx, ny, fileheader, sharedir, 'MP', verbose])
        if verbose:
            print L
        os.spawnvpe(os.P_NOWAIT, 'python', L.tolist(), os.environ)


    # run sextractor to estimate new background
    # -----------------------------------------

    if not os.path.exists("%s_image_crblaster.fits-catalogue%s.dat" % (file2cat, cataloguestring)) and filesci != 0:
        if useweights == 'external' or useweights == 'internal' or useweights == 'own':
            command = "sex -c %s/etc/default.sex %s_image_crblaster.fits -CATALOG_NAME %s_image_crblaster.fits-catalogue%s.dat -WEIGHT_IMAGE %s_varmap.fits.fz -BACK_SIZE %i -VERBOSE_TYPE %s" % (hitsdir, file2, file2cat, cataloguestring, file2, backsize, sexverbose)
            command = "sex -c %s/etc/default.sex %s_image_crblaster.fits -CATALOG_NAME %s_image_crblaster.fits-catalogue%s.dat -WEIGHT_TYPE BACKGROUND -BACK_SIZE %i -VERBOSE_TYPE %s" % (hitsdir, file2, file2cat, cataloguestring, backsize, sexverbose)
            print command
        else:
            command = "sex -c %s/etc/default.sex %s_image_crblaster.fits -CATALOG_NAME %s_image_crblaster.fits-catalogue%s.dat -WEIGHT_TYPE BACKGROUND -BACK_SIZE %i -VERBOSE_TYPE %s" % (hitsdir, file2, file2cat, cataloguestring, backsize, sexverbose)
        if verbose:
            print "    %s\n" % command
        os.system(command)

#        # clean varmap
#        if useweights == 'own':
#            command = "rm %s_wtmap.fits.fz" % file2
#            if verbose:
#                print "    %s\n" % command
#            os.system(command)

    else:
        if verbose:
            print "\nSextractor already run on %s" % file2

    (x1sex, y1sex, z1sex, e_z1sex, r1sex, f1sex) = np.loadtxt("%s_image_crblaster.fits-catalogue%s.dat" % (file1cat, cataloguestring), usecols = (1, 2, 5, 6, 8, 9)).transpose()
    (x2, y2, z2, e_z2, r2, f2) = np.loadtxt("%s_image_crblaster.fits-catalogue%s.dat" % (file2cat, cataloguestring), usecols = (1, 2, 5, 6, 8, 9)).transpose()
    maskflag1 = (f1sex == 0) & (r1sex < 5) & (x1sex > 30) & (x1sex < nx - 30) & (y1sex > 30) & (y1sex < ny - 30)
    maskflag2 = (f2 == 0) & (r2 < 5) & (x2 > 30) & (x2 < nx - 30) & (y2 > 30) & (y2 < ny - 30)
    maskflag1 = (x1sex > 30) & (x1sex < nx - 30) & (y1sex > 30) & (y1sex < ny - 30)
    maskflag2 = (x2 > 30) & (x2 < nx - 30) & (y2 > 30) & (y2 < ny - 30)
    x1 = x1sex[maskflag1]
    y1 = y1sex[maskflag1]
    z1 = z1sex[maskflag1]
    e_z1 = e_z1sex[maskflag1]
    r1 = r1sex[maskflag1]
    x2 = x2[maskflag2]
    y2 = y2[maskflag2]
    z2 = z2[maskflag2]
    e_z2 = e_z2[maskflag2]
    r2 = r2[maskflag2]
    
    # Pixel coordinate matching
    # -------------------------
    
    if verbose:
        print "\nMatching sources and finding transformation...\n"

    rcrowd = np.sqrt(nx * ny / np.sqrt(len(x1) * len(x2)) / np.pi)
    pixscale = 1.
    nstarmin = 100.
    # try to select the 150 brightest stars in each set to start matching
    matched = match(100, pixscale, order, x1, y1, z1, e_z1, r1, x2, y2, z2, e_z2, r2, 100, 100, 0, nx, 0, ny, rcrowd, rcrowd,
                    np.percentile(z1, 100. * (1. - nstarmin / len(z1))), min(1e6, np.percentile(z1, 100)),
                    np.percentile(z2, 100. * (1. - nstarmin / len(z2))), min(1e6, np.percentile(z2, 100)))
#    matched = match(100, pixscale, order, x1, y1, z1, e_z1, r1, x2, y2, z2, e_z2, r2, 100, 100, 0, nx, 0, ny, rcrowd, rcrowd,
#                    np.percentile(z1, max(0, min(100. * (1. - nmin / len(z1)), 90))), np.percentile(z1, 100),
#                    np.percentile(z2, max(0, min(100. * (1. - nmin / len(z2)), 90))), np.percentile(z2, 100))
#    matched = match(100, pixscale, order, x1, y1, z1, e_z1, r1, x2, y2, z2, e_z2, r2, 100, 100, 0, nx, 0, ny, rcrowd, rcrowd, np.percentile(z1, 10), np.percentile(z1, 90), np.percentile(z2, 10), np.percentile(z2, 90))

    if not (matched is None):
        (x1sel, y1sel, r1sel, z1sel, e_z1sel, x2sel, y2sel, r2sel, z2sel, e_z2sel, \
         dx, dy, x1t, y1t, rms, aflux, e_aflux, sol_astrometry) = matched
    else:
        print "\n\nWARNING: Cannot find astrometric solution in field %s, CCD %s, between epochs %i and %i.\n\n" % (field, CCD, fileref, filesci)
        sys.exit(18)

    dxpix = int(dx)
    dypix = int(dy)
    r1median = np.median(r1)
    minr1 = r1median - 2. * np.median(np.abs(r1 - r1median))
    maxr1 = min(1.5 * r1median, r1median + 2. * np.median(np.abs(r1 - r1median)))
    r2median = np.median(r2)
    minr2 = r2median - 2. * np.median(np.abs(r2 - r2median))
    maxr2 = min(1.5 * r2median, r2median + 2. * np.median(np.abs(r2 - r2median)))
    conv1st = (r1median < r2median)

    if verbose:
        print "\nReference to science image pixel transformation\n-----------------------------------------------"
        print "O(1) terms:", sol_astrometry[2:6]
        if order > 1:
            print "O(2) terms:", sol_astrometry[6:12]
        if order > 2:
            print "O(3) terms:", sol_astrometry[12:20]
        print "dx: %f, dy: %f, a1: %f, a2: %f" % (dx, dy, sol_astrometry[0], sol_astrometry[1])
        print "rms: %f pixels (using %i sources)" % (rms, len(x1sel))
        print "median r1: %f, median r2: %f" % (r1median, r2median)
        print "aflux: %f +- %f" % (aflux, e_aflux)
    
    # exit if not good matching
    if len(x1sel) < 10 or rms >= 2:
        print "\n\nWARNING: Not enough stars or rms of astrometric solution too big (%i objects, rms: %f pixels)\n\n" % (len(x1sel), rms)
        sys.exit(19)

    fig, ax = plt.subplots(1, 2, figsize = (14, 7))
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[1].set_xlabel("dx")
    ax[1].set_ylabel("dy")
    
    # plot selected objects
    ax[0].scatter(x1sel, y1sel, marker = 'o', c = 'r', s = 1, edgecolors = 'none')
    ax[0].scatter(x2sel, y2sel, marker = 'o', c = 'b', s = 30, edgecolors = 'none', alpha = 0.2)
    ax[0].scatter(x1t, y1t, marker = 'o', c = 'k', s = 1, edgecolors = 'none')
    ax[1].scatter(x1t - x2sel, y1t - y2sel, marker = 'o', c = 'k', s = 1, edgecolors = 'none')
    #np.savetxt("/home/apps/astro/WEB/TESTING/fforster/test.dat", np.vstack([x1sel, x1t, x2sel, y1sel, y1t, y2sel]).transpose())
    #sys.exit()
    for ix in range(len(x1t)):
        ax[0].plot([x1t[ix], x1sel[ix]], [y1t[ix], y1sel[ix]], 'r')

    # add text
    for j in range(len(x1sel)):
        ax[0].text(x1sel[j], y1sel[j], "%i" % j, fontsize = 5, color = 'b', alpha = 0.5)
    
    # set title
    ax[1].set_title("RMS: %f" % rms, fontsize = 8)
    ax[0].set_xlim(min(x1sel), max(x1sel))
    ax[0].set_ylim(min(y1sel), max(y1sel))

    # save figure
    plt.savefig("%s/%s/%s/CALIBRATIONS/match_%s_%s_%02i-%02i.png" % (webdir, field, CCD, field, CCD, filesci, fileref), dpi = 120)

    # plot selected objects
    maskz = (z1sel > 0) & (z2sel > 0)
    try:
        # plot photometry of selected objects
        fig, ax = plt.subplots(figsize = (6, 4))
        ax.scatter(z1sel[maskz], z2sel[maskz], marker = 'o', c = 'r', s = 5, alpha = 0.5, edgecolors = 'none')
        ax.plot(np.array([min(z1sel[maskz]), max(z1sel[maskz])]), aflux * np.array([min(z1sel), max(z1sel)]), 'gray')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title("Photometric calibration (aflux: %f)" % aflux, fontsize = 8)
        #ax.set_xlim(min(z1sel), max(z1sel))
        #ax.set_ylim(min(z2sel), max(z2sel))
        ax.set_xlabel("ADU %i" % fileref, fontsize = 8)
        ax.set_ylabel("ADU %i" % filesci, fontsize = 8)
        plt.grid(True)
        
        # save figure
        plt.savefig("%s/%s/%s/CALIBRATIONS/match_%s_%s_%02i-%02i_flux.png" % (webdir, field, CCD, field, CCD, filesci, fileref), bbox_inches = 'tight')
    except:
        print "\n\nWARNING: Cannot plot photometric solution in ADUs\n\n"
                                                                                                                                           
    
    # save sources and solution
    np.save("%s/%s/%s/CALIBRATIONS/match_%s_%s_%02i-%02i.npy" % (sharedir, field, CCD, field, CCD, filesci, fileref), np.hstack((np.array([aflux, e_aflux, rms, order]), sol_astrometry)))
    np.save("%s/%s/%s/CALIBRATIONS/sources_%s_%s_%02i-%02i.npy" % (sharedir, field, CCD, field, CCD, filesci, fileref), np.vstack([x1sel, y1sel, r1sel, z1sel, e_z1sel, x2sel, y2sel, r2sel, z2sel, e_z2sel]))

    # find coordinate transformation using USNO catalogue and astrometric solution found in the header
    (RAguess, DECguess) = RADEC(x1sel, y1sel, CD[0, 0, 0], CD[0, 0, 1], CD[0, 1, 0], CD[0, 1, 1], CRPIX[0, 0], CRPIX[0, 1], CRVAL[0, 0], CRVAL[0, 1], PV[0])


    # Celestial coordinate matching
    # -----------------------------
    
    if verbose:
        print "\n\nMatching celestial coordinates...\n\n"

    arcsec = 1. / 60. / 60. # in deg
    pixscale = arcsec
    compphoto = USNO_B
    maskUSNO = (USNORA > min(RAguess)) & (USNORA < max(RAguess)) & (USNODEC > min(DECguess)) & (USNODEC < max(DECguess))
    rcrowd1 = np.sqrt(15. * (max(RAguess) - min(RAguess)) * (max(DECguess) - min(DECguess)) / len(RAguess) / np.pi) / 2.
    rcrowd2 = np.sqrt(15. * (max(RAguess) - min(RAguess)) * (max(DECguess) - min(DECguess)) / len(USNORA) / np.pi) / 2.
    if len(z1sel) > len(USNO_B):
        print "z1sel > USNO_B"
        z2lim = 0
        z1lim = 1. - (1. - z2lim) * len(USNO_B) / len(z1sel)
    else:
        print "z1sel < USNO_B"
        z1lim = 0
        z2lim = 0#1. - (1. - z1lim) * len(z1sel) / len(USNO_B)

    matched = match(200, 0.27 * arcsec, order, 15. * RAguess, DECguess, z1sel, e_z1sel, r1sel,
                    15. * USNORA[maskUSNO], USNODEC[maskUSNO], 10**(-USNO_B[maskUSNO] / 2.5), None, None,
                    15. * arcsec, 15. * arcsec, 15. * min(RAguess), 15. * max(RAguess), min(DECguess), max(DECguess),
                    rcrowd1, rcrowd2,
                    np.percentile(z1, 100. * z1lim), max(z1sel),
                    10**(-np.percentile(USNO_B, 100. * (1. - z2lim)) / 2.5), 10**(-min(USNO_B) / 2.5))

    if not (matched is None):
        (RA1sel, DEC1sel, r1seldeg, z1seldeg, e_z1seldeg, RA2sel, DEC2sel, r2seldeg, z2seldeg, e_z2seldeg, \
         dRA, dDEC, RA1t, DEC1t, rmsdeg, afluxADUB, e_afluxADUB, sol_astrometry_RADEC) = matched

        print "\nReference image to USNO catalogue RA-DEC transformation\n-----------------------------------------------------------"
        print "O(1) terms:", sol_astrometry_RADEC[2:6]
        if order > 1:
            print "O(2) terms:", sol_astrometry_RADEC[6:12]
        if order > 2:
            print "O(3) terms:", sol_astrometry_RADEC[12:20]
        print "dx: %f, dy: %f, a1deg: %f, a2deg: %f" % (dRA, dDEC, sol_astrometry_RADEC[0], sol_astrometry_RADEC[1])
        print "rms: %f arcsec (using %i sources)" % (rmsdeg * 60. * 60., len(RA1sel))
        print "aflux: %e +- %e" % (afluxADUB, e_afluxADUB)

        # plot match with USNO catalogue in RA DEC 
        fig, ax = plt.subplots()
        ax.scatter(RA1sel / 15., DEC1sel, marker = 'o', c = 'r', s = 1, edgecolors = 'none')
        ax.scatter(RA2sel / 15., DEC2sel, marker = 'o', c = 'b', s = 30, edgecolors = 'none', alpha = 0.2)
        ax.scatter(RA1t / 15., DEC1t, marker = 'o', c = 'k', s = 1, edgecolors = 'none')
        for ix in range(len(RA1t)):
            ax.plot([RA1t[ix] / 15., RA1sel[ix] / 15.], [DEC1t[ix], DEC1sel[ix]], 'r')
        for j in range(len(RA1sel)):
            ax.text(RA1sel[j] / 15., DEC1sel[j], "%i" % j, fontsize = 5, color = 'b', alpha = 0.5)
        ax.set_xlim(min(RAguess), max(RAguess))
        ax.set_ylim(min(DECguess), max(DECguess))
        ax.set_title("rms: %f arcsec" % (rmsdeg * 60. * 60.), fontsize = 8)
        plt.savefig("%s/%s/%s/CALIBRATIONS/USNO_%s_%s_%02i-%02i.png" % (webdir, field, CCD, field, CCD, filesci, fileref))

        # show flux calibration
        fig, ax = plt.subplots(figsize = (6, 4))
        ax.scatter(-2.5 * np.log10(z1seldeg * afluxADUB), -2.5 * np.log10(z2seldeg), marker = '.', c = 'r', edgecolors = 'none')
        ax.plot([24, 12], [24, 12], 'gray', alpha = 0.5)
        ax.set_title("USNO B vs fit", fontsize = 8)
        ax.set_xlabel('-2.5 log10(ADU ref) + %f' % (-2.5 * np.log10(afluxADUB)))
        ax.set_ylabel('USNO B')
        ax.set_xlim(24, 12)
        ax.set_ylim(24, 12)
        plt.grid(True)
        plt.savefig("%s/%s/%s/CALIBRATIONS/USNO_%s_%s_%02i-%02i_flux.png" % (webdir, field, CCD, field, CCD, filesci, fileref), bbox_inches = 'tight')

        # save sources and solution
        np.save("%s/%s/%s/CALIBRATIONS/matchRADEC_%s_%s_%02i-%02i.npy" % (sharedir, field, CCD, field, CCD, filesci, fileref), np.hstack((np.array([afluxADUB, e_afluxADUB, rmsdeg, CRVAL[0, 0], CRVAL[0, 1], CRPIX[0, 0], CRPIX[0, 1], CD[0, 0, 0], CD[0, 0, 1], CD[0, 1, 0], CD[0, 1, 1], nPV1, nPV2, order]), sol_astrometry_RADEC, PV[0].flatten())))
        np.save("%s/%s/%s/CALIBRATIONS/sourcesRADEC_%s_%s_%02i-%02i.npy" % (sharedir, field, CCD, field, CCD, filesci, fileref), np.vstack([RA1sel, DEC1sel, RA1sel, z1seldeg, e_z1seldeg, RA2sel, DEC2sel, r2seldeg, z2seldeg, e_z2seldeg]))

    else:

        print "WARNING: No celestial astrometric solution found"
        sys.exit(20)


    # Projection
    # -----------------------------

    printtime("projection")

    # project image given inverse transformation (transform coordinates from 1 to 2) and fits
    (MJDproc, n2, varn2) = doproject(image, varimage, sol_astrometry, useweights)

    (nx2, ny2) = np.shape(n2)
    n2 = n2[0:ny, 0:nx]
    varn2 = varn2[0:ny, 0:nx]

    varn2[np.invert(np.isfinite(varn2))] = varmax

    # save date after reading fits file in doproject
    date = MJDproc - MJDref
    if verbose:
        print "\nDays after reference date:", date

    # subtract background from second, projected image
    background2 = dobg(n2, backgroundtype, "%s_image_crblaster_grid%02i_%s.fits" % (file2.replace(indir, outdir), fileref, resampling), True, sol_astrometry[0])
    n2 = n2 - background2

    # mask bad pixels
    if dozeromask:
        n2[varn2 >= varmax] = 0

    # print the local time
    printtime("PCAstars")


    # Save star snapshots for PCA and artificial transients (add more after convolution filters for PCA)
    # -----------------------------------------------------

    psfPCA = []
    psfadd = []
    var_psfadd = []
    psfflux = []
    e_psfflux = []
    psfradius = []
    psfSNR = []
    psfref = np.zeros((npsf, npsf))

    print "np.shape(x1sel)", np.shape(x1sel)
    print "n1 median", np.median(n1)

    for istar in range(len(x1sel)):

        # if radius not too small, not too big
        if r1sel[istar] < minr1 or r1sel[istar] > maxr1 or r2sel[istar] < minr2 or r2sel[istar] > maxr2:
            continue
            
        # added star central coordinates
        ipix = np.round(x1sel[istar])
        jpix = np.round(y1sel[istar])

        # find closest star, if too close, ignore
        mask = (x1sel != x1sel[istar]) & (y1sel != y1sel[istar])
        dist2 = (x1sel[mask] - ipix)**2 + (y1sel[mask] - jpix)**2
        if np.min(dist2) < npsf2:
            continue

        # background
        bg1 = background1[int(jpix - npsfh - 1): int(jpix + npsfh), int(ipix - npsfh - 1): int(ipix + npsfh)]
        bg2 = background2[int(jpix - npsfh - 1): int(jpix + npsfh), int(ipix - npsfh - 1): int(ipix + npsfh)]

        # extract signal
        if conv1st:
            signal = n2[int(jpix - npsfh - 1): int(jpix + npsfh), int(ipix - npsfh - 1): int(ipix + npsfh)] # ADU
            variance = varn2[int(jpix - npsfh - 1): int(jpix + npsfh), int(ipix - npsfh - 1): int(ipix + npsfh)]

        else:
            signal = n1[int(jpix - npsfh - 1): int(jpix + npsfh), int(ipix - npsfh - 1): int(ipix + npsfh)] # ADU
            variance = varn1[int(jpix - npsfh - 1): int(jpix + npsfh), int(ipix - npsfh - 1): int(ipix + npsfh)]
            
        # try computing center of mass to decide adding or not
        try:
            rCM = np.sqrt((np.dot(X.flatten(), signal.flatten()) / np.sum(signal) - npsf / 2.)**2 \
                          + (np.dot(Y.flatten(), signal.flatten()) / np.sum(signal) - npsf / 2.)**2)

            print conv1st, np.min(signal + bg1 + bg2), np.sum(signal), rCM, max(r1median, r2median)

            if np.min(signal + bg1 + bg2) > 0 and np.sum(signal) > 0 and rCM < max(r1median, r2median):

                if filtername == 'u' and np.sum(signal) > 1e5:
                    continue
                elif filtername == 'g' and np.sum(signal) > 1e5:  #  for Kepler data I used 1e4
                    continue
                elif filtername == 'r' and np.sum(signal) > 1e5:
                    continue

                signalsnr = signal / np.sqrt((signal + bg1 + bg2) / gain) # SNR
                
                if np.isfinite(np.max(signalsnr)) and np.min(signalsnr) > -5:

                    psfadd.append(signal)
                    var_psfadd.append(variance)
                    
                    psfflux.append(z2sel[istar])
                    e_psfflux.append(e_z2sel[istar])
                    psfradius.append(r2sel[istar])
                    #psfSNR.append(np.max(signal) / np.sqrt((np.median(bg1) + np.median(bg2)) / gain))
                    psfSNR.append(z2sel[istar] / e_z2sel[istar])
                    psfPCA.append(np.concatenate([np.array([ipix, jpix]), signalsnr.flatten()]))

        except:

            print "Could not compute rCM."

            
    # Use comparison between optimal photometry and sextractor photometry to select real stars
    # -----------------------------------------------------------------------------

    psfadd = np.array(psfadd)
    psfref = np.sum(psfadd, axis = 0)
    psf = (psfref / np.sum(psfref)).flatten()

    optflux = np.zeros(len(psfflux))
    var_optflux = np.zeros(len(psfflux))

    for istar in range(len(psfflux)):
        optflux[istar], var_optflux[istar] = getoptphot(psf, psfadd[istar], var_psfadd[istar], maskpsf)

    ratioflux = optflux / psfflux
    maskpsfflux = (ratioflux > 1) 
    psfref = np.sum(psfadd[maskpsfflux], axis = 0)
    psf = (psfref / np.sum(psfref)).flatten()

    doplotoptphot = True
    if doplotoptphot:

        xlin = np.linspace(0, max(psfflux), 2)

        fig, ax = plt.subplots(ncols = 3, nrows = 2, figsize = (21, 12))
        for istar in range(len(psfflux)):

            ax[0, 0].errorbar(optflux[istar], psfflux[istar], xerr = np.sqrt(var_optflux[istar]), yerr = e_psfflux[istar])
            ax[0, 0].text(optflux[istar], psfflux[istar], "%i" % istar, fontsize = 7)
            ax[0, 1].scatter(optflux[istar], ratioflux[istar])
            ax[0, 1].text(optflux[istar], ratioflux[istar], "%i" % istar, fontsize = 7)
            ax[0, 2].scatter(psfradius[istar], ratioflux[istar])
            ax[0, 2].text(psfradius[istar], ratioflux[istar], "%i" % istar, fontsize = 7)

        ax[0, 0].set_xlabel("Optimal photometry")
        ax[0, 0].set_ylabel("Sextractor photometry")
        ax[0, 0].plot(xlin, xlin, c = 'k')
        ax[0, 0].plot(xlin, np.median(psfflux / optflux) * xlin, c = 'gray')
        ax[0, 0].plot(xlin, (np.median(psfflux / optflux) + np.median(np.abs(np.median(psfflux / optflux) - psfflux / optflux))) * xlin, c = 'gray')
        ax[0, 0].plot(xlin, (np.median(psfflux / optflux) - np.median(np.abs(np.median(psfflux / optflux) - psfflux / optflux))) * xlin, c = 'gray')
        ax[0, 1].set_xlabel("Optimal photometry")
        ax[0, 1].set_ylabel("Optimal photometry / sextractor photometry")
        ax[0, 1].axhline(1, c = 'gray')
        ax[0, 2].set_xlabel("Source radius [pix]")
        ax[0, 2].set_ylabel("Optimal photometry / sextractor photometry")
        ax[0, 2].axhline(1, c = 'gray')

    for istar in range(len(psfflux)):
        
        if not maskpsfflux[istar]:
            continue

        optflux[istar], var_optflux[istar] = getoptphot(psf, psfadd[istar], var_psfadd[istar], maskpsf)

    ratioflux = optflux / psfflux

    if doplotoptphot:
        for istar in range(len(psfflux)):

            if not maskpsfflux[istar]:
                continue
            
            ax[1, 0].errorbar(optflux[istar], psfflux[istar], xerr = np.sqrt(var_optflux[istar]), yerr = e_psfflux[istar])
            ax[1, 0].text(optflux[istar], psfflux[istar], "%i" % istar, fontsize = 7)
            ax[1, 1].errorbar(optflux[istar], ratioflux[istar], xerr = np.sqrt(var_optflux[istar]), yerr = np.sqrt(var_optflux[istar] / psfflux[istar]**2 + ratioflux[istar]**2 * (e_psfflux[istar] / psfflux[istar])**2))
            ax[1, 1].text(optflux[istar], ratioflux[istar], "%i" % istar, fontsize = 7)
            ax[1, 2].scatter(psfradius[istar], ratioflux[istar])
            ax[1, 2].text(psfradius[istar], ratioflux[istar], "%i" % istar, fontsize = 7)
           
        ax[1, 0].set_xlabel("Optimal photometry")
        ax[1, 0].set_ylabel("Sextractor photometry")
        ax[1, 0].plot(xlin, xlin, c = 'k')
        ax[1, 0].plot(xlin, np.median(psfflux / optflux) * xlin, c = 'gray')
        ax[1, 0].plot(xlin, (np.median(psfflux / optflux) + np.median(np.abs(np.median(psfflux / optflux) - psfflux / optflux))) * xlin, c = 'gray')
        ax[1, 0].plot(xlin, (np.median(psfflux / optflux) - np.median(np.abs(np.median(psfflux / optflux) - psfflux / optflux))) * xlin, c = 'gray')
        ax[1, 1].set_xlabel("Optimal photometry")
        ax[1, 1].set_ylabel("Optimal photometry / sextractor photometry")
        ax[1, 1].axhline(1, c = 'gray')
        ax[1, 2].set_xlabel("Source radius [pix]")
        ax[1, 2].set_ylabel("Optimal photometry / sextractor photometry")
        ax[1, 2].axhline(1, c = 'gray')

        plt.savefig("%s/TESTING/fforster/LIMITS/optphotVSsextractor_%s_%s_%02i-%02i.png" % (webdir, field, CCD, filesci, fileref))

    doplot = False
    if doplot:
        nplots = int(np.ceil((len(psfflux) + 1) / 10.))
        fig, ax = plt.subplots(10, nplots, figsize = (10. * (nplots / 10.), 10.))
        for iplot in range(nplots):
            for j in range(10):
                istar = iplot * 10 + j
                if istar - 1 < len(psfflux):
                    ax[j, iplot].axes.get_xaxis().set_visible(False)
                    ax[j, iplot].axes.get_yaxis().set_visible(False)
                    ax[j, iplot].imshow(psfadd[istar - 1], interpolation = 'nearest')
                    ax[j, iplot].text(0, 0, "%i" % (istar - 1), fontsize = 7)
                    if not maskpsfflux[istar - 1] and istar != 0:
                        ax[j, iplot].spines['top'].set_color('red')
                        ax[j, iplot].spines['bottom'].set_color('red')
                        ax[j, iplot].spines['left'].set_color('red')
                        ax[j, iplot].spines['right'].set_color('red')
                        ax[j, iplot].plot([0, npsf], [0, npsf], 'k')
                        ax[j, iplot].plot([npsf, 0], [0, npsf], 'k')
                        ax[j, iplot].set_xlim(0, npsf)
                        ax[j, iplot].set_ylim(0, npsf)

        ax[0, 0].imshow(psfref, interpolation = 'nearest')

        plt.savefig("%s/TESTING/fforster/LIMITS/optphotVSsextractor_%s_%s_%02i-%02i_stars.png" % (webdir, field, CCD, filesci, fileref))

    # update psf quantities

    var_psfadd = np.array(var_psfadd)
    psfflux = np.array(psfflux)
    e_psfflux = np.array(e_psfflux)
    psfradius = np.array(psfradius)
    psfSNR = np.array(psfSNR)
    print "-----> psfSNR shape", np.shape(psfSNR)
    psfPCA = np.array(psfPCA)
    print "-----> psfPCA shape", np.shape(psfPCA)

    psfadd = psfadd[maskpsfflux]
    var_psfadd = var_psfadd[maskpsfflux]
    psfflux = psfflux[maskpsfflux]
    e_psfflux = e_psfflux[maskpsfflux]
    psfradius = psfradius[maskpsfflux]
    psfSNR = psfSNR[maskpsfflux]
    optflux = optflux[maskpsfflux]
    var_optflux = var_optflux[maskpsfflux]
    print "-----> psfSNR shape after optimal photometry vs sextractor flux mask", np.shape(psfSNR)
    psfPCA = psfPCA[maskpsfflux]
    print "-----> psfPCA shape after optimal photometry vs sextractor flux mask", np.shape(psfPCA)

    psfPCA = psfPCA.tolist()

    # print the local time

    printtime("convolution")


    # Convolution kernel training
    # ------------------------------------------

    if verbose:
        print "\nComputing convolution kernels..."
    nt = np.empty_like(n1)
    varnt = np.empty_like(n1)
    varnt2 = np.empty_like(n1)
    diff = np.empty_like(n1)
    invVAR = np.empty_like(n1)
    
    lasttime = time.time()

    # find kernels serially
    solfilter = np.zeros((nparty, npartx, nf, nf))
    kratio = np.zeros((nparty, npartx))
    ksupport = np.zeros((nparty, npartx))
    knorm2 = np.zeros((nparty, npartx))
    knormsum2 = np.zeros((nparty, npartx))
    # this can be done in parallel
    for ipart in range(nparty):
        for jpart in range(npartx):
            
            i1 = int(nfh + dxconv * ipart)
            i2 = int(i1 + dxconv)
            j1 = int(nfh + dyconv * jpart)
            j2 = int(j1 + dyconv)
            ipixref = (i1 + i2) / 2
            jpixref = (j1 + j2) / 2

            # find kernel
            # TODO: how to define flux limit cuts?
            (nclose, solfilter[ipart, jpart], kratio[ipart, jpart], ksupport[ipart, jpart], knorm2[ipart, jpart], knormsum2[ipart, jpart]) \
                = dokernel(jpixref, ipixref, 333, 2e2, 1e4, x1sel, y1sel, r1sel, conv1st) # 5e5 5e5

    # Add stars artificially to new locations (only used for random forest training)
    # ---------------------------------------

    if doadd:
        if verbose:
            print "\nAdding fake stars... (conv1st: %s)" % conv1st

        # find positions of galay candidates to use more realistic positions
        (igal, jgal) = findgalaxies(x1sex, y1sex, z1sex, r1sex, f1sex, 2 * npsf)
        igal = igal + np.random.normal(0, 3., np.size(igal))
        jgal = jgal + np.random.normal(0, 3., np.size(igal))

        # mask grid positions too close to galaxies
        maskadd = np.ones(len(ijadds), dtype = bool)
        for imask in range(len(ijadds)):
            if np.min((ijadds[imask, 1] - igal)**2 + (ijadds[imask, 0] - jgal)**2) < 2 * npsf:
                maskadd[imask] = False
        ijadds = ijadds[maskadd]

        # add galaxy positions
        ijadds = np.vstack((ijadds, np.vstack((igal, jgal)).transpose()))
        addtype = np.zeros(np.shape(ijadds)[0], dtype = int) # regular grid
        addtype[-len(igal):len(addtype)] = 1 # near galaxy position

        # inverse cumulative function of a dn/dx propto x^(-alpha) between xmin and xmax
        def x_CDF(CDF, xmin, xmax, alpha):
            return (CDF * (xmax**(1.-alpha) - xmin**(1.-alpha)) + xmin**(1.-alpha))**(1. / (1. - alpha))
                
        # add random stars
        randomN = np.random.rand(len(ijadds)) * len(psfadd)
        for irandom in range(len(ijadds)):

            # corner coordinates of stamp
            iadd = int(ijadds[irandom, 1] - npsfh)
            jadd = int(ijadds[irandom, 0] - npsfh)

            # background of images to subtract
            bg1 = background1[iadd: iadd + npsf, jadd: jadd + npsf]
            bg2 = background2[iadd: iadd + npsf, jadd: jadd + npsf]

            # add random star
            probrandom = 0 #0.2
            if np.random.random() < probrandom:
                idx = int(randomN[irandom])
                
                # add scaled down star if too dim
                SNRadd = psfSNR[idx]
                alphascale = 1.
                psfnew = psfadd[idx]
                psfnewflux = optflux[idx] * alphascale
                psfnewradius = psfradius[idx]
                
                sigmascale = np.sqrt((0.1 + np.abs(psfnew / gain)) * (1. - alphascale))
                if alphascale < 0.9:
                    psfnew += np.random.normal(0, sigmascale, np.shape(psfnew))
                    addtype[irandom] += 2 # scaled down star from empirical SNR distribution

            else:  # force star SNR distribution

                # select a random star until we find a relatively bright one
                idx = int(randomN[irandom]) # random number
                while psfSNR[idx] < np.percentile(psfSNR, 30):  # if star too bright try again
                    idx = int(np.random.random() * len(psfadd))

                # select star and scale it to follow desired SNR distribution
                psfnew = psfadd[idx] # extract the input star psf
                SNRadd = x_CDF(np.random.random(), 2.5, 100., 3.) # new flux SNR to force with SNR distribution between 3 and 100 and power law of -3
                alphascale = (SNRadd / psfSNR[idx]) #(optflux[idx] / np.sqrt(var_optflux[idx])) #psfSNR[idx] #  in the low SNR regime the SNR is proportional to the signal
                psfnew = psfnew * alphascale # scale the psf
                psfnewflux = optflux[idx] * alphascale # the scaled flux of the new psf
                psfnewradius = psfradius[idx] # the radius of the input star
                addtype[irandom] += 4 # scaled down star with forced SNR distribution

            # this gives positive and negative stars, which is more general
            try:
                if conv1st or not addstars2ref:
                    n2[iadd: iadd + npsf, jadd: jadd + npsf] = n2[iadd: iadd + npsf, jadd: jadd + npsf] + psfnew
                    varn2[iadd: iadd + npsf, jadd: jadd + npsf] = varn2[iadd: iadd + npsf, jadd: jadd + npsf] + np.abs(psfnew / gain)
                    
                    psfnewflux, var_psfnewflux = getoptphot(psf, n2[iadd: iadd + npsf, jadd: jadd + npsf], varn2[iadd: iadd + npsf, jadd: jadd + npsf], maskpsf)
                    e_psfnewflux = np.sqrt(var_psfnewflux)

                    print "----> SNRaddcomp", SNRadd, optflux[idx] / np.sqrt(var_optflux[idx]), psfnewflux / e_psfnewflux, alphascale, optflux[idx], np.sqrt(var_optflux[idx]), psfnewflux, np.sqrt(var_psfnewflux)

                    if verbose:
                        print "Adding star at location %i, %i (flux: %i, sum(psf): %i, e_flux: %i, SNRadd: %f)" % (iadd + npsfh, jadd + npsfh, psfnewflux, np.sum(psfnew), e_psfnewflux, SNRadd)

                    # log into fakestars file
                    statsfile.write("ADD pix %i %i flux %f %f radius %f SNRadd %f SNRorig %f addtype %i\n" % (iadd + npsfh, jadd + npsfh, psfnewflux, e_psfnewflux, psfnewradius, psfnewflux / e_psfnewflux, optflux[idx] / np.sqrt(var_optflux[idx]), addtype[irandom]))

                else:
                    if addstars2ref:
                        n1[iadd: iadd + npsf, jadd: jadd + npsf] = n1[iadd: iadd + npsf, jadd: jadd + npsf] + psfnew
                        varn1[iadd: iadd + npsf, jadd: jadd + npsf] = varn1[iadd: iadd + npsf, jadd: jadd + npsf] + np.abs(psfnew / gain)
            except:
                print "\n\nWARNING: Cannot add star at position %i, %i\n\n" % (iadd, jadd)
                #print psfnew


    # Apply convolution kernels
    # -------------------------
    
    if verbose:
        print "\nDoing convolution..."
    ilimkernel = np.zeros((nparty, 2), dtype = int)
    jlimkernel = np.zeros((npartx, 2), dtype = int)

    for ipart in range(nparty):
        for jpart in range(npartx):

            if verbose:
                print "    Convolution in partition %i, %i" % (ipart, jpart)

            i1 = int(nfh + dxconv * ipart)
            i2 = int(i1 + dxconv)
            j1 = int(nfh + dyconv * jpart)
            j2 = int(j1 + dyconv)

            ilimkernel[ipart] = np.array([i1, i2])
            jlimkernel[jpart] = np.array([j1, j2])

            #print i1, i2, j1, j2
            #print i1 - nfh, i2 + nfh, j1 - nfh, j2 + nfh
            #print np.shape(n1), np.shape(n2)
            
            # apply kernel
            psftest = solfilter[ipart, jpart]
            print "kernel squared sum:", np.sum(psftest**2)
#            if conv1st:
#                convolution.conv(int(dxconv), int(dyconv), nf, solfilter[ipart, jpart], n1[i1 - nfh: i2 + nfh, j1 - nfh: j2 + nfh], varn1[i1 - nfh: i2 + nfh, j1 - nfh: j2 + nfh])
#            else:
#                convolution.conv(int(dxconv), int(dyconv), nf, solfilter[ipart, jpart], n2[i1 - nfh: i2 + nfh, j1 - nfh: j2 + nfh], varn2[i1 - nfh: i2 + nfh, j1 - nfh: j2 + nfh])
            if conv1st:
                convolution.conv(int(dxconv), int(dyconv), nf, psftest, n1[i1 - nfh: i2 + nfh, j1 - nfh: j2 + nfh], varn1[i1 - nfh: i2 + nfh, j1 - nfh: j2 + nfh])
            else:
                convolution.conv(int(dxconv), int(dyconv), nf, psftest, n2[i1 - nfh: i2 + nfh, j1 - nfh: j2 + nfh], varn2[i1 - nfh: i2 + nfh, j1 - nfh: j2 + nfh])

            # recover solution
            #print i1, i2, j1, j2
            #print int(dxconv), int(dyconv)
            nt[i1: i2, j1: j2] = convolution.iout[0: int(dxconv), 0: int(dyconv)]
            varnt[i1: i2, j1: j2] = convolution.varout[0: int(dxconv), 0: int(dyconv)]
            #varnt2[i1: i2, j1: j2] = convolution.varout2[0: int(dxconv), 0: int(dyconv)]
    
            # start computing diff and variance
            if conv1st:
                diff[i1: i2, j1: j2] = n2[i1: i2, j1: j2] - nt[i1: i2, j1: j2] # difference in ADU
                #invVAR[i1: i2, j1: j2] = varn2[i1: i2, j1: j2] + varn1[i1: i2, j1: j2] * knorm2[ipart, jpart] \
                #                         + knorm2[ipart, jpart] * (varn1[i1: i2, j1: j2] - varnt[i1: i2, j1: j2])
                #invVAR[i1: i2, j1: j2] = varn2[i1: i2, j1: j2] + varn1[i1: i2, j1: j2] * knorm2[ipart, jpart] \
                #                         + varn1[i1: i2, j1: j2] * knorm2[ipart, jpart] - varn1[i1: i2, j1: j2] * knormsum2[ipart, jpart]
                print "---------------------------->", knorm2[ipart, jpart], knormsum2[ipart, jpart]
                invVAR[i1: i2, j1: j2] = varn2[i1: i2, j1: j2] + varn1[i1: i2, j1: j2] * knorm2[ipart, jpart] 
#                                         + varnt[i1: i2, j1: j2] / knormsum2[ipart, jpart] 
#                                         - varn1[i1: i2, j1: j2] * knormsum2[ipart, jpart]
            else:
                diff[i1: i2, j1: j2] = nt[i1: i2, j1: j2] - n1[i1: i2, j1: j2] # difference in ADU
                invVAR[i1: i2, j1: j2] = varn1[i1: i2, j1: j2] + varn2[i1: i2, j1: j2] * knorm2[ipart, jpart] \
#                                         + varnt[i1: i2, j1: j2] / knormsum2[ipart, jpart] \
#                                         - varn2[i1: i2, j1: j2] * knormsum2[ipart, jpart]
# + varn1[i1: i2, j1: j2] # variance in ADU


    # Save remaining star snapshots for PCA (including convolution of original stars)
    # -------------------------------------

    for istar in range(len(psfPCA)):
        ipix = np.round(psfPCA[istar][0])
        jpix = np.round(psfPCA[istar][1])
        bg1 = background1[int(jpix - npsfh - 1): int(jpix + npsfh), int(ipix - npsfh - 1): int(ipix + npsfh)]
        bg2 = background2[int(jpix - npsfh - 1): int(jpix + npsfh), int(ipix - npsfh - 1): int(ipix + npsfh)]
        signal = nt[int(jpix - npsfh - 1): int(jpix + npsfh), int(ipix - npsfh - 1): int(ipix + npsfh)]
        if np.min(signal + bg1 + bg2) > 0 and np.max(signal) < 30000:
            #psfref += signal
            signal = signal / np.sqrt((signal + bg1 + bg2) / gain) # SNR
            if np.isfinite(np.max(signal)) and np.min(signal) > -3:
                psfPCA.append(np.concatenate([np.array([ipix, jpix]), signal.flatten()]))

    psfPCA = np.array(psfPCA)
    if conv1st:
        fileout \
            = "%s/%s/%s/CALIBRATIONS/PCA_stars_%s_%s_%02i-%02it_grid%02i_%s.npy" % (sharedir, field, CCD, field, CCD, filesci, fileref, fileref, resampling)
    else:
        fileout \
            = "%s/%s/%s/CALIBRATIONS/PCA_stars_%s_%s_%02it-%02i_grid%02i_%s.npy" % (sharedir, field, CCD, field, CCD, filesci, fileref, fileref, resampling)

    if verbose:
        print "Saving reference PCA..."
        print "    PCA psf's shape:", np.shape(psfPCA)
    np.save(fileout, psfPCA)
    

    # Save reference psf for flux computation
    # ---------------------------------------
    
    psfref = psfref / np.sum(psfref)
    if conv1st:
        fileout \
            = "%s/%s/%s/CALIBRATIONS/psf_%s_%s_%02i-%02it_grid%02i_%s.npy" % (sharedir, field, CCD, field, CCD, filesci, fileref, fileref, resampling)
    else:
        fileout \
            = "%s/%s/%s/CALIBRATIONS/psf_%s_%s_%02it-%02i_grid%02i_%s.npy" % (sharedir, field, CCD, field, CCD, filesci, fileref, fileref, resampling)
    if verbose:
        print "Saving PSF..."
    np.save(fileout, psfref)
    fig, ax = plt.subplots()
    impsf = ax.imshow(psfref, interpolation = 'nearest')
    cbar = fig.colorbar(impsf)
    plt.savefig((fileout.replace(sharedir, "%s" % webdir)).replace(".npy", ".png"), bbox_inches = 'tight')
    

    # Difference and inverse variance image
    # --------------------------------------------------------

    printtime("difference")

    # save difference image
    if conv1st:
        fileout \
            = "%s/%s/%s/Diff_%s_%s_%02i-%02it_grid%02i_%s.fits" % (outdir, field, CCD, field, CCD, filesci, fileref, fileref, resampling)
    else:
        fileout \
            = "%s/%s/%s/Diff_%s_%s_%02it-%02i_grid%02i_%s.fits" % (outdir, field, CCD, field, CCD, filesci, fileref, fileref, resampling)
    if verbose:
        print "Saving differences..."
    fits1[0].data = diff
    fits1.writeto(fileout, clobber = True)

    # mask bad pixels or subtractions
    if verbose:
        print "Final image shapes:"
        print "   diff, invVAR, varn1, varn2, varnt:", np.shape(diff), np.shape(invVAR), np.shape(varn1), np.shape(varn2), np.shape(varnt)
    mask = (invVAR < varmax) & np.isfinite(invVAR) & np.isfinite(diff) & np.isfinite(varn1) & np.isfinite(varn2) & np.isfinite(varnt) \
        & (n1 > -5. * np.sqrt(background1 / gain)) & (n2 > -5. * np.sqrt(background2 / gain)) \
        & (varn1 < varmax) & (varn2 < varmax) & (varnt < varmax)

    if verbose:
        print "   inverse variance mask shape:", np.shape(mask)
    
    if saveconv:
        if conv1st:
            fileout \
                = "%s/%s/%s/diffvar_%s_%s_%02i-%02it_grid%02i_%s.fits" % (outdir, field, CCD, field, CCD, filesci, fileref, fileref, resampling)
        else:
            fileout \
                = "%s/%s/%s/diffvar_%s_%s_%02it-%02i_grid%02i_%s.fits" % (outdir, field, CCD, field, CCD, filesci, fileref, fileref, resampling)
        if verbose:
            print "Saving difference variance image"
        fits1[0].data = invVAR
        fits1.writeto(fileout, clobber = True)    

    # save 1/variance image before computing SNR
    invVAR[mask] = 1. / invVAR[mask]
    invVAR[np.invert(mask)] = 0
    invVAR[0:dn, :] = 0
    invVAR[ny - dn: ny, :] = 0
    invVAR[:, 0:dn] = 0
    invVAR[:, nx - dn: nx] = 0

    if conv1st:
        fileout \
            = "%s/%s/%s/invVAR_%s_%s_%02i-%02it_grid%02i_%s.fits" % (outdir, field, CCD, field, CCD, filesci, fileref, fileref, resampling)
    else:
        fileout \
            = "%s/%s/%s/invVAR_%s_%s_%02it-%02i_grid%02i_%s.fits" % (outdir, field, CCD, field, CCD, filesci, fileref, fileref, resampling)
    if verbose:
        print "Saving 1/variance image..."
    fits1[0].data = invVAR
    fits1.writeto(fileout, clobber = True)    

    if saveSNR:
        fits1[0].data = diff * invVAR
        fits1.writeto(fileout.replace("invVAR", "diffSNR"), clobber = True)    
        if verbose:
            print "----------> Saving SNR image"
    else:
        if verbose:
            print "Skipping SNR image saving..."

    if saveconv:
        if verbose:
            print "----------> Saving convolved image..."
        fits1[0].data = nt
        fits1.writeto(fileout.replace("invVAR", "conv"), clobber = True)    

        if verbose:
            print "----------> Saving convolved variance image..."
        fits1[0].data = varnt
        fits1.writeto(fileout.replace("invVAR", "convvar"), clobber = True)    
    else:
        if verbose:
            print "Skipping convolved image and image variance saving..."
        


    # Do optimal photometry for the entire image and save
    # ----------------------------------------------------
    
    printtime("optphot")
    
    if verbose:
        print "Some difference image statistics:"
        print "   sum of non finite:", np.sum(np.invert(np.isfinite(diff)))
        print "   diff dtype, shape, min, max:", diff.dtype, np.shape(diff), np.min(diff), np.max(diff)
        print "   invVAR dtype, shape, min, max:", invVAR.dtype, np.shape(invVAR), np.min(invVAR), np.max(invVAR)

    if verbose:
        print "\nEntering fortran optimal photometry routine..."
    optimalphotometry.dooptphot(ny, nx, npsf, fcut, psfref, diff, invVAR)
    if verbose:
        print "Leaving fortran, optimal photometry ready\n"

    flux = np.empty_like(invVAR)
    fluxSNR = np.empty_like(invVAR)
    
    flux = optimalphotometry.flux[0:ny, 0:nx]
    fluxSNR = optimalphotometry.varflux[0:ny, 0:nx]
    fluxSNR = flux / np.sqrt(fluxSNR)
    fluxSNR[np.invert(np.isfinite(fluxSNR))] = 0

    if saveoptphot:
        if conv1st:
            fileout \
                    = "%s/%s/%s/flux_%s_%s_%02i-%02it_grid%02i_%s.fits" % (outdir, field, CCD, field, CCD, filesci, fileref, fileref, resampling)
        else:
            fileout \
                    = "%s/%s/%s/flux_%s_%s_%02it-%02i_grid%02i_%s.fits" % (outdir, field, CCD, field, CCD, filesci, fileref, fileref, resampling)
        fits1[0].data = flux
        fits1.writeto(fileout, clobber = True)
        fits1[0].data = fluxSNR
        fits1.writeto(fileout.replace("flux", "fluxSNR"), clobber = True)
        
        print "------> Saving flux and fluxSNR images"
    else:
        print "Skipping flux and fluxSNR image saving..."
        

    if docheckSNRhisto:
        fig, ax = plt.subplots(ncols = 2, figsize = (12, 6))
        region = fluxSNR[600:3400, 600:1400]
        region = region[region != 0]
        dbin = 0.25
        bins = np.arange(-20, 20, dbin)
        binsc = (bins[:-1] + bins[1:]) / 2.
        (hist, edges) = np.histogram(region, bins = bins, normed = True)
        ax[0].plot(binsc, hist, c = 'b', label = "flux SNR")
        ax[0].plot(binsc, stats.norm.pdf(binsc), c = 'r', label = "G(0, 1)")
        ax[0].set_xlabel("SNR")
        ax[0].set_ylabel("pdf")
        perc16 = np.percentile(region, 16)
        perc84 = np.percentile(region, 84)
        ax[0].axvline(perc16, c = 'b')
        ax[0].axvline(perc84, c = 'b')
        ax[0].axvline(1., c = 'r')
        ax[0].axvline(-1., c = 'r')
        ax[0].set_xlim(-5, 5)
        ax[0].set_ylim(0, 0.45)
        SNRlim = SNRlim * max(np.abs(perc16), np.abs(perc84), 1.)
        if verbose:
            print "\n\nWARNING - new SNRlim after histogram check: %s\n\n" % SNRlim
        ax[0].legend(fancybox = False, prop = {'size':8}, loc = 2)
        ax[1].plot(binsc, hist, c = 'b', label = "flux SNR")
        ax[1].plot(binsc, stats.norm.pdf(binsc), c = 'r', label = "G(0, 1)")
        ax[1].fill_between(binsc, hist, color = 'gray', where = (binsc > 5))
        ax[1].set_xlabel("SNR")
        ax[1].set_ylabel("pdf")
        ax[1].axvline(perc16, c = 'b')
        ax[1].axvline(perc84, c = 'b')
        ax[1].axvline(1., c = 'r')
        ax[1].axvline(-1., c = 'r')
        ax[1].set_yscale('log')
        ax[1].set_ylim(1e-5, 1e-1)
        plt.savefig("%s/%s/%s/CANDIDATES/histo_fluxSNR_%s_%s_%02i-%02i.png" % (webdir, field, CCD, field, CCD, filesci, fileref))

    # Select candidates with simple flux SNR criterion and save them
    # --------------------------------------------------------------

    isel = []
    jsel = []
    SNRsel = []
    
    n1[np.invert(np.isfinite(n1))] = nmax
    n2[np.invert(np.isfinite(n2))] = nmax
    nt[np.invert(np.isfinite(nt))] = nmax

    if conv1st:
        maskSNR = mask & (np.abs(fluxSNR) > SNRlim) & ((n2 < 10000) | (nt < 10000)) & (varn1 < varmax) & (varn2 < varmax)
    else:
        maskSNR = mask & (np.abs(fluxSNR) > SNRlim) & ((nt < 10000) | (n1 < 10000)) & (varn1 < varmax) & (varn2 < varmax)

    nsort = np.sum(maskSNR)
    if verbose:
        print "Number of pixels to sort: %i" % nsort

    if nsort > 1e6:
        print "\n\nWARNING: Too many pixels to sort, probably a bad subtraction."
        sys.exit(28)
        
    printtime("sorting")

    argsort = np.argsort(np.abs(fluxSNR[maskSNR]))[::-1]
    SNRleft = fluxSNR[maskSNR][argsort]
    ileft = xidx[maskSNR][argsort]
    jleft = yidx[maskSNR][argsort]

    if verbose:
        print "Selecting candidates..."

    # loop to select candidates
    while len(ileft) > 0:

        isel.append(ileft[0])    # pixel coordinate
        jsel.append(jleft[0])    # pixel coordinate
        SNRsel.append(SNRleft[0])  # integrated flux SNR

        maskdist = np.sqrt(\
            (ileft - ileft[0])**2 + (jleft - jleft[0])**2) > (npsf / 2.)

        ileft = ileft[maskdist]
        jleft = jleft[maskdist]
        SNRleft = SNRleft[maskdist]


    # Save candidate information for later feature computation and Random Forest classification
    # -----------------------------------------------------------------------------------------

    printtime("savingfeatures")

    cands = []

    for icand in range(len(isel)):
        
        ipix = isel[icand]
        jpix = jsel[icand]
        bg1 = background1[ipix, jpix]
        bg2 = background2[ipix, jpix]
        fluxcand = flux[ipix, jpix]  # integrated flux
        fluxSNRcand = SNRsel[icand] # integrated flux signal to noise ratio
        pixSNRcand = diff[ipix, jpix] * np.sqrt(invVAR[ipix, jpix])  # image snr at peak integated flux snr image
        imSNR = diff[ipix - npsfh: ipix + npsfh + 1, jpix - npsfh: jpix + npsfh + 1] * np.sqrt(invVAR[ipix - npsfh: ipix + npsfh + 1, jpix - npsfh: jpix + npsfh + 1])
        imSNR = imSNR.flatten()  # SNR image
        im1 = n1[ipix - npsfh: ipix + npsfh + 1, jpix - npsfh: jpix + npsfh + 1].flatten()  # image 1
        im1orig = n1orig[ipix - npsfh: ipix + npsfh + 1, jpix - npsfh: jpix + npsfh + 1].flatten()  # image 1

        # early detection of cosmic rays in the reference star
        if useweights == 'internal':
            cosmicdiff = np.sum((im1[rs2D < 5] - im1orig[rs2D < 5])**2)
            if cosmicdiff > 5000:
                print "-----> Cosmic ray in reference image of candidate detected (cosmicdiff = %e), skipping candidate..." % cosmicdiff
                continue

        im2 = n2[ipix - npsfh: ipix + npsfh + 1, jpix - npsfh: jpix + npsfh + 1].flatten()  # image 2
        imt = nt[ipix - npsfh: ipix + npsfh + 1, jpix - npsfh: jpix + npsfh + 1].flatten()  # convolved image (either image 1 or image 2)
        ikernel = 0
        jkernel = 0
        for ipart in range(nparty):
            if isel[icand] >= ilimkernel[ipart, 0] and isel[icand] < ilimkernel[ipart, 1]:
                ikernel = ipart
                break
        for jpart in range(npartx):
            if jsel[icand] >= jlimkernel[jpart, 0] and jsel[icand] < jlimkernel[jpart, 1]:
                jkernel = jpart
                break

        cands.append(np.concatenate([np.array([ipix, jpix, kratio[ikernel, jkernel], ksupport[ikernel, jkernel], fluxcand, fluxSNRcand, pixSNRcand, bg1, bg2, MJDref, date, aflux]), imSNR, im1, im2, imt]))

    cands = np.array(cands)

    if conv1st:
        fileout \
            = "%s/%s/%s/CANDIDATES/cand_%s_%s_%02i-%02it_grid%02i_%s.npy" % (sharedir, field, CCD, field, CCD, filesci, fileref, fileref, resampling)
    else:
        fileout \
            = "%s/%s/%s/CANDIDATES/cand_%s_%s_%02it-%02i_grid%02i_%s.npy" % (sharedir, field, CCD, field, CCD, filesci, fileref, fileref, resampling)

    # save
    if verbose:
        print "\nSaving candidates..."
        print "    Cand. shape:", np.shape(cands)
    np.save(fileout, cands)
    
    del flux, fluxSNR#, im1orig
        

# #########################################
# ########  candidate analysis ############
# #########################################

printtime("docandidates")


# Load random forest classifier
# -----------------------------

if (docandidates and doML) or doadd:
    
    # Load random forest classifier

    from sklearn.ensemble import RandomForestClassifier
    #    rfc = pickle.load(open("%s/DECAM_classifier_13A_01-09_20150212.pkl" % sharedir, 'rb'))
    #    rfc = pickle.load(open("%s/DECAM_classifier_13A_01-09_20150212.pkl" % sharedir, 'rb'))

    # 20160623: commented while we don't have odd even classifier with new scikit learn
    #if field[:8] == 'Blind13A':
    #    if np.mod(int(CCD[1:]), 2) == 0:
    #        rfc = pickle.load(open("%s/DECAM_classifier_13A_01-05_odd.pkl" % sharedir, 'rb')) # use odd classifier for even CCDs
    #    else:
    #        rfc = pickle.load(open("%s/DECAM_classifier_13A_01-05_even.pkl" % sharedir, 'rb')) # use even classifier for odd CCDs
    #else:
    #    rfc = pickle.load(open("%s/DECAM_classifier_15A_01-50_02-09_final.pkl" % sharedir, 'rb'))
    rfc = pickle.load(open("%s/RF_model_without_dl.pkl" % sharedir, 'rb'))
    rfc.set_params(n_jobs = 1)


    if (dofilter or dorevisit) and dorfcpair:
        rfcpair = pickle.load(open("%s/DECAM_classifier_14A_05-09_selected.pkl" % sharedir, 'rb'))
        rfcpair.set_params(n_jobs = 1)
        
    print "Random Forest Classifier loaded"

    # Hybrid Random Forest (with deep learning as feature)
    rfc_hybrid = pickle.load(open("%s/RF_model.pkl" % sharedir, 'rb'))
    rfc_hybrid.set_params(n_jobs = 1)

    # Load deep learning classifier (DeepHiTS)
    sys.path.append("/home/apps/astro/HiTS/devel/DeepDetector")
    from DeepDetector import DeepDetector, normalize_stamp

    # Deep Learning without psf
    DeepHiTSmodel_nopsf = "/home/apps/astro/HiTS/devel/DeepDetector/convnet_7_nopsf.pkl"
    DeepHiTSarch_nopsf = "/home/apps/astro/HiTS/devel/DeepDetector/arch7_cross.py"

    print 'Loading DeepDetector without psf at files', DeepHiTSmodel_nopsf, DeepHiTSarch_nopsf
    deepdetector_nopsf = DeepDetector(DeepHiTSarch_nopsf, DeepHiTSmodel_nopsf, 1, im_chan=4, im_size=21)

    # Deep Learning with psf
    DeepHiTSmodel_withpsf = "/home/apps/astro/HiTS/devel/DeepDetector/convnet_7_5im.pkl"
    DeepHiTSarch_withpsf = "/home/apps/astro/HiTS/devel/DeepDetector/arch7.py"

    print 'Loading DeepDetector with psf at files', DeepHiTSmodel_withpsf, DeepHiTSarch_withpsf
    deepdetector_withpsf = DeepDetector(DeepHiTSarch_withpsf, DeepHiTSmodel_withpsf, 1, im_chan=5, im_size=21)


if dorevisit and dorfcpair:
    from sklearn.ensemble import RandomForestClassifier
    rfcpair = pickle.load(open("%s/DECAM_classifier_14A_05-09_selected.pkl" % sharedir, 'rb'))



# Function to do candidate analysis on given candidate file
# -----------------------------------------------------
def docandidateanalysis(idxfilesci):

    # remove any candidate image generated
    candmatch = "cand_%s_%s_%02i*-%02i*_grid%02i_%s.npy_*.png" % (field, CCD, idxfilesci, fileref, fileref, resampling)
    command = "rm -f %s/%s/%s/CANDIDATES/%s" % (sharedir, field, CCD, candmatch)
    if verbose:
        print "    %s\n" % command
    os.system(command)


    # PCA analysis on SNR images
    # --------------------------------------------------
    
    starfiles = os.listdir("%s/%s/%s/CALIBRATIONS/" % (sharedir, field, CCD))

    # add stars from current difference
    psfPCA = None
    starmatch = "PCA_stars_%s_%s_%02i(.*?)-%02i(.*?)_grid%02i_%s.npy" % (field, CCD, idxfilesci, fileref, fileref, resampling)
    for starfile in starfiles:
        if re.search(starmatch, starfile):
            psfPCA = np.load("%s/%s/%s/CALIBRATIONS/%s" % (sharedir, field, CCD, starfile))

    # add more stars until enough of them are found for PCA
    starmatch = "PCA_stars_%s_%s_(.*?)-%02i(.*?)_grid%02i_%s.npy" % (field, CCD, fileref, fileref, resampling)
    for starfile in starfiles:
        if re.search(starmatch, starfile):
            print starfile, np.shape(psfPCA), np.shape("%s/%s/%s/CALIBRATIONS/%s" % (sharedir, field, CCD, starfile))
            psfPCA = np.vstack([psfPCA, np.load("%s/%s/%s/CALIBRATIONS/%s" % (sharedir, field, CCD, starfile))])
            if np.shape(psfPCA)[0] >= npsf2:
                break
            
    if verbose:
        print "    psfPCA shape:", np.shape(psfPCA)

    if np.shape(psfPCA)[0] < npsf2:
        print "\n\n Not enough reference stars (%i) to do PCA, using SVD decomposition instead for first components\n\n" % np.shape(psfPCA)[0]
        #sys.exit(-1)
                           
    # do PCA using sklearn

    print "Using sklearn PCA"
    from sklearn.decomposition import PCA

    try:
        pca = PCA(n_components = nk)#min(np.shape(psfPCA)[0], npsf2))
        pca.fit(psfPCA[:, 2: npsf2 + 2])
        eigenval = pca.explained_variance_ratio_
        eigenvec = pca.components_
    except:
        print "\n\n Failed to do SVD decomposition\n\n"
        sys.exit(40)

    # do NMF decomposition
    doNMF = False
    if doNMF:
        from sklearn.decomposition import ProjectedGradientNMF
        
        nmf = ProjectedGradientNMF(n_components = nk, sparseness='components', init='random', random_state=0)
        nmf.fit(np.abs(psfPCA[:, 2: npsf2 + 2]))
        dictionaries = nmf.components_
        

    # plot PCA summary
    # ----------------

    if doplotPCA:
        print "plotting PCA eigenvectors"
        # plot eigenvalues
        plt.clf()
        fig, ax = plt.subplots()
        ax.plot(np.cumsum(eigenval))# / np.sum(eigenval))
        #ax.plot(eigenval)
        ax.plot(np.array([nk, nk]), np.array([0, 1]), 'k')
        ax.set_ylim(0, 1)
        ax.set_xlabel("Number of components")
        ax.set_ylabel("Fraction of the total variance")
        fileout = "%s/%s/%s/CALIBRATIONS/PCA_eigenvals_%s_%s_%02it-%02i_grid%02i_%s.png" \
            % (webdir, field, CCD, field, CCD, idxfilesci, fileref, fileref, resampling)
        print "Saving file: %s" % fileout
        plt.savefig(fileout)
                
        # plot images of eigenvectors
        nyeigen = 1
        plt.clf()
        fig, ax = plt.subplots(ncols = nk, figsize = (19, 11))
        for jPCA in range(nk):
            idx = jPCA
            try:
                ax[jPCA].imshow(eigenvec[idx].reshape(npsf, npsf), interpolation = 'nearest')
            except:
                print "-----------> Cannot plot eigenvector"
                continue
        fileout = "%s/%s/%s/CALIBRATIONS/PCA_eigenvec_%s_%s_%02it-%02i_grid%02i_%s.png" \
            % (webdir, field, CCD, field, CCD, idxfilesci, fileref, fileref, resampling)
        plt.savefig(fileout)

        if doNMF:
            # plot images of NMF dictionaries
            plt.clf()
            fig, ax = plt.subplots(ncols = nk, figsize = (19, 11))
            for iNMF in range(nk):
                ax[iNMF].imshow(dictionaries[iNMF].reshape(npsf, npsf), interpolation = 'nearest')
            fileout = "%s/%s/%s/CALIBRATIONS/NMF_dictionaries_%s_%s_%02it-%02i_grid%02i_%s.png" \
                      % (webdir, field, CCD, field, CCD, idxfilesci, fileref, fileref, resampling)
            plt.savefig(fileout)


    # Reference psf
    # -------------------------------------

    psfmatch = "psf_%s_%s_%02i(.*?)-%02i(.*?)_grid%02i_%s.npy" % (field, CCD, idxfilesci, fileref, fileref, resampling)
    for starfile in starfiles:
        if re.search(psfmatch, starfile):
            filepsf = starfile
            psfref = np.load("%s/%s/%s/CALIBRATIONS/%s" % (sharedir, field, CCD, filepsf))
    psfrefnorm = np.sqrt(np.sum(psfref**2))
    psfrefflat = psfref.flatten()
    if verbose:
        print "    psfref shape:", np.shape(psfrefflat)
        
    # find radius that encloses 90% of the psf flux
    for rlim in np.arange(1., npsf / 2, 0.1):
        fluxpsf = np.sum(psfrefflat[rs2D <= rlim])
        if fluxpsf > 0.80:  # 0.95
            break
    if verbose:
        print "    Photometry radius: %4.1f pixels" % (rlim)


    # Load Transient candidates
    # ----------------------------------------------------------

    candfiles = os.listdir("%s/%s/%s/CANDIDATES/" % (sharedir, field, CCD))

    candmatch = "cand_%s_%s_%02i(.*?)-%02i(.*?)_grid%02i_%s.npy" % (field, CCD, idxfilesci, fileref, fileref, resampling)
    for candfile in candfiles:
        if re.search(candmatch, candfile):
            print candfile
            cands = np.load("%s/%s/%s/CANDIDATES/%s" % (sharedir, field, CCD, candfile))
            break
    if verbose:
        print "    cand shape:", np.shape(cands)
    conv1st = (re.findall(candmatch, candfile)[0][0] != 't')
    
    if verbose:
        print "    conv1st: %s" % conv1st

    # load scalars from candidates file (ninfo must be last number + 1!)
    # ---------------------------------
    
    ipixcand = cands[:, 0]
    jpixcand = cands[:, 1]
    kratiocand = cands[:, 2]
    ksupportcand = cands[:, 3]
    fluxcand = cands[:, 4]
    fluxSNRcand = cands[:, 5]
    pixSNRcand = cands[:, 6]
    bg1 = cands[:, 7]
    bg2 = cands[:, 8]
    MJDrefcand = cands[:, 9]
    deltaMJDcand = cands[:, 10]
    afluxcand = cands[:, 11]
    
    # loop among candidates to compute features
    # -----------------------------------------

    #if verbose:
        # Random Forest attributes:
        #attr = ["fluxSNR", "pixSNR", "PCA0", "diffcoeff", "crosscorr", "R2", "SW", "std", "symmidx", "ncand", "minimmax", "CR1", "CR2"]
        #print "    Computing candidate features to be fed to random forest classifier (%s)" % attr

    # THIS COULD BE DONE IN PARALLEL
    printtime("features")

    lasttime = time.time()


    #SERIAL-START
    fake = np.zeros(len(cands), dtype = bool)
    faketype = -1 * np.ones(len(cands), dtype = bool)
    selected = np.ones(len(cands), dtype = bool)
    probs = np.zeros(len(cands))
    save = np.ones(len(cands), dtype = bool)
    coeffcand = np.zeros((nk, len(cands)))
    coeffabscand = np.zeros((nk, len(cands)))
    diffcoeff = np.zeros(len(cands))
    symmidx = np.zeros(len(cands))
    ncand = np.zeros(len(cands))
    kratio = np.zeros(len(cands))
    ksupport = np.zeros(len(cands))
    flux = np.zeros(len(cands))
    e_flux = np.zeros(len(cands))
    crosscorr = np.zeros(len(cands))
    crosscorr3 = np.zeros(len(cands))
    crosscorr5 = np.zeros(len(cands))
    crosscorr8 = np.zeros(len(cands))
    dCCPCA0 = np.zeros(len(cands))
    dhu0_2 = np.zeros(len(cands))
    dhu1_2 = np.zeros(len(cands))
    dhu2_2 = np.zeros(len(cands))
    dhu3_2 = np.zeros(len(cands))
    dhu4_2 = np.zeros(len(cands))
    dhu5_2 = np.zeros(len(cands))
    dhu6_2 = np.zeros(len(cands))
    dhu7_2 = np.zeros(len(cands))
    dhu0_4 = np.zeros(len(cands))
    dhu1_4 = np.zeros(len(cands))
    dhu2_4 = np.zeros(len(cands))
    dhu3_4 = np.zeros(len(cands))
    dhu4_4 = np.zeros(len(cands))
    dhu5_4 = np.zeros(len(cands))
    dhu6_4 = np.zeros(len(cands))
    dhu7_4 = np.zeros(len(cands))
    dhu0_4gt = np.zeros(len(cands))
    dhu1_4gt = np.zeros(len(cands))
    dhu2_4gt = np.zeros(len(cands))
    dhu3_4gt = np.zeros(len(cands))
    dhu4_4gt = np.zeros(len(cands))
    dhu5_4gt = np.zeros(len(cands))
    dhu6_4gt = np.zeros(len(cands))
    dhu7_4gt = np.zeros(len(cands))
    bump = np.zeros(len(cands))
    entropy = np.zeros(len(cands))
    offset = np.zeros(len(cands))
    nmax1 = np.zeros(len(cands))
    nmax2 = np.zeros(len(cands))
    ratiomax1 = np.zeros(len(cands))
    ratiomax2 = np.zeros(len(cands))
    norm = np.zeros(len(cands))
    R2 = np.zeros(len(cands))
    SW = np.zeros(len(cands))
    std = np.zeros(len(cands))
    im1min = np.zeros(len(cands))
    im1max = np.zeros(len(cands))
    im2min = np.zeros(len(cands))
    im2max = np.zeros(len(cands))
    CR1 = np.zeros(len(cands))
    CR2 = np.zeros(len(cands))

    for icand in range(len(cands)):

        # check that candidate is inside definition range of transformed image
        # --------------------------------------------------------------------

        ## very close to a bad pixel
        if np.max(varn1[np.max([0, int(ipixcand[icand]) - 3]): np.min([int(ipixcand[icand]) + 3, ny]), np.max([0, int(jpixcand[icand]) - 3]): np.min([int(jpixcand[icand]) + 3, nx])]) >= varmax:
            selected[icand] = False
            save[icand] = False
            continue
            
        delta = 30
        if ipixcand[icand] <= delta or jpixcand[icand] <= delta or ipixcand[icand] >= ny - 1 - delta or jpixcand[icand] >= nx - 1 - delta:
            selected[icand] = False
            save[icand] = False
            continue

        (ipixt, jpixt) = applytransformation(order, ipixcand[icand], jpixcand[icand], sol_astrometry)
        if ipixt <= delta or jpixt <= delta or ipixt >= ny - 1 - delta or jpixt >= nx - 1 - delta:
            selected[icand] = False
            save[icand] = False
            continue
        
        
        # count the number of candidates within a radius of 10 npsf
        # ---------------------------------------------------------

        distcand = np.sqrt((ipixcand - ipixcand[icand])**2 + (jpixcand - jpixcand[icand])**2)
        maskcand = distcand < 5. * npsf
        ncand[icand] = np.sum(maskcand)
        avgcands = np.average(fluxSNRcand[maskcand])
        stdcands = np.std(fluxSNRcand[maskcand])
        if ncand[icand] >= ncandlim: # probably near a very bright star
            selected[icand] = False
            save[icand] = False
            continue
            """
            if np.abs((fluxSNRcand[icand] - avgcands) / stdcands) > 10:
            print "     ", fluxSNRcand[icand], (fluxSNRcand[icand] - avgcands) / stdcands
            else:
                continue
            """

        # if not in a very dense candidate region continue computing features
        # -------------------------------------------------------------------
        
        imSNR = cands[icand, ninfo: ninfo + npsf2]
        im1 = cands[icand, ninfo + npsf2: ninfo + 2 * npsf2]
        im2 = cands[icand, ninfo + 2 * npsf2: ninfo + 3 * npsf2]
        imt = cands[icand, ninfo + 3 * npsf2: ninfo + 4 * npsf2]
        if conv1st:
            diff = im2 - imt
        else:
            diff = imt - im1


        # candidate flux and error
        # ------------------------
        
        flux[icand] = fluxcand[icand]
        e_flux[icand] = fluxcand[icand] / fluxSNRcand[icand]
        # bring photometry to the scale of the reference image
        if conv1st: 
           flux[icand] = flux[icand] / afluxcand[icand]
           e_flux[icand] = e_flux[icand] / afluxcand[icand]


        # candidate cross-correlation with psf
        # ------------------------------------

        maskr2 = np.array(rs2D < 2)
        maskr3 = np.array(rs2D < 3)
        maskr4 = np.array(rs2D < 4)
        maskr5 = np.array(rs2D < 5)
        maskr8 = np.array(rs2D < 8)
        maskr4gt = np.array(rs2D >= 4)

        crosscorr[icand] = np.abs(np.dot(diff, psfrefflat) / np.sqrt(np.sum(diff**2)) / psfrefnorm)
        crosscorr3[icand] = np.abs(np.dot(diff[maskr3], psfrefflat[maskr3]) \
                                   / np.sqrt(np.sum(diff[maskr3]**2)) / np.sqrt(np.sum(psfrefflat[maskr3]**2)))
        crosscorr5[icand] = np.abs(np.dot(diff[maskr5], psfrefflat[maskr5]) \
                                   / np.sqrt(np.sum(diff[maskr5]**2)) / np.sqrt(np.sum(psfrefflat[maskr5]**2)))
        crosscorr8[icand] = np.abs(np.dot(diff[maskr8], psfrefflat[maskr8]) \
                                   / np.sqrt(np.sum(diff[maskr8]**2)) / np.sqrt(np.sum(psfrefflat[maskr8]**2)))


        # candidate norm
        # --------------
        
        norm[icand] = np.sqrt(np.sum(imSNR**2))


        # normalized PCA coefficients
        # ---------------------------
        
        modelPCA = np.zeros(npsf2)
        for iPCA in range(nk):
            coeffcand[iPCA, icand] = np.dot(imSNR, eigenvec[iPCA])
            coeffabscand[iPCA, icand] = np.dot(np.abs(imSNR), eigenvec[iPCA])
            modelPCA = modelPCA + eigenvec[iPCA] * coeffcand[iPCA, icand]
        diffPCA = imSNR - modelPCA

        diffcoeff[icand] = np.abs(np.abs(coeffabscand[0, icand]) - np.abs(coeffcand[0, icand])) / norm[icand]
        coeffcand[:, icand] = np.abs(coeffcand[:, icand]) / norm[icand]
        coeffabscand[:, icand] = np.abs(coeffabscand[:, icand]) / norm[icand]


        # difference between PCA0 and cross correlation
        # ---------------------------------------------
        dCCPCA0[icand] = np.abs(crosscorr[icand]) - np.abs(coeffcand[0, icand])
        

        # Hu moments
        # ---------------------------------------------
        if conv1st:
            #diff = im2 - imt
            im1hu = im2
        else:
            #diff = imt - im1
            im1hu = im1
        im2hu = imt

        """
        im1hu = im1
        im2hu = im2
        """

        im1mask = np.array((np.abs(im1hu) < 65000) & (np.isfinite(im1hu)))
        im2mask = np.array((np.abs(im2hu) < 65000) & (np.isfinite(im2hu)))

        mask2 = im1mask & im2mask & maskr2
        mask4 = im1mask & im2mask & maskr4
        mask4gt = im1mask & im2mask & maskr4gt
        if np.sum(mask2) > 0:
            min2 = np.minimum(np.min(im1hu[mask2]), np.min(im2hu[mask2]))
            max2 = np.maximum(np.max(im1hu[mask2]), np.max(im2hu[mask2]))
            if min2 != max2:
                im1hu2 = (im1hu - min2) / (max2 - min2)
                im2hu2 = (im2hu - min2) / (max2 - min2)
                (dhu0_2[icand], dhu1_2[icand], dhu2_2[icand], dhu3_2[icand], dhu4_2[icand], dhu5_2[icand], \
                 dhu6_2[icand], dhu7_2[icand]) = np.abs(humoments(im1hu2, mask2) - humoments(im2hu2, mask2))
        if np.sum(mask4) > 0:
            min4 = np.minimum(np.min(im1hu[mask4]), np.min(im2hu[mask4]))
            max4 = np.maximum(np.max(im1hu[mask4]), np.max(im2hu[mask4]))
            if min4 != max4:
                im1hu4 = (im1hu - min4) / (max4 - min4)
                im2hu4 = (im2hu - min4) / (max4 - min4)
                (dhu0_4[icand], dhu1_4[icand], dhu2_4[icand], dhu3_4[icand], dhu4_4[icand], dhu5_4[icand], \
                 dhu6_4[icand], dhu7_4[icand]) = np.abs(humoments(im1hu4, mask4) - humoments(im2hu4, mask4))
        if np.sum(mask4gt) > 0:
            min4 = np.minimum(np.min(im1hu[mask4gt]), np.min(im2hu[mask4gt]))
            max4 = np.maximum(np.max(im1hu[mask4gt]), np.max(im2hu[mask4gt]))
            if min4 != max4:
                im1hu4 = (im1hu - min4) / (max4 - min4)
                im2hu4 = (im2hu - min4) / (max4 - min4)
                (dhu0_4gt[icand], dhu1_4gt[icand], dhu2_4gt[icand], dhu3_4gt[icand], dhu4_4gt[icand], dhu5_4gt[icand], \
                 dhu6_4gt[icand], dhu7_4gt[icand]) = np.abs(humoments(im1hu4, mask4gt) - humoments(im2hu4, mask4gt))

            
        # Bump feature
        # ----------------------------------------
        if conv1st:
            #diff = im2 - imt
            im1bump2 = im2[maskr2 & np.isfinite(im2) & (im2 < 65000)]
            im1bump = im2[np.isfinite(im2) & (im2 < 65000)]
        else:
            #diff = imt - im1
            im1bump2 = im1[maskr2 & np.isfinite(im1) & (im1 < 65000)]
            im1bump = im1[np.isfinite(im1) & (im1 < 65000)]
        im2bump2 = imt[maskr2 & np.isfinite(imt) & (imt < 65000)]
        im2bump = imt[np.isfinite(imt) & (imt < 65000)]

        try:
            bump[icand] = np.abs(np.average(im1bump2) / np.average(im1bump) - np.average(im2bump2) / np.average(im2bump))
        except:
            "Failed to compute bump for candidate %i" % icand

        # Entropy
        # ------------------------------------------
        im1sum = np.abs(np.sum(im1[np.isfinite(im1) & (im1 < 65000)]))
        im1ent = np.abs(im1[maskr2 & np.isfinite(im1) & (im1 < 65000)]) / im1sum
        entropy[icand] = np.sum(im1ent * np.log(im1ent))

        
        # maximum of imSNR in absolute value
        # ---------------------------------------------
        offset[icand] = rs2D.flatten()[np.argmax(imSNR.flatten())]


        # number of maxima and ratio between median maximum and median of the original images
        # ------------------------------------------------------------------------
        nmax1[icand], ratiomax1[icand] = get_local_maxima(im1hu.reshape((npsf, npsf)), 10)
        nmax2[icand], ratiomax2[icand] = get_local_maxima(im2hu.reshape((npsf, npsf)), 10)


        # coefficient of determination
        # ----------------------------

        R2[icand] =  1. - np.sum(diffPCA**2) / norm[icand]**2


        # standard deviation of difference
        # --------------------------------

        std[icand] = np.std(diffPCA)


        # Shapiro Wilk test for normality
        # -------------------------------
        
        SW[icand] = stats.shapiro(diffPCA)[1]


        # extreme values of reference images
        # ----------------------------------

        bgnoise = np.sqrt(cands[icand, 7]**2 + cands[icand, 8]**2)
        im1min[icand] = np.min(im1) / bgnoise
        im1max[icand] = np.max(im1) / bgnoise
        im2min[icand] = np.min(im2) / bgnoise
        im2max[icand] = np.max(im2) / bgnoise
            

        # cosmic ray diagnostic
        # ---------------------

        im1 = im1.reshape((npsf, npsf))
        im2 = im2.reshape((npsf, npsf))
        CR1[icand] = CR(im1, npsf)
        CR2[icand] = CR(im2, npsf)


        # symmetry index
        # --------------
        
        radialavg = np.average(diff[maskrs])
        val = np.zeros(2 * nthetas)
        err = np.zeros(2 * nthetas)
        for itheta in range(2 * nthetas):
            val[itheta] = np.average(diff[maskthetas[itheta]])
            err[itheta] = np.abs(np.std(diff[maskthetas[itheta]]))
        symmidx[icand] = np.max(np.abs(val - radialavg) / err)

        # fake candidates (when option --addstars)
        # ---------------

        if doadd:

            distfake = (ijadds - np.array([jpixcand[icand], ipixcand[icand]]))**2
            distfake = np.sqrt(distfake[:, 0] + distfake[:, 1])
            closestfake = np.argmin(distfake)

            if distfake[closestfake] < 3: #  probably an artificial source

                attcand = np.array([kratiocand[icand], ksupportcand[icand], \
                           fluxSNRcand[icand], pixSNRcand[icand], \
                           coeffcand[0, icand], \
                           coeffcand[1, icand], coeffcand[2, icand], coeffcand[3, icand], coeffcand[4, icand], coeffcand[5, icand], coeffcand[6, icand], \
                           diffcoeff[icand], crosscorr[icand], crosscorr3[icand], crosscorr5[icand], crosscorr8[icand], dCCPCA0[icand], \
                           dhu0_2[icand], dhu1_2[icand], dhu2_2[icand], dhu3_2[icand], dhu4_2[icand], dhu5_2[icand], dhu6_2[icand], dhu7_2[icand], \
                           dhu0_4[icand], dhu1_4[icand], dhu2_4[icand], dhu3_4[icand], dhu4_4[icand], dhu5_4[icand], dhu6_4[icand], dhu7_4[icand], \
                           dhu0_4gt[icand], dhu1_4gt[icand], dhu2_4gt[icand], dhu3_4gt[icand], dhu4_4gt[icand], dhu5_4gt[icand], dhu6_4gt[icand], dhu7_4gt[icand], \
                           bump[icand], entropy[icand], offset[icand], nmax1[icand], nmax2[icand], ratiomax1[icand], ratiomax2[icand], \
                           R2[icand], SW[icand], std[icand], symmidx[icand], ncand[icand], \
                           max(im1min[icand], im2min[icand]), \
                           min(im1max[icand], im2max[icand]), \
                           max(CR1[icand], CR2[icand])])
                
                # classify with random forest
                if np.isfinite(np.sum(attcand)):
                    probs[icand] = rfc.predict_proba(attcand.reshape(1, -1))[0][1]

                print "Found fake star at position: %i (pix %i %i, type %i, prob %f, flux: %f)" % (closestfake, ipixcand[icand], jpixcand[icand], addtype[closestfake], probs[icand], flux[icand])

                # log into fakestars file
                statsfile.write("FOUND. pix %i %i SNR %f pixSNR %f flux %f %f type %i prob %f\n" % (ipixcand[icand], jpixcand[icand], fluxSNRcand[icand], pixSNRcand[icand], flux[icand], e_flux[icand], addtype[closestfake], probs[icand]))

                fake[icand] = True
                faketype[icand] = addtype[closestfake]

            elif distfake[closestfake] < npsfh * np.sqrt(2.):  # probably a source near an artificial source

                save[icand] = False
                fake[icand] = False
                selected[icand] = False


        # Start Random Forest or manual filtering based on features
        # ---------------------------------------------------------

        # random forest classifier
        # ------------------------
        
        if doML:

            # new random forest input
            attcand = np.array([kratiocand[icand], ksupportcand[icand], \
                       fluxSNRcand[icand], pixSNRcand[icand], \
                       coeffcand[0, icand], \
                       coeffcand[1, icand], coeffcand[2, icand], coeffcand[3, icand], coeffcand[4, icand], coeffcand[5, icand], coeffcand[6, icand], \
                       diffcoeff[icand], crosscorr[icand], crosscorr3[icand], crosscorr5[icand], crosscorr8[icand], dCCPCA0[icand], \
                       dhu0_2[icand], dhu1_2[icand], dhu2_2[icand], dhu3_2[icand], dhu4_2[icand], dhu5_2[icand], dhu6_2[icand], dhu7_2[icand], \
                       dhu0_4[icand], dhu1_4[icand], dhu2_4[icand], dhu3_4[icand], dhu4_4[icand], dhu5_4[icand], dhu6_4[icand], dhu7_4[icand], \
                       dhu0_4gt[icand], dhu1_4gt[icand], dhu2_4gt[icand], dhu3_4gt[icand], dhu4_4gt[icand], dhu5_4gt[icand], dhu6_4gt[icand], dhu7_4gt[icand], \
                       bump[icand], entropy[icand], offset[icand], nmax1[icand], nmax2[icand], ratiomax1[icand], ratiomax2[icand], \
                       R2[icand], SW[icand], std[icand], symmidx[icand], ncand[icand], \
                       max(im1min[icand], im2min[icand]), \
                       min(im1max[icand], im2max[icand]), \
                       max(CR1[icand], CR2[icand])])

            # classify with random forest
            if np.isfinite(np.sum(attcand)):

                # deep learning
                if conv1st:
                    imdeep = np.vstack([normalize_stamp(imt.flatten()), normalize_stamp(im2.flatten()), normalize_stamp(diff), normalize_stamp(imSNR)])
                else:
                    imdeep = np.vstack([normalize_stamp(im1.flatten()), normalize_stamp(imt.flatten()), normalize_stamp(diff), normalize_stamp(imSNR)])
                probDL = deepdetector_nopsf.predict_sn(imdeep)[0][1]

                # random forest
                probRF = rfc.predict_proba(attcand.reshape(1, -1))[0][1]

                ## hybrid classifier
                probRF_hybrid = rfc_hybrid.predict_proba((np.hstack([attcand, probDL])).reshape(1, -1))[0][1]

                # deep learning with psf
                probDL_withpsf = deepdetector_withpsf.predict_sn(np.vstack([imdeep, normalize_stamp(psfref.flatten())]))[0][1]
                print "RF", probRF, "DL", probDL, "RF_hybrid", probRF_hybrid, "DL_psf", probDL_withpsf, fluxSNRcand[icand], crosscorr[icand], bump[icand], offset[icand], ipixcand[icand], jpixcand[icand]

                # probability mode
                if probmode == 'RF':
                    probs[icand] = probRF
                elif probmode == 'DL':
                    probs[icand] = probDL
                elif probmode == 'RF+DL':
                    probs[icand] = probRF_hybrid
                elif probmode == 'max_prob':
                    probs[icand] = max(probRF, probDL, probRF_hybrid, probDL_withpsf)

                # threshold
                selected[icand] = probs[icand] >= prob_threshold

            else:
                selected[icand] = False


        # 'manual' classifier (do not use)
        # --------------------------------
        
        else:

            if CR1[icand] > CRlim or CR2[icand] > CRlim: # probably a cosmic ray
                selected[icand] = False
                if fake[icand] and verbose:
                    print "    CR:", CR1[icand], CR2[icand], ", (SNR: %f)" % fluxSNRcand[icand]
                continue
            
            if im1max[icand] > imlim and im2max[icand] > imlim: # probably a bright stars and a bad subtraction
                selected[icand] = False
                if fake[icand] and verbose:
                    print "    immax:", im1max[icand], im2max[icand], ", (SNR: %f)" % fluxSNRcand[icand]
                continue
            
            if np.min(imSNR) < -4 and np.max(imSNR) > 4: # probably a bad subtraction
                selected[icand] = False
                if fake[icand] and verbose:
                    print "    extremes:", np.min(imSNR), np.max(imSNR), ", (SNR: %f)" % fluxSNRcand[icand]
                continue
            
            if np.abs(crosscorr[icand]) < cclim: # wrong shape or too dim to tell
                selected[icand] = False
                if fake[icand] and verbose:
                    print "    cc:", np.abs(crosscorr[icand]), ", (SNR: %f)" % fluxSNRcand[icand]
                continue
            
            if diffcoeff[icand] >= diffcoefflim: # probably a bad subtraction
                selected[icand] = False
                if fake[icand] and verbose:
                    print "    diffcoeff:", diffcoeff[icand], ", (SNR: %f)" % fluxSNRcand[icand]
                continue
            
            if symmidx[icand] >= symmidxlim: # probably a bad subtraction
                selected[icand] = False
                if fake[icand] and verbose:
                    print "    symmidx:", symmidx[icand], ", (SNR: %f)" % fluxSNRcand[icand]
                continue
            
            if np.abs(np.abs(coeffcand[0, icand]) - np.abs(crosscorr[icand])) >= PCA0ccdifflim: # probably a bad subtraction
                selected[icand] = False
                if fake[icand] and verbose:
                    print "    PCA0 - cc:", np.abs(coeffcand[0, icand]) - np.abs(crosscorr[icand]), ", (SNR: %f)" % fluxSNRcand[icand]
                continue
    #SERIAL-END

    ncan = len(cands)

    print "Feature computation time:", time.time() - lasttime

    printtime("candidateplotting")

    # plot selected candidates (focus on false positives)
    # ---------------------------------------------------

    lasttime = time.time()
    # THIS COULD BE DONE IN PARALLEL (maybe plot only repeated candidates...)
    for icand in range(ncan):

        if doplotcandidates and (selected[icand] and save[icand] and not fake[icand]):


            imSNR = cands[icand, ninfo: ninfo + npsf2].reshape((npsf, npsf))
            im1 = cands[icand, ninfo + npsf2: ninfo + 2 * npsf2].reshape((npsf, npsf))
            im2 = cands[icand, ninfo + 2 * npsf2: ninfo + 3 * npsf2].reshape((npsf, npsf))
            imt = cands[icand, ninfo + 3 * npsf2: ninfo + 4 * npsf2].reshape((npsf, npsf))

            if not (doplotnegatives or fluxSNRcand[icand] > 0):
                continue

            if conv1st:
                diff = im2 - imt
            else:
                diff = imt - im1

            if verbose:
                print "    Plotting candidate %i (fake: %s, selected: %s)" % (icand, fake[icand], selected[icand])
        
            plt.clf()
            fig, ax = plt.subplots(nrows = 1, ncols = 6, figsize = (14, 2.))

            # im1
            ax[0].imshow(im1, interpolation = 'nearest')#, cmap = 'gray')
            ax[0].contour(X, Y, imSNR, 6, colors = 'k')
            ax[0].set_title("%02i: %4.1f - %4.1f (rms: %4.1f)" % (fileref, np.min(im1), np.max(im1), np.sqrt(bg1[icand] / gain)), fontsize = 6)

            # im2
            ax[1].imshow(im2, interpolation = 'nearest')#, cmap = 'gray')
            ax[1].contour(X, Y, imSNR, 6, colors = 'k')
            ax[1].set_title("%02i: %4.1f - %4.1f (rms: %4.1f)" % (idxfilesci, np.min(im2), np.max(im2), np.sqrt(bg2[icand] / gain)), fontsize = 6)

            # imt
            ax[2].imshow(imt, interpolation = 'nearest')#, cmap = 'gray')
            ax[2].contour(X, Y, imSNR, 6, colors = 'k')
            if conv1st:
                ax[2].set_title("%02i conv.: %4.1f - %4.1f" % (fileref, np.min(imt), np.max(imt)), fontsize = 6)
            else:
                ax[2].set_title("%02i conv.: %4.1f - %4.1f" % (idxfilesci, np.min(imt), np.max(imt)), fontsize = 6)
                
            # diff
            ax[3].imshow(diff, interpolation = 'nearest')#, cmap = 'gray')
            ax[3].contour(X, Y, imSNR, 6, colors = 'k')
            ax[3].set_title("Diff: %4.0f - %4.0f (flux: %5.0f)" % (np.min(diff), np.max(diff), fluxcand[icand]), fontsize = 6)

            # candidate
            ax[4].imshow(imSNR, interpolation = 'nearest')#, cmap = 'gray')
            if fake[icand]:
                candstring = "(Fake) SNR"
            else:
                candstring = "SNR"
            ax[4].set_title("pix %s: %4.1f (flux %s: %4.1f)" % (candstring, pixSNRcand[icand], candstring, fluxSNRcand[icand]), fontsize = 6)

            # location
            ax[5].scatter(jpixcand[np.invert(save)], ipixcand[np.invert(save)], marker = '.', c = 'gray', alpha = 0.5, edgecolors = 'none')
            ax[5].scatter(jpixcand[save], ipixcand[save], marker = '.', c = 'b', alpha = 0.5, edgecolors = 'none')
            ax[5].scatter(jpixcand[selected], ipixcand[selected], marker = '.', c = 'r', edgecolors = 'none')
            ax[5].scatter(jpixcand[icand], ipixcand[icand], marker = 'o', facecolors = 'none', s = 50, c = 'r', )
            ax[5].set_title("Location %i, %i (ncand: %i)" % (jpixcand[icand], ipixcand[icand], ncand[icand]), fontsize = 6)
            ax[5].set_xlim(0, nx)
            ax[5].set_ylim(0, ny)
        
            plt.savefig("%s/%s/%s/CANDIDATES/%s_%04i_%04i.png" % (webdir, field, CCD, candfile.replace('.npy', ''), ipixcand[icand], jpixcand[icand]), bbox_inches = 'tight')
            plt.close(fig)


    if verbose:
        print datetime.datetime.now()


    # Plot candidate statistics
    # -------------------------

    doplot = False
    if doplot:

        if verbose:
            print "Plotting candidate statistics..."

        # PCA 0 vs cross correlation
        fig, ax = plt.subplots()
        ax.set_xlabel("Cross-correlation")
        ax.set_ylabel("PCA_0")
        ax.scatter(np.abs(crosscorr)[save], np.abs(coeffcand[0])[save], c = np.abs(fluxSNRcand)[save], marker = '.', s = 1, alpha = 0.5, lw = 1)
        imfake = ax.scatter(np.abs(crosscorr)[save & (fake | selected)], np.abs(coeffcand[0])[save & (fake | selected)], c = np.abs(fluxSNRcand)[save & (fake | selected)], marker = 'o', s = 20, edgecolors = 'none')
        if np.sum(save & fake) > 0:
            ax.scatter(np.abs(crosscorr)[save & fake & np.invert(selected)], np.abs(coeffcand[0])[save & fake & np.invert(selected)], marker = 'o', facecolors = 'none', s = 100, alpha = 0.5)
        ax.scatter(np.abs(crosscorr)[save & selected & np.invert(fake)], np.abs(coeffcand[0])[save & selected & np.invert(fake)], marker = '*', facecolors = 'none', s = 300, alpha = 0.5)
        ax.plot([0, 1], [0, 1], 'gray', alpha = 0.4)
        ax.plot([0, 1], [-PCA0ccdifflim, 1. - PCA0ccdifflim], 'gray', alpha = 0.4, ls = ':')
        ax.plot([0, 1], [PCA0ccdifflim, 1. + PCA0ccdifflim], 'gray', alpha = 0.4, ls = ':')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(imfake, cax = cax)    
        cbar.set_label("SNR")
        ax.set_title("Dot: Fake or selected, Circle: False negative, Star: False positive?", fontsize = 7)
        plt.savefig("%s/%s/%s/CANDIDATES/%s_PCA0vscc.png" % (webdir, field, CCD, candfile))


        # central cross correlation 0 vs cross correlation
        fig, ax = plt.subplots()
        ax.set_xlabel("Cross-correlation")
        ax.set_ylabel("PCA_0")
        ax.scatter(np.abs(crosscorr)[save], np.abs(crosscorrcentral)[save], c = np.abs(fluxSNRcand)[save], marker = '.', s = 1, alpha = 0.5, lw = 1)
        imfake = ax.scatter(np.abs(crosscorr)[save & (fake | selected)], np.abs(crosscorrcentral)[save & (fake | selected)], c = np.abs(fluxSNRcand)[save & (fake | selected)], marker = 'o', s = 20, edgecolors = 'none')
        if np.sum(save & fake) > 0:
            ax.scatter(np.abs(crosscorr)[save & fake & np.invert(selected)], np.abs(crosscorrcentral)[save & fake & np.invert(selected)], marker = 'o', facecolors = 'none', s = 100, alpha = 0.5)
        ax.scatter(np.abs(crosscorr)[save & selected & np.invert(fake)], np.abs(crosscorrcentral)[save & selected & np.invert(fake)], marker = '*', facecolors = 'none', s = 300, alpha = 0.5)
        ax.plot([0, 1], [0, 1], 'gray', alpha = 0.4)
        ax.plot([0, 1], [-PCA0ccdifflim, 1. - PCA0ccdifflim], 'gray', alpha = 0.4, ls = ':')
        ax.plot([0, 1], [PCA0ccdifflim, 1. + PCA0ccdifflim], 'gray', alpha = 0.4, ls = ':')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(imfake, cax = cax)    
        cbar.set_label("SNR")
        ax.set_title("Dot: Fake or selected, Circle: False negative, Star: False positive?", fontsize = 7)
        plt.savefig("%s/%s/%s/CANDIDATES/%s_centralccvscc.png" % (webdir, field, CCD, candfile))


        # dCCPCA0 0 vs cross correlation
        fig, ax = plt.subplots()
        ax.set_xlabel("Cross-correlation")
        ax.set_ylabel("|Cross-correlation - PCA_0|")
        ax.scatter(np.abs(crosscorr)[save], np.abs(dCCPCA0)[save], c = np.abs(fluxSNRcand)[save], marker = '.', s = 1, alpha = 0.5, lw = 1)
        imfake = ax.scatter(np.abs(crosscorr)[save & (fake | selected)], np.abs(dCCPCA0)[save & (fake | selected)], c = np.abs(fluxSNRcand)[save & (fake | selected)], marker = 'o', s = 20, edgecolors = 'none')
        if np.sum(save & fake) > 0:
            ax.scatter(np.abs(crosscorr)[save & fake & np.invert(selected)], np.abs(dCCPCA0)[save & fake & np.invert(selected)], marker = 'o', facecolors = 'none', s = 100, alpha = 0.5)
        ax.scatter(np.abs(crosscorr)[save & selected & np.invert(fake)], np.abs(dCCPCA0)[save & selected & np.invert(fake)], marker = '*', facecolors = 'none', s = 300, alpha = 0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.5)
        cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(imfake, cax = cax)    
        cbar.set_label("SNR")
        ax.set_title("Dot: Fake or selected, Circle: False negative, Star: False positive?", fontsize = 7)
        plt.savefig("%s/%s/%s/CANDIDATES/%s_dCCPCA0vscc.png" % (webdir, field, CCD, candfile))


        # drmoment 0 vs cross correlation
        fig, ax = plt.subplots()
        ax.set_xlabel("Cross-correlation")
        ax.set_ylabel("delta Irr")
        ax.scatter(np.abs(crosscorr)[save], np.abs(drmoment)[save], c = np.abs(fluxSNRcand)[save], marker = '.', s = 1, alpha = 0.5, lw = 1)
        imfake = ax.scatter(np.abs(crosscorr)[save & (fake | selected)], np.abs(drmoment)[save & (fake | selected)], c = np.abs(fluxSNRcand)[save & (fake | selected)], marker = 'o', s = 20, edgecolors = 'none')
        if np.sum(save & fake) > 0:
            ax.scatter(np.abs(crosscorr)[save & fake & np.invert(selected)], np.abs(drmoment)[save & fake & np.invert(selected)], marker = 'o', facecolors = 'none', s = 100, alpha = 0.5)
        ax.scatter(np.abs(crosscorr)[save & selected & np.invert(fake)], np.abs(drmoment)[save & selected & np.invert(fake)], marker = '*', facecolors = 'none', s = 300, alpha = 0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 2)
        cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(imfake, cax = cax)    
        cbar.set_label("SNR")
        ax.set_title("Dot: Fake or selected, Circle: False negative, Star: False positive?", fontsize = 7)
        plt.savefig("%s/%s/%s/CANDIDATES/%s_drmomentvscc.png" % (webdir, field, CCD, candfile))


        # dxmoment vs dymoment
        fig, ax = plt.subplots()
        ax.set_xlabel("delta Ixx")
        ax.set_ylabel("delta Iyy")
        ax.scatter(np.abs(dxmoment)[save], np.abs(dymoment)[save], c = np.abs(fluxSNRcand)[save], marker = '.', s = 1, alpha = 0.5, lw = 1)
        imfake = ax.scatter(np.abs(dxmoment)[save & (fake | selected)], np.abs(dymoment)[save & (fake | selected)], c = np.abs(fluxSNRcand)[save & (fake | selected)], marker = 'o', s = 20, edgecolors = 'none')
        if np.sum(save & fake) > 0:
            ax.scatter(np.abs(dxmoment)[save & fake & np.invert(selected)], np.abs(dymoment)[save & fake & np.invert(selected)], marker = 'o', facecolors = 'none', s = 100, alpha = 0.5)
        ax.scatter(np.abs(dxmoment)[save & selected & np.invert(fake)], np.abs(dymoment)[save & selected & np.invert(fake)], marker = '*', facecolors = 'none', s = 300, alpha = 0.5)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(imfake, cax = cax)    
        cbar.set_label("SNR")
        ax.set_title("Dot: Fake or selected, Circle: False negative, Star: False positive?", fontsize = 7)
        plt.savefig("%s/%s/%s/CANDIDATES/%s_dxmomentvsdymoment.png" % (webdir, field, CCD, candfile))

        # diffcoeff vs cross correlation
        # ------------------------------
        
        fig, ax = plt.subplots()
        ax.set_xlabel("Cross-correlation")
        ax.set_ylabel("diffcoeff") 
        ax.scatter(np.abs(crosscorr)[save], np.abs(diffcoeff)[save], c = np.abs(fluxSNRcand)[save], marker = '.', s = 1, alpha = 0.5, lw = 1)
        imfake = ax.scatter(np.abs(crosscorr)[save & (fake | selected)], np.abs(diffcoeff)[save & (fake | selected)], c = np.abs(fluxSNRcand)[save & (fake | selected)], marker = 'o', s = 20, edgecolors = 'none')
        if np.sum(save & fake) > 0:
            ax.scatter(np.abs(crosscorr)[save & fake & np.invert(selected)], np.abs(diffcoeff)[save & fake & np.invert(selected)], marker = 'o', facecolors = 'none', s = 100, alpha = 0.5)
        ax.scatter(np.abs(crosscorr)[save & selected & np.invert(fake)], np.abs(diffcoeff)[save & selected & np.invert(fake)], marker = '*', facecolors = 'none', s = 300, alpha = 0.5)
        ax.plot([0, 1], [diffcoefflim, diffcoefflim], 'gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.5)
        cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(imfake, cax = cax)    
        cbar.set_label("SNR")
        ax.set_title("Dot: Fake Or Selected, Circle: False negative, Star: False positive?", fontsize = 7)
        plt.savefig("%s/%s/%s/CANDIDATES/%s_diffcoeffvscc.png" % (webdir, field, CCD, candfile))
    
        # std vs cross correlation
        # ------------------------
        
        fig, ax = plt.subplots()
        ax.set_xlabel("Cross-correlation")
        ax.set_ylabel("std")
        ax.scatter(np.abs(crosscorr)[save], np.abs(std)[save], c = np.abs(fluxSNRcand)[save], marker = '.', s = 1, alpha = 0.5, lw = 1)
        imfake = ax.scatter(np.abs(crosscorr)[save & (fake | selected)], np.abs(std)[save & (fake | selected)], c = np.abs(fluxSNRcand)[save & (fake | selected)], marker = 'o', s = 20, edgecolors = 'none')
        if np.sum(save & fake) > 0:
            ax.scatter(np.abs(crosscorr)[save & fake & np.invert(selected)], np.abs(std)[save & fake & np.invert(selected)], marker = 'o', facecolors = 'none', s = 100, alpha = 0.5)
        ax.scatter(np.abs(crosscorr)[save & selected & np.invert(fake)], np.abs(std)[save & selected & np.invert(fake)], marker = '*', facecolors = 'none', s = 300, alpha = 0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 2)
        cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(imfake, cax = cax)    
        cbar.set_label("SNR")
        ax.set_title("Dot: Fake Or Selected, Circle: False negative, Star: False positive?", fontsize = 7)
        plt.savefig("%s/%s/%s/CANDIDATES/%s_stdvscc.png" % (webdir, field, CCD, candfile))
    
        # std vs cross correlation
        # ------------------------
        
        fig, ax = plt.subplots()
        ax.set_xlabel("Cross-correlation")
        ax.set_ylabel("ncand")
        ax.scatter(np.abs(crosscorr)[save], np.abs(ncand)[save], c = np.abs(fluxSNRcand)[save], marker = '.', s = 1, alpha = 0.5, lw = 1)
        imfake = ax.scatter(np.abs(crosscorr)[save & (fake | selected)], np.abs(ncand)[save & (fake | selected)], c = np.abs(fluxSNRcand)[save & (fake | selected)], marker = 'o', s = 20, edgecolors = 'none')
        if np.sum(save & fake) > 0:
            ax.scatter(np.abs(crosscorr)[save & fake & np.invert(selected)], np.abs(ncand)[save & fake & np.invert(selected)], marker = 'o', facecolors = 'none', s = 100, alpha = 0.5)
        ax.scatter(np.abs(crosscorr)[save & selected & np.invert(fake)], np.abs(ncand)[save & selected & np.invert(fake)], marker = '*', facecolors = 'none', s = 300, alpha = 0.5)
        ax.plot([0, 1], [ncandlim, ncandlim], 'gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, np.max(ncand))
        cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(imfake, cax = cax)    
        cbar.set_label("SNR")
        ax.set_title("Dot: Fake Or Selected, Circle: False negative, Star: False positive?", fontsize = 7)
        plt.savefig("%s/%s/%s/CANDIDATES/%s_ncandvscc.png" % (webdir, field, CCD, candfile))
    
        # symmetry index vs cross correlation
        # -----------------------------------
        
        fig, ax = plt.subplots()
        ax.set_xlabel("Cross-correlation")
        ax.set_ylabel("symmidx")
        ax.scatter(np.abs(crosscorr)[save], np.abs(symmidx)[save], c = np.abs(fluxSNRcand)[save], marker = '.', s = 1, alpha = 0.5, lw = 1)
        imfake = ax.scatter(np.abs(crosscorr)[save & (fake | selected)], np.abs(symmidx)[save & (fake | selected)], c = np.abs(fluxSNRcand)[save & (fake | selected)], marker = 'o', s = 20, edgecolors = 'none')
        if np.sum(save & fake) > 0:
            ax.scatter(np.abs(crosscorr)[save & fake & np.invert(selected)], np.abs(symmidx)[save & fake & np.invert(selected)], marker = 'o', facecolors = 'none', s = 100, alpha = 0.5)
        ax.scatter(np.abs(crosscorr)[save & selected & np.invert(fake)], np.abs(symmidx)[save & selected & np.invert(fake)], marker = '*', facecolors = 'none', s = 300, alpha = 0.5)
        ax.plot([0, 1], [symmidxlim, symmidxlim], 'gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 5)
        cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(imfake, cax = cax)    
        cbar.set_label("SNR")
        ax.set_title("Dot: Fake Or Selected, Circle: False negative, Star: False positive?", fontsize = 7)
        plt.savefig("%s/%s/%s/CANDIDATES/%s_symmidxvscc.png" % (webdir, field, CCD, candfile))
    
        # SNR vs cross correlation
        # ------------------------
        
        fig, ax = plt.subplots()
        ax.set_xlabel("Cross-correlation")
        ax.set_ylabel("fluxSNRcand")
        ax.scatter(np.abs(crosscorr)[save], np.abs(fluxSNRcand)[save], c = np.abs(fluxSNRcand)[save], marker = '.', s = 1, alpha = 0.5, lw = 1)
        imfake = ax.scatter(np.abs(crosscorr)[save & (fake | selected)], np.abs(fluxSNRcand)[save & (fake | selected)], c = np.abs(fluxSNRcand)[save & (fake | selected)], marker = 'o', s = 20, edgecolors = 'none')
        if np.sum(save & fake) > 0:
            ax.scatter(np.abs(crosscorr)[save & fake & np.invert(selected)], np.abs(fluxSNRcand)[save & fake & np.invert(selected)], marker = 'o', facecolors = 'none', s = 100, alpha = 0.5)
        ax.scatter(np.abs(crosscorr)[save & selected & np.invert(fake)], np.abs(fluxSNRcand)[save & selected & np.invert(fake)], marker = '*', facecolors = 'none', s = 300, alpha = 0.5)
        ax.plot([0, 1], [SNRlim, SNRlim], 'gray')
        ax.set_xlim(0, 1)
        ax.set_yscale("log")
        cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(imfake, cax = cax)    
        cbar.set_label("SNR")
        ax.set_title("Dot: Fake Or Selected, Circle: False negative, Star: False positive?", fontsize = 7)
        plt.savefig("%s/%s/%s/CANDIDATES/%s_SNRvscc.png" % (webdir, field, CCD, candfile))
    
        # log10SW vs cross correlation
        # ----------------------------
        
        fig, ax = plt.subplots()
        ax.set_xlabel("Cross-correlation")
        ax.set_ylabel("log10 SW")
        ax.scatter(np.abs(crosscorr), np.log10(SW), c = np.abs(fluxSNRcand), marker = '.', s = 1, alpha = 0.5, lw = 1)
        imfake = ax.scatter(np.abs(crosscorr)[save & (fake | selected)], np.log10(SW)[save & (fake | selected)], c = np.abs(fluxSNRcand)[save & (fake | selected)], marker = 'o', s = 20, edgecolors = 'none')
        if np.sum(save & fake) > 0:
            ax.scatter(np.abs(crosscorr)[save & fake & np.invert(selected)], np.log10(SW)[save & fake & np.invert(selected)], marker = 'o', facecolors = 'none', s = 100, alpha = 0.5)
        ax.scatter(np.abs(crosscorr)[save & selected & np.invert(fake)], np.log10(SW)[save & selected & np.invert(fake)], marker = '*', facecolors = 'none', s = 300, alpha = 0.5)
        ax.set_xlim(0, 1)
        cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(imfake, cax = cax)    
        cbar.set_label("SNR")
        ax.set_title("Dot: Fake Or Selected, Circle: False negative, Star: False positive?", fontsize = 7)
        plt.savefig("%s/%s/%s/CANDIDATES/%s_log10SWvscc.png" % (webdir, field, CCD, candfile))


    # save features binary file
    # ---------------------------------

    if conv1st:
        fileout \
            = "%s/%s/%s/CANDIDATES/features_%s_%s_%02i-%02it_grid%02i_%s.npy" % (sharedir, field, CCD, field, CCD, idxfilesci, fileref, fileref, resampling)
    else:
        fileout \
            = "%s/%s/%s/CANDIDATES/features_%s_%s_%02it-%02i_grid%02i_%s.npy" % (sharedir, field, CCD, field, CCD, idxfilesci, fileref, fileref, resampling)
    bgnoise = np.sqrt(bg1[save]**2 + bg2[save]**2)

    features = np.vstack([
        ipixcand[save], jpixcand[save], kratiocand[save], ksupportcand[save], fluxSNRcand[save], pixSNRcand[save], bg1[save], bg2[save], MJDrefcand[save], deltaMJDcand[save], \
        fake[save], faketype[save], probs[save], coeffcand[:, save], coeffabscand[:, save], \
        diffcoeff[save], flux[save], e_flux[save], crosscorr[save], crosscorr3[save], crosscorr5[save], crosscorr8[save], dCCPCA0[save], \
        dhu0_2[save], dhu1_2[save], dhu2_2[save], dhu3_2[save], dhu4_2[save], dhu5_2[save], dhu6_2[save], dhu7_2[save], \
        dhu0_4[save], dhu1_4[save], dhu2_4[save], dhu3_4[save], dhu4_4[save], dhu5_4[save], dhu6_4[save], dhu7_4[save], \
        dhu0_4gt[save], dhu1_4gt[save], dhu2_4gt[save], dhu3_4gt[save], dhu4_4gt[save], dhu5_4gt[save], dhu6_4gt[save], dhu7_4gt[save], \
        bump[save], entropy[save], offset[save], nmax1[save], nmax2[save], ratiomax1[save], ratiomax2[save], \
        norm[save], R2[save], SW[save], std[save], symmidx[save], ncand[save], \
        im1min[save], im1max[save], im2min[save], im2max[save], \
        CR1[save], CR2[save]]).transpose()
    cands = cands[save]

    if verbose:
        print "----> Features and cands shapes:", np.shape(features), np.shape(cands)

    # when adding star add features more carefully
    if doadd:
        masknan = np.isfinite(np.sum(features, axis = 1))
        if np.sum(masknan) == 0:
            print "WARNING: all candidates contain nan features"
        masklarge = (np.abs(np.sum(features, axis = 1)) < 1e10)
        features = features[masknan & masklarge]
        cands = cands[masknan & masklarge]
        save = save[masknan & masklarge]
    np.save(fileout, features)

    print np.shape(features), np.shape(cands)

    # trim and update candidates file
    # -------------------------------

    if verbose:
        print "Updating %s file after triming" % candfile
    np.save("%s/%s/%s/CANDIDATES/%s" % (sharedir, field, CCD, candfile), cands)

    
# Do candidate analysis
#########################################

printtime("analyze")

if docandidates:

    if verbose:
        print "\n\nEpoch %i" % (filesci)

    docandidateanalysis(filesci)


# define routine to plot catalogue information and look for matches
# -----------------------------------------------------------------

def plotcatalogue(ax, jsonname, RAavg, DECavg, ipixavg, jpixavg, isrepeated, positiveflux, probpair, labels, RAmin, RAmax, DECmin, DECmax, sharedir, field, CCD, singlecands):

    # booleans for public data
    GAIAdata = False
    USNOdata = False
    SDSSdata = False
    Simbaddata = False
    GCVSdata = False
    RecentSNedata = False
    UnclassifiedSNedata = False
    MPdata = False

    # matches star boolean
    matchesstar = np.zeros(len(isrepeated), dtype = bool)


    # load catalogue data
    # --------------------
    
    catfiles = os.listdir("%s/%s/%s/CALIBRATIONS" % (sharedir, field, CCD))
    docatalogues = True
    for catfile in catfiles:
        if re.match("(.*?)_%s_%s_(\d\d).npy$" % (field, CCD), catfile) and docatalogues:
            (catalogue, epoch) = re.findall("(.*?)_%s_%s_(\d\d).npy$" % (field, CCD), catfile)[0]
            catdata = np.load("%s/%s/%s/CALIBRATIONS/%s" % (sharedir, field, CCD, catfile))
            if catalogue == 'GAIA':
                try:
                    if verbose:
                        print "    Loading %s data..." % catalogue
                    (GAIARA, GAIADEC, GAIAID, GAIA_G) = catdata
                    if catdata != []:
                        USNOdata = True
                except:
                    print "---> WARNING: Cannot load GAIA data..."
            elif catalogue == 'USNO':
                try:
                    if verbose:
                        print "    Loading %s data..." % catalogue
                    (USNORA, USNODEC, USNONAME, USNO_B, USNO_R) = catdata
                    USNORA = np.array(USNORA, dtype = float)
                    USNODEC = np.array(USNODEC, dtype = float)
                    USNONAME = np.array(USNONAME)
                    USNO_B = np.array(USNO_B, dtype = float)
                    USNO_R = np.array(USNO_R, dtype = float)
                    if catdata != []:
                        USNOdata = True
                except:
                    print "---> WARNING: Cannot load USNO data..."
            elif catalogue == 'SDSS':
                try: 
                    if verbose:
                        print "    Loading %s data..." % catalogue
                    try:
                        (SDSSRA, SDSSDEC, SDSS_ObjID, SDSS_psfMag_u, SDSS_psfMagErr_u, SDSS_psfMag_g, SDSS_psfMagErr_g, SDSS_psfMag_r, SDSS_psfMagErr_r, SDSS_psfMag_i, SDSS_psfMagErr_i, SDSS_psfMag_z, SDSS_psfMagErr_z) = catdata
                    except:
                        (SDSSRA, SDSSDEC, SDSS_ObjID, SDSS_psfMag_u, SDSS_psfMagErr_u, SDSS_psfMag_g, SDSS_psfMagErr_g, SDSS_psfMag_r, SDSS_psfMagErr_r) = catdata
                    SDSSRA = np.array(SDSSRA, dtype = float)
                    SDSSDEC = np.array(SDSSDEC, dtype = float)
                    SDSSNAME = np.array(SDSS_ObjID, dtype = str)
                    if catdata != []:
                        SDSSdata = True
                except:
                    print "---> WARNING: Cannot load %s data file..." % catalogue
            elif catalogue == 'Simbad':
                try:
                    if verbose:
                        print "    Loading %s data..." % catalogue
                    (SimbadRA, SimbadDEC, SimbadNAME, SimbadTYPE) = catdata
                    SimbadRA = np.array(SimbadRA, dtype = float)
                    SimbadDEC = np.array(SimbadDEC, dtype = float)
                    SimbadNAME = np.array(SimbadNAME, dtype = str)
                    SimbadTYPE = np.array(SimbadTYPE, dtype = str)
                    if catdata != []:
                        Simbaddata = True
                except:
                    print "---> WARNING: Cannot load %s data file..." % catalogue

            elif catalogue == 'GCVS':
                try:
                    if verbose:
                        print "    Loading %s data..." % catalogue
                    (GCVSRA, GCVSDEC, GCVSNAME, GCVSTYPE) = catdata
                    GCVSRA = np.array(GCVSRA, dtype = float)
                    GCVSDEC = np.array(GCVSDEC, dtype = float)
                    GCVSNAME = np.array(GCVSNAME, dtype = str)
                    if catdata != []:
                        GCVSdata = True
                except:
                    print "---> WARNING: Cannot load %s data file..." % catalogue
            elif catalogue == 'MP':
                try:
                    if verbose:
                        print "    Loading %s data (epoch %i)..." % (catalogue, int(epoch))
                    (MPNAMEe, MPRAe, MPDECe, MPMJDe) = catdata
                    if MPdata:
                        MPRA = np.hstack((MPRA, np.array(MPRAe, dtype = float)))
                        MPDEC = np.hstack((MPDEC, np.array(MPDECe, dtype = float)))
                        MPNAME = np.hstack((MPNAME, np.array(MPNAMEe, dtype = str)))
                        MPMJD = np.hstack((MPMJD, np.array(MPMJDe, dtype = str)))
                    else:
                        MPRA = np.array(MPRAe, dtype = float)
                        MPDEC = np.array(MPDECe, dtype = float)
                        MPNAME = np.array(MPNAMEe, dtype = str)
                        MPMJD = np.array(MPMJDe, dtype = float)
                        MPdata = True
                except:
                    print "---> WARNING: Cannot load %s data file..." % catalogue
            elif catalogue == 'RecentSNe':
                try:
                    (RecentSNeRA, RecentSNeDEC, RecentSNeNAME, RecentSNeTYPE) = catdata
                    RecentSNeRA = np.array(RecentSNeRA, dtype = float)
                    RecentSNeDEC = np.array(RecentSNeDEC, dtype = float)
                    RecentSNeNAME = np.array(RecentSNeNAME, dtype = str)
                    if catdata != []:
                        RecentSNedata = True
                except:
                    print "---> WARNING: Cannot load %s data file..." % catalogue
            elif catalogue == 'UnclassifiedSNe':
                try:
                    (UnclassifiedSNeRA, UnclassifiedSNeDEC, UnclassifiedSNeNAME) = catdata
                    UnclassifiedSNeRA = np.array(UnclassifiedSNeRA, dtype = float)
                    UnclassifiedSNeDEC = np.array(UnclassifiedSNeDEC, dtype = float)
                    UnclassifiedSNeNAME = np.array(UnclassifiedSNeNAME, dtype = str)
                    if catdata != []:
                        UnclassifiedSNedata = True
                except:
                    print "---> WARNING: Cannot load %s data file..." % catalogue


    # Public catalogue summary and plots
    # ----------------------------------

    jsonpositiverepeated = [[RAmin, RAmax, DECmin, DECmax], field, CCD, fileref]
    jsonALL = [[RAmin, RAmax, DECmin, DECmax], field, CCD, fileref]
    jsonpublic = []
    
    if verbose:                
        print "\n    GAIA: %s, USNO: %s, SDSS: %s, MP: %s, Simbad: %s, GCVS: %s, RecentSNe: %s, UnclassifiedSNe: %s" % (GAIAdata, USNOdata, SDSSdata, MPdata, Simbaddata, GCVSdata, RecentSNedata, UnclassifiedSNedata)

    if GAIAdata:
        jsonpublic.append(['GAIA', len(GAIARA), GAIARA.tolist(), GAIADEC.tolist(), GAIANAME.tolist()])
        if ax != None:
            ax.scatter(GAIARA, GAIADEC, marker = '*', alpha = 0.5, edgecolors = 'none', s = 50, c = 'b')
    if USNOdata:
        jsonpublic.append(['USNO', len(USNORA), USNORA.tolist(), USNODEC.tolist(), USNONAME.tolist()])
        if ax != None:
            ax.scatter(USNORA, USNODEC, marker = '*', alpha = 0.5, edgecolors = 'none', s = 50, c = 'b')
    if SDSSdata:
        jsonpublic.append(['SDSS', len(SDSSRA), SDSSRA.tolist(), SDSSDEC.tolist(), SDSSNAME.tolist()])
        if ax != None:
            ax.scatter(SDSSRA, SDSSDEC, marker = 's', alpha = 0.5, s = 10, edgecolors = 'none', c = 'b')
    if Simbaddata:
        jsonpublic.append(['Simbad', len(SimbadRA), SimbadRA.tolist(), SimbadDEC.tolist(), SimbadTYPE.tolist(), SimbadNAME.tolist()])
        if ax != None:
            ax.scatter(SimbadRA, SimbadDEC, marker = 'o', alpha = 0.5, edgecolors = 'none', s = 100, c = 'b')
        for i in range(len(SimbadRA)):
            if SimbadRA[i] >= RAmin and SimbadRA[i] <= RAmax and SimbadDEC[i] >= DECmin and SimbadDEC[i] <= DECmax:
                if SimbadRA[i] > RAmin + (RAmax - RAmin) / 3.:
                    if ax != None:
                        ax.text(SimbadRA[i], SimbadDEC[i], "%s" % (SimbadTYPE[i]), fontsize = 7, ha = 'left')
                else:
                    if ax != None:
                        ax.text(SimbadRA[i], SimbadDEC[i], "%s" % (SimbadTYPE[i]), fontsize = 7, ha = 'right')
    if GCVSdata:
        jsonpublic.append(['GCVS', len(GCVSRA), GCVSRA.tolist(), GCVSDEC.tolist(), GCVSNAME.tolist()])
        if ax != None:
            ax.scatter(GCVSRA, GCVSDEC, marker = '*', alpha = 0.5, edgecolors = 'none', s = 100, c = 'red')
    if MPdata:
        jsonpublic.append(['MP', len(MPRA), MPRA.tolist(), MPDEC.tolist(), MPMJD.tolist(), MPNAME.tolist()])
        idxname = np.argsort(MPNAME)
        MPRA = MPRA[idxname]
        MPDEC = MPDEC[idxname]
        MPNAME = MPNAME[idxname]
        namei = 'thisisnotaname'
        for i in range(len(MPNAME)):
            if namei != MPNAME[i]:
                namei = MPNAME[i]
                MPRAplot = MPRA[MPNAME == namei]
                MPDECplot = MPDEC[MPNAME == namei]
                idxRA = np.argsort(MPRAplot)
                MPRAplot = MPRAplot[idxRA]
                MPDECplot = MPDECplot[idxRA]
                if ax != None:
                    ax.errorbar(MPRAplot, MPDECplot, marker = 's', alpha = 0.5, markersize = 10, c = 'red')
    if RecentSNedata:
        jsonpublic.append(['RecentSNe', len(RecentRA), RecentSNeRA.tolist(), RecentSNeDEC.tolist(), RecentSNeNAME.tolist()])
        if ax != None:
            ax.scatter(RecentSNeRA, RecentSNeDEC, marker = 'o', alpha = 0.5, edgecolors = 'none', s = 100, c = 'red')
    if UnclassifiedSNedata:
        jsonpublic.append(['UnclassifiedSNe', len(UnclassifiedSNeRA), UnclassifiedSNeRA.tolist(), UnclassifiedSNeDEC.tolist(), UnclassifiedSNeNAME.tolist()])
        if ax != None:
            ax.scatter(UnclassifiedSNeRA, UnclassifiedSNeDEC, marker = 'o', alpha = 0.5, edgecolors = 'none', s = 200, c = 'red')


    # look for closest matches with catalogue
    # ---------------------------------------
    
    for i in range(len(RAavg)):
        if i == 0:
            if ax != None:
                ax.text(RAmin, DECmin - (DECmax - DECmin) / 12., "* Simbad types in http://simbad.u-strasbg.fr/simbad/sim-display?data=otypes", fontsize = 6, ha = 'right')
        if USNOdata:
            dist = ((RAavg[i] - USNORA) * 15.)**2 + (DECavg[i] - USNODEC)**2
            idxmin = np.argmin(dist)
            nrmsmatch = 4.
            if np.sqrt(dist[idxmin]) / rmsdeg < nrmsmatch:
                if ax != None:
                    ax.scatter(RAavg[i], DECavg[i], marker = 'x', s = 200, c = 'b', lw = 1)
                print "    Possible match with USNO source %s (mag B: %f, mag R: %f) at RA: %f, DEC: %f (pixel %i, %i, delta: %f arcsec)" % (USNONAME[idxmin], USNO_B[idxmin], USNO_R[idxmin], RAavg[i], DECavg[i], ipixavg[i], jpixavg[i], np.sqrt(dist[idxmin]) * 60. * 60.)
                matchesstar[i] = True
        if SDSSdata:
            dist = ((RAavg[i] - SDSSRA) * 15.)**2 + (DECavg[i] - SDSSDEC)**2
            idxmin = np.argmin(dist)
            if np.sqrt(dist[idxmin]) / rmsdeg < nrmsmatch:
                if ax != None:
                    ax.scatter(RAavg[i], DECavg[i], marker = 'x', s = 200, c = 'b', lw = 1)
                print "    Possible match with SDSS source %s at RA: %f, DEC: %f (pixel %i, %i, delta: %f arcsec)" % (SDSSNAME[idxmin], RAavg[i], DECavg[i], ipixavg[i], jpixavg[i], np.sqrt(dist[idxmin]) * 60. * 60.)
                matchesstar[i] = True
        if Simbaddata:
            dist = ((RAavg[i] - SimbadRA) * 15.)**2 + (DECavg[i] - SimbadDEC)**2
            idxmin = np.argmin(dist)
            if np.sqrt(dist[idxmin]) / rmsdeg < nrmsmatch:
                if ax != None:
                    ax.scatter(RAavg[i], DECavg[i], marker = 'x', s = 200, c = 'b', lw = 1)
                print "    Possible match with Simbad source %s at RA: %f, DEC: %f (pixel %i, %i, delta: %f arcsec)" % (SimbadNAME[idxmin], RAavg[i], DECavg[i], ipixavg[i], jpixavg[i], np.sqrt(dist[idxmin]) * 60. * 60.)
        if GCVSdata:
            dist = ((RAavg[i] - GCVSRA) * 15.)**2 + (DECavg[i] - GCVSDEC)**2
            idxmin = np.argmin(dist)
            if np.sqrt(dist[idxmin]) / rmsdeg < nrmsmatch:
                if ax != None:
                    ax.scatter(RAavg[i], DECavg[i], marker = 'x', s = 200, c = 'r', lw = 1)
                print "    Possible match with GCVS source %s at RA: %f, DEC: %f (pixel %i, %i, delta: %f arcsec)" % (GCVSNAME[idxmin], RAavg[i], DECavg[i], ipixavg[i], jpixavg[i], np.sqrt(dist[idxmin]) * 60. * 60.)
                matchesstar[i] = True
        if MPdata:
            dist = ((RAavg[i] - MPRA) * 15.)**2 + (DECavg[i] - MPDEC)**2
            idxmin = np.argmin(dist)
            if np.sqrt(dist[idxmin]) / rmsdeg < nrmsmatch:
                if ax != None:
                    ax.scatter(RAavg[i], DECavg[i], marker = 'x', s = 200, c = 'r', lw = 1)
                print "    Possible match with MP source %s at RA: %f, DEC: %f (pixel %i, %i, delta: %f arcsec)" % (MPNAME[idxmin], RAavg[i], DECavg[i], ipixavg[i], jpixavg[i], np.sqrt(dist[idxmin]) * 60. * 60.)
        if RecentSNedata:
            dist = ((RAavg[i] - RecentSNeRA) * 15.)**2 + (DECavg[i] - RecentSNeDEC)**2
            idxmin = np.argmin(dist)
            if np.sqrt(dist[idxmin]) / rmsdeg < nrmsmatch:
                if ax != None:
                    ax.scatter(RAavg[i], DECavg[i], marker = 'x', s = 200, c = 'r', lw = 1)
                print "    Possible match with RecentSNe source %s at RA: %f, DEC: %f (pixel %i, %i, delta: %f arcsec)" % (RecentSNeNAME[idxmin], RAavg[i], DECavg[i], ipixavg[i], jpixavg[i], np.sqrt(dist[idxmin]) * 60. * 60.)
        if UnclassifiedSNedata:
            dist = ((RAavg[i] - UnclassifiedSNeRA) * 15.)**2 + (DECavg[i] - UnclassifiedSNeDEC)**2
            idxmin = np.argmin(dist)
            if np.sqrt(dist[idxmin]) / rmsdeg < nrmsmatch:
                if ax != None:
                    ax.scatter(RAavg[i], DECavg[i], marker = 'x', s = 200, c = 'r', lw = 1)
                print "    Possible match with UnclassifiedSNe source %s at RA: %f, DEC: %f (pixel %i, %i, delta: %f arcsec)" % (UnclassifiedSNeNAME[idxmin], RAavg[i], DECavg[i], ipixavg[i], jpixavg[i], np.sqrt(dist[idxmin]) * 60. * 60.)
                      

    # prepare title according to data found
    # -------------------------------------
                      
    titlestring = 'USNO (blue stars)'
    if SDSSdata:
        titlestring = "%s + SDSS (blue sq.)" % titlestring
    if Simbaddata:
        titlestring = "%s + Simbad (blue circ.)" % titlestring
    if GCVSdata:
        titlestring = "%s + GCVS (red stars)" % titlestring
    if MPdata:
        titlestring = "%s + MP (red sq.)" % titlestring
    if RecentSNedata:
        titlestring = "%s + SNe (red stars)" % titlestring
    if UnclassifiedSNedata:
        titlestring = "%s + SNe cand. (big red stars)" % titlestring
    if singlecands:
        titlestring = "%s + cand. (single: empty sq., multiple: green circ.) + cand. match (cross)" % titlestring
    else:
        titlestring = "%s + cand. (green circ.) + cand. match (cross)" % titlestring
    if ax != None:
        ax.set_title(titlestring, fontsize = 7)


    # compute candidate image coordinates
    # -----------------------------------
    
    (icoord, jcoord) = (74 + (841 - 64) * (RAmax - RAavg) / (RAmax - RAmin), 16 + (415 - 16) * (DECmax - DECavg) / (DECmax - DECmin))

    # add candidates to json vector and save file
    # -------------------------------------------

    labels = np.array(labels, dtype = str)

    jsonpositiverepeated.append(['Candidates', len(ipixavg[isrepeated & positiveflux]), (ipixavg[isrepeated & positiveflux]).tolist(), (jpixavg[isrepeated & positiveflux]).tolist(), (RAavg[isrepeated & positiveflux]).tolist(), (DECavg[isrepeated & positiveflux]).tolist(), (isrepeated[isrepeated & positiveflux]).tolist(), (positiveflux[isrepeated & positiveflux]).tolist(), (probpair[isrepeated & positiveflux]).tolist(), (labels[isrepeated & positiveflux]).tolist(), (matchesstar[isrepeated & positiveflux]).tolist()])
    with open(jsonname, 'w') as jsonfile:
        jsonfile.write(json.dumps(jsonpositiverepeated))

    jsonALL.append(['Candidates', len(ipixavg), ipixavg.tolist(), jpixavg.tolist(), RAavg.tolist(), DECavg.tolist(), isrepeated.tolist(), positiveflux.tolist(), probpair.tolist(), labels.tolist(), matchesstar.tolist()])
    with open(jsonname.replace(".json", "_single.json"), 'w') as jsonfile:
        jsonfile.write(json.dumps(jsonALL))

    with open(jsonname.replace(".json", "_public.json"), 'w') as jsonfile:
        jsonfile.write(json.dumps(jsonpublic))


    # return values
    # -------------
    
    return (icoord, jcoord, np.sum(np.invert(matchesstar) & isrepeated))


# Filter candidates using all available epochs
# ###########################################

printtime("dofilter")

if dofilter:


    # restore color map
    # -----------------
    
    cmap = 'gray'


    # load reference exptime and airmass
    # ----------------------------------
    
    fitsref = fits.open("%s/%s/%s/%s_%s_%02i_image_crblaster.fits" % (refdir, field, CCD, field, CCD, fileref))
    headerref = fitsref[0].header
    MJDref = float(headerref['MJD-OBS'])
    del fitsref
    try:
        exptime = float(headerref['EXPTIME'])
    except:
        print "\n\nWARNING: Cannot find field 'EXPTIME'\n\n"
        sys.exit(12)
    try:
        airmass = float(headerref['AIRMASS'])
    except:
        print "\n\nWARNING: Cannot find field 'AIRMASS'\n\n"
        sys.exit(12)

    if verbose:
        print "\n\nLooking for candidate coincidences among different epochs...\n"


    # delete already created timeseries
    # ---------------------------------

    command = "rm -rf %s/%s/%s/CANDIDATES/*timeseries*grid%02i*" % (sharedir, field, CCD, fileref)
    if verbose:
        print "    %s\n" % command
    os.system(command)
    command = "rm -rf %s/%s/%s/CANDIDATES/*timeseries*grid%02i*" % (webdir, field, CCD, fileref)
    if verbose:
        print "    %s\n" % command
    os.system(command)
    command = "rm -rf %s/%s/%s/*timeseries*grid%02i*" % (sharedir, field, CCD, fileref)
    if verbose:
        print "    %s\n" % command
    os.system(command)
    command = "rm -rf %s/%s/%s/*timeseries*grid%02i*" % (webdir, field, CCD, fileref)
    if verbose:
        print "    %s\n" % command
    os.system(command)


    # load data necessary for absolute coordinate transformation (select data with lowest rms)
    # ----------------------------------------------------------
    
    RADECmatch = "matchRADEC_%s_%s_(.*?)-%02i.npy$" % (field, CCD, fileref)
    calfiles = os.listdir("%s/%s/%s/CALIBRATIONS" % (sharedir, field, CCD))
    rmsdeg = 1e9
    for calfile in calfiles:
        if not re.match(RADECmatch, calfile):
            continue
        matchRADEC = np.load("%s/%s/%s/CALIBRATIONS/%s" % (sharedir, field, CCD, calfile))
        
        if matchRADEC[2] < rmsdeg:
            (afluxADUB, e_afluxADUB, rmsdeg, CRVAL[0, 0], CRVAL[0, 1], CRPIX[0, 0], CRPIX[0, 1], CD[0, 0, 0], CD[0, 0, 1], CD[0, 1, 0], CD[0, 1, 1], nPV1, nPV2, ordersol) = matchRADEC[0:14]

            # unpack sol_astrometry_RADEC terms
            nend = 20
            if ordersol == 2:
                nend = 26
            elif ordersol == 3:
                nend = 34
            sol_astrometry_RADEC = matchRADEC[14: nend]

            # unpack PV terms
            PV[0] = matchRADEC[nend: nend + int(nPV1 * nPV2)].reshape((int(nPV1), int(nPV2)))


    ## load USNO data
    ## --------------
    #
    #try:
    #    (USNORA, USNODEC, USNONAME, USNO_B, USNO_R) = np.load("%s/%s/%s/CALIBRATIONS/USNO_%s_%s_%02i.npy" % (sharedir, field, CCD, field, CCD, fileref))
    #except:
    #    print "---> WARNING: Cannot load USNO data..."


    # look for matches
    # ----------------
    
    candmatch = "cand_%s_%s_(.*?)_grid%02i_%s.npy$" % (field, CCD, fileref, resampling)
    candfiles = os.listdir("%s/%s/%s/CANDIDATES" % (sharedir, field, CCD))
    features = None
    cands = None
    conv1starray = None
    labels = None
    # loop among candidates npy files
    for candfile in candfiles:
        if re.search(candmatch, candfile):
            stringdiff = re.findall(candmatch, candfile)[0]
            featurefile = "features_%s_%s_%s_grid%02i_%s.npy" % (field, CCD, stringdiff, fileref, resampling)
            if not os.path.exists("%s/%s/%s/CANDIDATES/%s" % (sharedir, field, CCD, featurefile)):
                print "Missing feature file: %s" % featurefile
                docandidateanalysis(int(stringdiff[0:2]))
            if features is None:
                features = np.load("%s/%s/%s/CANDIDATES/%s" % (sharedir, field, CCD, featurefile))
                cands = np.load("%s/%s/%s/CANDIDATES/%s" % (sharedir, field, CCD, candfile))
                conv1starray = np.zeros(np.shape(cands)[0], dtype = bool)
                conv1starray[:] = (stringdiff[2] != 't')
                labels = np.zeros(np.shape(cands)[0], dtype = object)
                labels[:] = stringdiff
            else:
                features = np.vstack([features, np.load("%s/%s/%s/CANDIDATES/%s" % (sharedir, field, CCD, featurefile))]) 
                cands = np.vstack([cands, np.load("%s/%s/%s/CANDIDATES/%s" % (sharedir, field, CCD, candfile))])
                conv1st = np.zeros(np.shape(cands)[0] - len(conv1starray), dtype = bool)
                conv1st[:] = (stringdiff[2] != 't')
                conv1starray = np.concatenate((conv1starray, conv1st))
                labelaux = np.zeros(len(conv1st), dtype = object)
                labelaux[:] = stringdiff
                labels = np.concatenate((labels, labelaux))

    if verbose:
        print "    Features shape:", np.shape(features)
        print "    cands shape:", np.shape(cands)
        print "    conv1st shape:", np.shape(conv1starray)
        if np.shape(cands)[0] != np.shape(features)[0]:
            print "\n\nWARNING: it appears that the --dofilter option was used after --dosubtraction without the --candidates option. Run again including the --candidates option\n\n"
            sys.exit(21)

    # check for coincidences
    ipixall = features[:, 0]
    jpixall = features[:, 1]
    kratioall = features[:, 2]
    ksupportall = features[:, 3]
    refMJD = features[:, 8]
    deltaMJD = features[:, 9]
    fake = np.array(features[:, 10], dtype = bool)
    probs = features[:, 12]
    selected = (probs >= prob_threshold)
    possible = (probs != 0)
    snr = features[:, 4]
    flux = features[:, 13 + 2 * nk + 1]
    e_flux = features[:, 13 + 2 * nk + 2]

    repeated = np.ones(len(ipixall))
    probpair = np.zeros(len(ipixall))
    check = np.ones(len(ipixall), dtype = bool)

    if verbose:
        print "    Number of selected candidates (not fake):", np.sum(selected & np.invert(fake))

    
    # loop among all selected candidates to check for coincidences
    # ------------------------------------------------------------
    
    radlim = 3.
    ipixmany = []
    jpixmany = []
    probpairmany = []
    labelsmany = []
    ntimes = []
    fluxmax = []

#    # common to all plots
#    fig, ax = plt.subplots(ncols = 20, nrows = 3, sharex = True, sharey = True, figsize = (0.8 * 20, 2.4))
#    ax[1, 0].set_ylabel("im1", fontsize = 8, labelpad = 0)
#    ax[0, 0].set_ylabel("im2", fontsize = 8, labelpad = 0)
#    ax[2, 0].set_ylabel("im2 - im1", fontsize = 8, labelpad = 0)
#    ax[0, 0].yaxis.label.set_color("white")
#    ax[1, 0].yaxis.label.set_color("white")
#    ax[2, 0].yaxis.label.set_color("white")
    
    for i in range(len(ipixall)):
    
        if fake[i] and not savefakes:
            continue
        

        # loop to look for coincidences with selections
        # ---------------------------------------------
        
        maskdist = np.array(np.sqrt((ipixall[i] - ipixall)**2 + (jpixall[i] - jpixall)**2) < radlim, dtype = bool)
        maskcoincidence = maskdist & check & selected
        maskpossiblecoincidence = maskdist & check & possible & np.invert(selected)
        check[maskcoincidence] = False

    
        # if selected at least once, save number of coincidences and plot timeseries with available information
        # -----------------------------------------------------------------------------------------------------

        if np.sum(maskcoincidence) > 1:

            repeated[maskcoincidence] = np.sum(maskcoincidence) 

            # compute probability between pairs of candidates using maximum snr coincidence
            # search for two maximum snr from available times
            # ------------------------------------------------------------------------------------------

            (idxsnr1, idxsnr2) = np.where(maskcoincidence)[0][np.argsort(snr[maskcoincidence])[::-1][:2]]       
            
            # extract features
            """
            output from SupernovaCandidateData_multiple.py
            4 flux SNR
	    5 pix SNR
	    10 --> real or artifact
	    13 PCAnorm
	    30 crosscorr
	    31 crosscorr3
	    32 crosscorr5
	    33 crosscorr8
	    34 dCCPCA0
	    35 dhu0_2
	    36 dhu1_2
	    37 dhu2_2
	    38 dhu3_2
	    39 dhu4_2
	    43 dhu0_4
	    44 dhu1_4
	    51 dhu0_4gt
	    59 bump
	    60 entropy
	    61 offset
	    64 ratiomax1
	    68 SW
	    71 ncand
	    75 minimmax
	    """

            pairfeatidx = [4, 5, 13, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 43, 44, 51, 59, 60, 61, 64, 68, 71, 75]
            attpaircand = np.array(np.hstack((features[idxsnr1, pairfeatidx], features[idxsnr2, pairfeatidx])))
            if dorfcpair:
                thisprob = rfcpair.predict_proba(attpaircand.reshape(1, -1))[0][1]
            else:
                thisprob = 0.5
            probpair[maskcoincidence] = thisprob
            probpairmany.append(thisprob)

            # sort
            MJDref = refMJD[maskcoincidence][0]
            dates = deltaMJD[maskcoincidence] + MJDref
            datespossible = deltaMJD[maskpossiblecoincidence] + MJDref

            idxsort = np.argsort(dates)
            idxsortpossible = np.argsort(datespossible)

            dates = dates[idxsort]
            datespossible = datespossible[idxsortpossible]

            fluxes = flux[maskcoincidence][idxsort]
            fluxespossible = flux[maskpossiblecoincidence][idxsortpossible]
            e_fluxes = e_flux[maskcoincidence][idxsort]
            e_fluxespossible = e_flux[maskpossiblecoincidence][idxsortpossible]
            
            probabilities = probs[maskcoincidence][idxsort]
            probabilitiespossible = probs[maskpossiblecoincidence][idxsortpossible]

            ipixavg = np.average(ipixall[maskcoincidence])
            jpixavg = np.average(jpixall[maskcoincidence])
            labelsmany.append(">".join(labels[maskcoincidence][idxsort]))
            ipixmany.append(ipixavg)
            jpixmany.append(jpixavg)
            ntimes.append(len(dates))
            fluxmax.append(max(fluxes))

            if verbose:
                print "    Pixel %i, %i flux: %s" % (ipixavg, jpixavg, flux[maskcoincidence][idxsort])


            # plot images as array of differences
            # -----------------------------------

            imSNR = cands[maskcoincidence, ninfo: ninfo + npsf2]
            im1 = cands[maskcoincidence, ninfo + npsf2: ninfo + 2 * npsf2]
            im2 = cands[maskcoincidence, ninfo + 2 * npsf2: ninfo + 3 * npsf2]
            imt = cands[maskcoincidence, ninfo + 3 * npsf2: ninfo + 4 * npsf2]
            conv1st = conv1starray[maskcoincidence]

            if doplottimeseriesimages and (np.max(fluxes) > 0 or doplotnegatives):

                fig, ax = plt.subplots(ncols = np.sum(maskcoincidence), nrows = 3, sharex = True, sharey = True, figsize = (0.8 * np.sum(maskcoincidence), 2.4))
                # horizontal labels
                ax[1, 0].set_ylabel("im1", fontsize = 8, labelpad = 0)
                ax[0, 0].set_ylabel("im2", fontsize = 8, labelpad = 0)
                ax[2, 0].set_ylabel("im2 - im1", fontsize = 8, labelpad = 0)
                ax[0, 0].yaxis.label.set_color("white")
                ax[1, 0].yaxis.label.set_color("white")
                ax[2, 0].yaxis.label.set_color("white")
 
                # plot differences
                # ----------------
    
                for idiff in range(len(idxsort)):
    
                    lasttime = time.time()
                    idx = idxsort[idiff]                    
    
                    # vertical labels
                    ax[0, idiff].set_title("%4.2fd" % (dates[idiff] - MJDref), fontsize = 6, color = 'white')
    
                    ax[1, idiff].imshow(im1[idx].reshape((npsf, npsf)), interpolation = 'nearest', cmap = cmap)
                    ax[0, idiff].imshow(im2[idx].reshape((npsf, npsf)), interpolation = 'nearest', cmap = cmap)
                    if conv1st[idx]:
                        imdiff = im2[idx] - imt[idx]
                    else:
                        imdiff = imt[idx] - im1[idx]
                    ax[2, idiff].imshow(imdiff.reshape((npsf, npsf)), interpolation = 'nearest', cmap = cmap)
                    ax[0, idiff].set_xticks([])
                    ax[0, idiff].set_yticks([])
                    ax[1, idiff].set_xticks([])
                    ax[1, idiff].set_yticks([])
                    ax[2, idiff].set_xticks([])
                    ax[2, idiff].set_yticks([])
                    
                
                fig.subplots_adjust(wspace = 0, hspace = 0)
                plt.savefig("%s/%s/%s/CANDIDATES/timeseries_%s_%s_%04i-%04i_grid%02i_%s_images.png" % (webdir, field, CCD, field, CCD, ipixavg, jpixavg, fileref, resampling), pad_inches = 0.01, facecolor = 'black', bbox_inches = 'tight')
                print time.time() - lasttime



            # plot flux evolution
            # -------------------
            
            timesplot = np.concatenate(([0], dates - MJDref))
            fluxesplot = np.concatenate(([0], fluxes))
            e_fluxesplot = np.concatenate(([0], e_fluxes))
            (magsplot, e1_magsplot, e2_magsplot) = ADU2mag(fluxesplot, e_fluxesplot, CCD, exptime, airmass)
            probsplot = np.concatenate(([0], probabilities))

            timesplotpossible = np.concatenate(([0], datespossible - MJDref))
            fluxesplotpossible = np.concatenate(([0], fluxespossible))
            e_fluxesplotpossible = np.concatenate(([0], e_fluxespossible))
            (magsplotpossible, e1_magsplotpossible, e2_magsplotpossible) = ADU2mag(fluxesplotpossible, e_fluxesplotpossible, CCD, exptime, airmass)
            probsplotpossible = np.concatenate(([0], probabilitiespossible))

            ipixplot = np.concatenate(([0], ipixall[maskcoincidence][idxsort]))
            jpixplot = np.concatenate(([0], jpixall[maskcoincidence][idxsort]))
            labelsplot = np.concatenate(([''], labels[maskcoincidence][idxsort]))

            # save candidate for revisit
            # --------------------------
            
            fileout = "%s/%s/%s/CANDIDATES/timeseries_%s_%s_%04i-%04i_grid%02i_%s_summary.npy" % (sharedir, field, CCD, field, CCD, ipixavg, jpixavg, fileref, resampling)
            timeseriesstack = np.concatenate(([ipixavg, jpixavg, np.sum(maskcoincidence), np.sum(maskpossiblecoincidence)], timesplot + MJDref, fluxesplot, e_fluxesplot, magsplot, e1_magsplot, e2_magsplot, probsplot, labelsplot, ipixplot, jpixplot))
            if np.sum(maskpossiblecoincidence) > 0:
                timeseriesstack = np.hstack((timeseriesstack, timesplotpossible + MJDref, fluxesplotpossible, e_fluxesplotpossible, magsplotpossible, e1_magsplotpossible, e2_magsplotpossible, probsplotpossible))

            np.save(fileout, timeseriesstack)
            with open((fileout.replace(".npy", ".json")).replace(sharedir, "%s" % webdir), 'w') as jsonfile:
                jsonfile.write(json.dumps(timeseriesstack.tolist()))

            if doplottimeseriesflux:

                if np.max(fluxesplot) > 0:
                    fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (24, 4))
                    ax0 = ax[0]
                else:
                    fig, ax = plt.subplots(figsize = (12, 4))
                    ax0 = ax
                
                ax0.errorbar(timesplot, fluxesplot, yerr = e_fluxesplot, marker = '*', markersize = 20, elinewidth = 1, lw = 0, alpha = 0.7)
                ax0.errorbar(timesplotpossible, fluxesplotpossible, yerr = e_fluxesplotpossible, marker = '*', markersize = 5, elinewidth = 1, lw = 0, alpha = 0.7)
                ax0.axhline(y = 0, color = 'gray')
                ax0.axvline(x = 0, color = 'gray')
                ax0.set_ylabel("flux - flux_ref [ADU ref]")
                ax0.set_xlabel("MJD - MJD_ref [days]")
                
                plt.grid(True)
                
                if np.max(fluxesplot) > 0:
                    ax[1].errorbar(timesplot, magsplot, yerr = (e1_magsplot, e2_magsplot), marker = '*', markersize = 20, elinewidth = 1, lw = 0, alpha = 0.7)
                    ax[1].errorbar(timesplotpossible, magsplotpossible, yerr = (e1_magsplotpossible, e2_magsplotpossible), marker = '*', markersize = 5, elinewidth = 1, lw = 0, alpha = 0.7)
                    ax[1].axvline(x = 0, color = 'gray')
                    ax[1].set_ylabel("flux - flux_ref [mag %s]" % filtername)
                    ax[1].set_xlabel("MJD - MJD_ref [days]")
                    ax[1].set_ylim(min(26, np.max(magsplot[magsplot != 30]) + 1.5), min(magsplot) - 0.5)
                    
                plt.grid(True)

                plt.tight_layout()
                plt.savefig("%s/%s/%s/CANDIDATES/timeseries_%s_%s_%04i-%04i_grid%02i_%s_flux.png" % (webdir, field, CCD, field, CCD, ipixavg, jpixavg, fileref, resampling), bbox_inches = 'tight')

        else:
            continue


    printtime("finalsave")

    # save all features file as a pickle binary
    # -----------------------------------------
    
    fileout = "%s/%s/%s/CANDIDATES/allfeatures_%s_%s_grid%02i_%s_summary.pkl" % (sharedir, field, CCD, field, CCD, fileref, resampling)
    allfeatures = np.vstack([labels, repeated, probpair, features.transpose()]).transpose()
    print "    All features shape:", np.shape(allfeatures)
    pickle.dump(allfeatures, open(fileout, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)


    # RA DEC limits
    # -------------
    
    (RAmin, DECmin) = RADEC(0, 0, CD[0, 0, 0], CD[0, 0, 1], CD[0, 1, 0], CD[0, 1, 1], CRPIX[0, 0], CRPIX[0, 1], CRVAL[0, 0], CRVAL[0, 1], PV[0])
    (RAmax, DECmax) = RADEC(nx, ny, CD[0, 0, 0], CD[0, 0, 1], CD[0, 1, 0], CD[0, 1, 1], CRPIX[0, 0], CRPIX[0, 1], CRVAL[0, 0], CRVAL[0, 1], PV[0])
    (RAmin, DECmin) = applytransformation(ordersol, 15. * RAmin, DECmin, sol_astrometry_RADEC)
    (RAmax, DECmax) = applytransformation(ordersol, 15. * RAmax, DECmax, sol_astrometry_RADEC)
    (RAmin, RAmax) = (min(RAmin, RAmax) / 15., max(RAmin, RAmax) / 15.)
    (DECmin, DECmax) = (min(DECmin, DECmax), max(DECmin, DECmax))


    # compute variables to plot for candidates that are only once
    # -----------------------------------------------

    maskonce = np.array(selected & (repeated == 1) & np.invert(fake))
    labels1 = labels[maskonce]
    (jpix1, ipix1) = (jpixall[maskonce], ipixall[maskonce])
    (RA1, DEC1) = RADEC(jpix1, ipix1, CD[0, 0, 0], CD[0, 0, 1], CD[0, 1, 0], CD[0, 1, 1], CRPIX[0, 0], CRPIX[0, 1], CRVAL[0, 0], CRVAL[0, 1], PV[0])
    (RA1, DEC1) = applytransformation(ordersol, 15. * RA1, DEC1, sol_astrometry_RADEC)
    RA1 = np.array(RA1) / 15.
    DEC1 = np.array(DEC1)
    if verbose:
        print "    Number of candidates that appear just once:", len(RA1)
    

    # compute variables to plot for repeated candidates
    # -------------------------------------

    (RAmany, DECmany) = RADEC(jpixmany, ipixmany, CD[0, 0, 0], CD[0, 0, 1], CD[0, 1, 0], CD[0, 1, 1], CRPIX[0, 0], CRPIX[0, 1], CRVAL[0, 0], CRVAL[0, 1], PV[0])
    (RAmany, DECmany) = applytransformation(ordersol, 15. * RAmany, DECmany, sol_astrometry_RADEC)
    RAmany = np.array(RAmany) / 15.
    DECmany = np.array(DECmany)
    if verbose:
        print "    Number of candidates that appear more than once:", len(RAmany)
    ntimes = np.array(ntimes)
    fluxmax = np.array(fluxmax)


    # stack all candidates to do catalogue comparison
    # -----------------------------------------------
    
    RA1many = np.hstack((RA1, RAmany))
    DEC1many = np.hstack((DEC1, DECmany))
    ipix1many = np.hstack((ipix1, ipixmany))
    jpix1many = np.hstack((jpix1, jpixmany))
    labels1many = np.hstack((labels1, labelsmany))
    probpairmany = np.array(probpairmany)
    probpair1many = np.hstack((np.zeros(len(RA1)), probpairmany))

    # repeated candidates
    isrepeated = np.hstack((np.zeros(len(RA1), dtype = bool), np.ones(len(RAmany), dtype = bool)))
    if verbose:
        print "    Positive flux candidates:", np.sum(fluxmax > 0)
    positiveflux = np.hstack((np.zeros(len(RA1), dtype = bool), fluxmax > 0))


    if dohtml:

        # plot all candidate maps
        # -----------------------
    
        fig, ax = plt.subplots(figsize = (10, 5))  # this produces an image where the corners are in 73, 415 and 848, 15
    

        # plot candidates that appear many times
        # --------------------------------------
    
        ax.scatter(RAmany, DECmany, marker = 'o', alpha = 0.5, s = 400, c = 'g', edgecolors = 'none')
    

        # plot candidates that appear just once
        # -------------------------------------
    
        ax.scatter(RA1, DEC1, marker = 's', alpha = 0.5, s = 200, facecolor = 'none', c = 'k')
    
        for i in range(len(RA1)):
            if RA1[i] >= RAmin and RA1[i] <= RAmax and DEC1[i] >= DECmin and DEC1[i] <= DECmax:
                ax.text(RA1[i], DEC1[i], "%i %i" % (ipix1[i], jpix1[i]), fontsize = 6)
        for i in range(len(RAmany)):
            if RAmany[i] >= RAmin and RAmany[i] <= RAmax and DECmany[i] >= DECmin and DECmany[i] <= DECmax:
                ax.text(RAmany[i], DECmany[i], "%i %i (%i times)" % (ipixmany[i], jpixmany[i], ntimes[i]), fontsize = 6)

                
        # plot candidates that appear many times and that are positive
        # ------------------------------------------------------------
    
        ax.scatter(RA1many[positiveflux], DEC1many[positiveflux], marker = '*', s = 1700, c = 'yellow', alpha = 0.5)


    # plot with catalogues for comparison
    # -----------------------------------

    jsonname = "%s/%s/%s/CANDIDATES/timeseries_map_%s_%s_%02i.json" % (webdir, field, CCD, field, CCD, fileref)
    if dohtml:
        ijcoord = plotcatalogue(ax, jsonname, RA1many, DEC1many, ipix1many, jpix1many, isrepeated, positiveflux, probpair1many, labels1many, RAmin, RAmax, DECmin, DECmax, sharedir, field, CCD, True)
    else:
        print "ipix1many labels1many", len(ipix1many), len(labels1many)
        ijcoord = plotcatalogue(None, jsonname, RA1many, DEC1many, ipix1many, jpix1many, isrepeated, positiveflux, probpair1many, labels1many, RAmin, RAmax, DECmin, DECmax, sharedir, field, CCD, True)


    # if candidates were plotted, prepare webpage
    # -------------------------------------------
    
    if not (ijcoord is None):
        (icoord, jcoord, ngood) = ijcoord

        if dohtml:
            # create webpage with information
            doweb(webdir, field, CCD, "timeseries", (ngood > 0), np.sum(positiveflux) > 0, fileref, resampling, RA1many, DEC1many, ipix1many, jpix1many, icoord, jcoord, verbose)
        
    if dohtml:
        # nice ticks
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(HMS))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(HMS))
        
        ax.set_xlim(RAmax, RAmin)
        ax.set_ylim(DECmin, DECmax)
        ax.set_xlabel('RA [hr]')
        ax.set_ylabel('DEC [deg]')
        plt.grid(True)
        plt.savefig("%s/%s/%s/CANDIDATES/timeseries_map_%s_%s_%02i.png" % (webdir, field, CCD, field, CCD, fileref), pad_inches = 0.01, bbox_inches = 'tight')


# Revisit repeated candidates
# ###########################

printtime("dorevisit")

if dorevisit:
    
    if verbose:
        print "\n\nRevisiting difference files to find detailed candidate information...\n"

    showunsubtracted = True
    dolarge = True
    cmap = 'gray'

    # for large images
    if dolarge:
        nplot = 101
        nploth = nplot / 2

    # delete already created timeseries
    command = "rm -rf %s/%s/%s/CANDIDATES/*final*grid%02i*" % (sharedir, field, CCD, fileref)
    if verbose:
        print "    %s\n" % command
    os.system(command)
    command = "rm -rf %s/%s/%s/CANDIDATES/*final*grid%02i*" % (webdir, field, CCD, fileref)
    if verbose:
        print "    %s\n" % command
    os.system(command)
    command = "rm -rf %s/%s/%s/*final*grid%02i*" % (sharedir, field, CCD, fileref)
    if verbose:
        print "    %s\n" % command
    os.system(command)
    command = "rm -rf %s/%s/%s/*final*grid%02i*" % (webdir, field, CCD, fileref)
    if verbose:
        print "    %s\n" % command
    os.system(command)

    # load reference date
    fitsref = fits.open("%s/%s/%s/%s_%s_%02i_image_crblaster.fits" % (refdir, field, CCD, field, CCD, fileref))
    headerref = fitsref[0].header
    MJDref = float(headerref['MJD-OBS'])
    del fitsref
    try:
        exptime = float(headerref['EXPTIME'])
    except:
        print "\n\nWARNING: Cannot find field 'EXPTIME'\n\n"
        sys.exit(12)
    try:
        airmass = float(headerref['AIRMASS'])
    except:
        print "\n\nWARNING: Cannot find field 'AIRMASS'\n\n"
        sys.exit(12)

    # load data necessary for absolute coordinate transformation (select data with lowest rms)
    RADECmatch = "matchRADEC_%s_%s_(.*?)-%02i.npy$" % (field, CCD, fileref)
    calfiles = os.listdir("%s/%s/%s/CALIBRATIONS" % (sharedir, field, CCD))
    rms = 1e9
    for calfile in calfiles:
        if re.match("matchRADEC_%s_%s_(.*?)-%02i.npy$" % (field, CCD, fileref), calfile):
            matchRADEC = np.load("%s/%s/%s/CALIBRATIONS/%s" % (sharedir, field, CCD, calfile))
            if matchRADEC[8] < rms:
                (afluxADUB, e_afluxADUB, rmsdeg, CRVAL[0, 0], CRVAL[0, 1], CRPIX[0, 0], CRPIX[0, 1], CD[0, 0, 0], CD[0, 0, 1], CD[0, 1, 0], CD[0, 1, 1], nPV1, nPV2, ordersol) = matchRADEC[0:14]

                # unpack sol_astrometry_RADEC terms
                nend = 20
                if ordersol == 2:
                    nend = 26
                elif ordersol == 3:
                    nend = 34
                sol_astrometry_RADEC = matchRADEC[14: nend]

                # unpack PV terms
                PV[0] = matchRADEC[nend: nend + int(nPV1 * nPV2)].reshape((int(nPV1), int(nPV2)))

    # extreme values
    (RAmin, DECmin) = RADEC(0, 0, CD[0, 0, 0], CD[0, 0, 1], CD[0, 1, 0], CD[0, 1, 1], CRPIX[0, 0], CRPIX[0, 1], CRVAL[0, 0], CRVAL[0, 1], PV[0])
    (RAmax, DECmax) = RADEC(nx, ny, CD[0, 0, 0], CD[0, 0, 1], CD[0, 1, 0], CD[0, 1, 1], CRPIX[0, 0], CRPIX[0, 1], CRVAL[0, 0], CRVAL[0, 1], PV[0])
    (RAmin, DECmin) = applytransformation(ordersol, 15. * RAmin, DECmin, sol_astrometry_RADEC)
    (RAmax, DECmax) = applytransformation(ordersol, 15. * RAmax, DECmax, sol_astrometry_RADEC)
    (RAmin, RAmax) = (min(RAmin, RAmax) / 15., max(RAmin, RAmax) / 15.)
    (DECmin, DECmax) = (min(DECmin, DECmax), max(DECmin, DECmax))

    # try doing photometric calibration with Sloan
    # ------------------------------------------------

    # try loading SDSS catalogue
    if os.path.exists("%s/%s/%s/CALIBRATIONS/SDSS_%s_%s_%02i.npy" % (sharedir, field, CCD, field, CCD, fileref)):
        
        try:

            print "Doing SDSS magnitude calibration..."
            # load sextractor catalogue
            (xsex, ysex, zsex, e_zsex, rsex, fsex) = np.loadtxt("%s_image_crblaster.fits-catalogue%s.dat" % (file1cat, cataloguestring), usecols = (1, 2, 5, 6, 8, 9)).transpose()
            zorig = zsex
            e_zorig = e_zsex
            rorig = rsex
            forig = fsex
            (RAsex, DECsex) = RADEC(xsex, ysex, CD[0, 0, 0], CD[0, 0, 1], CD[0, 1, 0], CD[0, 1, 1], CRPIX[0, 0], CRPIX[0, 1], CRVAL[0, 0], CRVAL[0, 1], PV[0])
            (RAsex, DECsex) = applytransformation(ordersol, 15. * RAsex, DECsex, sol_astrometry_RADEC)
            RAsex = RAsex / 15.
            
            catdata = np.load("%s/%s/%s/CALIBRATIONS/SDSS_%s_%s_%02i.npy" % (sharedir, field, CCD, field, CCD, fileref))
            try:
                (SDSSRA, SDSSDEC, SDSS_ObjID, SDSS_psfMag_u, SDSS_psfMagErr_u, SDSS_psfMag_g, SDSS_psfMagErr_g, SDSS_psfMag_r, SDSS_psfMagErr_r, SDSS_psfMag_i, SDSS_psfMagErr_i, SDSS_psfMag_z, SDSS_psfMagErr_z) = catdata
            except:
                (SDSSRA, SDSSDEC, SDSS_ObjID, SDSS_psfMag_u, SDSS_psfMagErr_u, SDSS_psfMag_g, SDSS_psfMagErr_g, SDSS_psfMag_r, SDSS_psfMagErr_r) = catdata
            
            # select those stars that match one star in SDSS
            RAsexmin = min(RAsex)
            RAsexmax = max(RAsex)
            DECsexmin = min(DECsex)
            DECsexmax = max(DECsex)
            masksex = (RAsex > RAsexmin + 0.1 * (RAsexmax - RAsexmin)) & (RAsex < RAsexmax - 0.1 * (RAsexmax - RAsexmin)) & (DECsex > DECsexmin + 0.1 * (DECsexmax - DECsexmin)) & (DECsex < DECsexmax - 0.1 * (DECsexmax - DECsexmin)) & (fsex == 0) & (rsex < 4) & (zsex > 0) & (e_zsex > 0)
            RAsex = np.array(RAsex[masksex], dtype = float)
            DECsex = np.array(DECsex[masksex], dtype = float)
            zsex = np.array(zsex[masksex], dtype = float)
            e_zsex = np.array(e_zsex[masksex], dtype = float)
            rsex = np.array(rsex[masksex], dtype = float)
            fsex = np.array(fsex[masksex], dtype = float)
            
            maskSDSS = (SDSSRA > RAsexmin + 0.1 * (RAsexmax - RAsexmin)) & (SDSSRA < RAsexmax - 0.1 * (RAsexmax - RAsexmin)) & (SDSSDEC > DECsexmin + 0.1 * (DECsexmax - DECsexmin)) & (SDSSDEC < DECsexmax - 0.1 * (DECsexmax - DECsexmin))
            SDSSRA = np.array(SDSSRA[maskSDSS], dtype = float)
            SDSSDEC = np.array(SDSSDEC[maskSDSS], dtype = float)
            filter = filtername
            exec("zSDSS = np.array(SDSS_psfMag_%s[maskSDSS], dtype = float)" % (filter))
            exec("e_zSDSS = np.array(SDSS_psfMagErr_%s[maskSDSS], dtype = float)" % (filter))
            
            # find closest match
            idxSEX = []
            idxSDSS = []
            selectedSEX = np.zeros(len(RAsex), dtype = bool)
            selectedSDSS = np.zeros(len(SDSSRA), dtype = bool)
            for i in range(len(RAsex)):
                if selectedSEX[i]:
                    continue
                dist = np.sqrt((SDSSRA - RAsex[i])**2 + (SDSSDEC - DECsex[i])**2)
                idx = np.argmin(dist)
                if dist[idx] < (3. * rmsdeg) and not selectedSDSS[idx]:
                    selectedSEX[i] = True
                    selectedSDSS[idx] = True
                    idxSEX.append(i)
                    idxSDSS.append(idx)
            idxSEX = np.array(idxSEX)
            idxSDSS = np.array(idxSDSS)
        
            # apply indices
            RAsex = RAsex[idxSEX]
            DECsex = DECsex[idxSEX]
            SDSSRA = SDSSRA[idxSDSS]
            SDSSDEC = SDSSDEC[idxSDSS]
            zsex = zsex[idxSEX]
            e_zsex = e_zsex[idxSEX]
            zSDSS = zSDSS[idxSDSS]
            e_zSDSS = e_zSDSS[idxSDSS]
            
            testplot = True
            if testplot:
                fig, ax = plt.subplots()
                for i in range(len(RAsex)):
                    ax.plot([SDSSRA[i], RAsex[i]], [SDSSDEC[i], DECsex[i]], 'r')
                    ax.set_xlabel('RA')
                    ax.set_xlabel('DEC')
                    ax.set_title("%s %s Reference epoch" % (field, CCD), fontsize = 8)
                plt.savefig("%s/%s/%s/CALIBRATIONS/SDSS_RADEC_%s_%s_%02i.png" % (sharedir, field, CCD, field, CCD, fileref))
                    
            mask = (zsex > 0) & (zSDSS > 0)
            zsex = zsex[mask]
            e_zsex = e_zsex[mask]
            zSDSS = zSDSS[mask]
            e_zSDSS = e_zSDSS[mask]
                    
            # fit flux
            const = np.median(zSDSS + 2.5 * np.log10(zsex))
            
            fig, ax = plt.subplots()
            ax.errorbar(-2.5 * np.log10(zsex) + const, zSDSS, lw = 0, xerr = e_zsex / zsex, yerr = e_zSDSS / zSDSS, c = 'r', alpha = 0.5, elinewidth = 1)
            ax.plot([15, 26], [15, 26], 'gray')
            ax.set_xlabel("-2.5 log10 (ADU) + %4.2f" % const)
            ax.set_ylabel("mag %s SDSS" % filter)
            ax.set_xlim(15, 26)
            ax.set_title("%s %s Reference epoch" % (field, CCD), fontsize = 8)
            plt.savefig("%s/%s/%s/CALIBRATIONS/SDSS_flux_%s_%s_%02i.png" % (sharedir, field, CCD, field, CCD, fileref))
            
            fig, ax = plt.subplots()
            ax.hist(-2.5 * np.log10(zorig[(e_zorig > 0) & (zorig > 0) & (zorig > 3. * e_zorig) & (rorig < 5)]) + const, bins = 15)
            ax.set_xlabel("-2.5 log10 (ADU) + %4.2f" % const)
            ax.set_ylabel("N")
            ax.set_xlim(15, 26)
            ax.set_title("%s %s Reference epoch" % (field, CCD), fontsize = 8)
            plt.savefig("%s/%s/%s/CALIBRATIONS/SDSS_hist_%s_%s_%02i.png" % (sharedir, field, CCD, field, CCD, fileref))
            
        except:
            
            print "\n\nNo SDSS file or cannot do photometric calibration\n\n"
        
    # create empty data
    ipixavg = None
    jpixavg = None
    RAavg = None
    DECavg = None
    npoints = None
    posflux = None
    MJD = None
    flux = None
    e_flux = None
    fluxmax = None

    # look for candidates detected more than once and extract positions and fluxes to look
    seriesmatch = "timeseries_%s_%s_(.*?)_grid%02i_%s_summary.npy" % (field, CCD, fileref, resampling)
    seriesfiles = os.listdir("%s/%s/%s/CANDIDATES" % (sharedir, field, CCD))
    for seriesfile in seriesfiles:
        if re.search(seriesmatch, seriesfile):
            series = np.load("%s/%s/%s/CANDIDATES/%s" % (sharedir, field, CCD, seriesfile))
        else:
            continue
        
        ipix = series[0]
        jpix = series[1]
        (RAraw, DECraw) = RADEC(jpix, ipix, CD[0, 0, 0], CD[0, 0, 1], CD[0, 1, 0], CD[0, 1, 1], CRPIX[0, 0], CRPIX[0, 1], CRVAL[0, 0], CRVAL[0, 1], PV[0])
        (RA, DEC) = applytransformation(ordersol, 15. * RAraw, DECraw, sol_astrometry_RADEC)
        RA = RA / 15.

        # extract central pixel, date, fluxes
        if ipixavg is None:
            ipixavg = np.array([ipix])
            jpixavg = np.array([jpix])
            RAavg = np.array([RA])
            DECavg = np.array([DEC])
            npoints = np.array([series[2]])
            npt = np.array([series[2] + 1], dtype = int)
            posflux = np.array([0])
            MJD = np.array(series[3: 3 + npt[0]])
            flux = np.array(series[3 + npt[0]: 3 + 2 * npt[0]])
            e_flux = np.array(series[3 + 2 * npt[0]: 3 + 3 * npt[0]])
            fluxmax = max(flux)
            mags = np.array(series[3 + 3 * npt[0]: 3 + 4 * npt[0]])
            e1_mags = np.array(series[3 + 4 * npt[0]: 3 + 5 * npt[0]])
            e2_mags = np.array(series[3 + 5 * npt[0]: 3 + 6 * npt[0]])
        else:
            ipixavg = np.hstack((ipixavg, ipix))
            jpixavg = np.hstack((jpixavg, jpix))
            npoints = np.hstack((npoints, series[2]))
            npt = np.array(np.hstack((npt, series[2] + 1)), dtype = int)
            RAavg = np.hstack((RAavg, RA))
            DECavg = np.hstack((DECavg, DEC))
            posflux = np.hstack((posflux, np.shape(flux)[0]))
            MJD = np.hstack((MJD, series[3: 3 + npt[-1]]))
            flux = np.hstack([flux, series[3 + npt[-1]: 3 + 2 * npt[-1]]])
            e_flux = np.hstack([e_flux, series[3 + 2 * npt[-1]: 3 + 3 * npt[-1]]])
            fluxmax = np.hstack([fluxmax, max(series[3 + npt[-1]: 3 + 2 * npt[-1]])])
            mags = np.hstack([mags, series[3 + 3 * npt[-1]: 3 + 4 * npt[-1]]])
            e1_mags = np.hstack([e1_mags, series[3 + 4 * npt[-1]: 3 + 5 * npt[-1]]])
            e2_mags = np.hstack([e2_mags, series[3 + 5 * npt[-1]: 3 + 6 * npt[-1]]])
            
        # print pixel
        if verbose:
            print "    Pixel %i, %i (%i repetitions)" % (ipixavg[-1], jpixavg[-1], npoints[-1])

    # exit revisit if no candidates to visit
    if not (RAavg is None):

        # numpify RA DEC
        RAavg = np.array(RAavg)
        DECavg = np.array(DECavg)
            
        # open fits files and save relevant information (opening each fits file only once!)
        # --------------------------------------------------------------------------------
    
        # search for difference files and save data
        diffmatch = "Diff_%s_%s_(.*?)_grid%02i_%s.fits" % (field, CCD, fileref, resampling)
        difffiles = os.listdir("%s/%s/%s" % (indir, field, CCD))
    
        MJDnew = []
        diffstring = []
        imdiff = None
        invVAR = None
        if dolarge:
            imdifflarge = None
            invVARlarge = None
            
        if showunsubtracted:
            im2 = None
            if dolarge:
                im2large = None
    
        # first image
        if showunsubtracted:
    
            if verbose:
                print "\n    Opening reference file..."
    
            fits1data = fits.open("%s/%s/%s/%s_%s_%02i_image_crblaster.fits" % (refdir, field, CCD, field, CCD, fileref))[0].data
            print useweights
            if useweights == 'external':
                fits1dataorig = fits.open("%s/%s/%s/%s_%s_%02i_image.fits.fz" % (refdir, field, CCD, field, CCD, fileref))[0].data
            elif useweights == 'internal':
                fits1dataorig = fits.open("%s/%s/%s/%s_%s_%02i_image.fits" % (refdir, field, CCD, field, CCD, fileref))[0].data

            for it in range(len(ipixavg)):
                if dolarge:
                    imlarge = fits1data[int(max(0, ipixavg[it] - nploth)): int(min(ny, ipixavg[it] + nploth + 1)), int(max(0, jpixavg[it] - nploth)): int(min(nx, jpixavg[it] + nploth + 1))]
                    imlargeorig = fits1dataorig[int(max(0, ipixavg[it] - nploth)): int(min(ny, ipixavg[it] + nploth + 1)), int(max(0, jpixavg[it] - nploth)): int(min(nx, jpixavg[it] + nploth + 1))]
                if it == 0:
                    im1 = fits1data[int(ipixavg[it] - npsfh): int(ipixavg[it] + npsfh + 1), int(jpixavg[it] - npsfh): int(jpixavg[it] + npsfh + 1)].flatten()
                    im1orig = fits1dataorig[int(ipixavg[it] - npsfh): int(ipixavg[it] + npsfh + 1), int(jpixavg[it] - npsfh): int(jpixavg[it] + npsfh + 1)].flatten()
                    print "---------->", ipixavg[it], jpixavg[it], np.sum((im1[rs2D < 5] - im1orig[rs2D < 5])**2)
                    if dolarge:
                        pos1large = np.array([0])
                        im1shapes = np.shape(imlarge)
                        im1large = imlarge.flatten()
                        im1largeorig = imlargeorig.flatten()
                else:
                    im1last = fits1data[int(ipixavg[it] - npsfh): int(ipixavg[it] + npsfh + 1), int(jpixavg[it] - npsfh): int(jpixavg[it] + npsfh + 1)].flatten()
                    im1 = np.hstack((im1, im1last))
                    im1origlast = fits1dataorig[int(ipixavg[it] - npsfh): int(ipixavg[it] + npsfh + 1), int(jpixavg[it] - npsfh): int(jpixavg[it] + npsfh + 1)].flatten()
                    im1orig = np.hstack((im1orig, im1origlast))
                    print np.shape(im1)
                    print "---------->", ipixavg[it], jpixavg[it], np.sum((im1last[rs2D < 5] - im1origlast[rs2D < 5])**2)
                    if dolarge:
                        pos1large = np.hstack((pos1large, np.shape(im1large)[0]))
                        im1shapes = np.vstack((im1shapes, np.shape(imlarge)))
                        im1large = np.hstack((im1large, imlarge.flatten()))
                        im1largeorig = np.hstack((im1largeorig, imlargeorig.flatten()))
            del fits1data
            del fits1dataorig
    
        if verbose:
            print "    Opening processed files and saving images..."
    
        # loop among useful files
        for difffile in difffiles:
            if re.search(diffmatch, difffile):
                diff = fits.open("%s/%s/%s/%s" % (indir, field, CCD, difffile))
            else:
                continue
            
            # append difference string and MJD
            diffstring.append(re.findall(diffmatch, difffile)[0])
            MJDnew.append(float(diff[0].header['MJD-OBS']))
    
            # extract difference pixels
            for it in range(len(ipixavg)):
                if dolarge:
                    imlarge = diff[0].data[int(max(0, ipixavg[it] - nploth)): int(min(ny, ipixavg[it] + nploth + 1)), int(max(0, jpixavg[it] - nploth)): int(min(nx, jpixavg[it] + nploth + 1))]
                if imdiff is None:
                    imdiff = diff[0].data[int(ipixavg[it] - npsfh): int(ipixavg[it] + npsfh + 1), int(jpixavg[it] - npsfh): int(jpixavg[it] + npsfh + 1)].flatten()
                    if dolarge:
                        poslarge = np.array([0])
                        diffshapes = np.shape(imlarge)
                        imdifflarge = imlarge.flatten()
                else:
                    imdiff = np.hstack([imdiff, diff[0].data[int(ipixavg[it] - npsfh): int(ipixavg[it] + npsfh + 1), int(jpixavg[it] - npsfh): int(jpixavg[it] + npsfh + 1)].flatten()])
                    if dolarge:
                        poslarge = np.hstack((poslarge, np.shape(imdifflarge)[0]))
                        diffshapes = np.vstack((diffshapes, np.shape(imlarge)))
                        imdifflarge = np.hstack((imdifflarge, imlarge.flatten()))
            del diff
    
            # extract invVAR pixels
            invVARfits = fits.open("%s/%s/%s/%s" % (indir, field, CCD, difffile.replace("Diff", "invVAR")))
                        
            for it in range(len(ipixavg)):
                if dolarge:
                    imlarge = invVARfits[0].data[int(max(0, ipixavg[it] - nploth)): int(min(ny, ipixavg[it] + nploth + 1)), int(max(0, jpixavg[it] - nploth)): int(min(nx, jpixavg[it] + nploth + 1))]
                if invVAR is None:
                    invVAR = invVARfits[0].data[int(ipixavg[it] - npsfh): int(ipixavg[it] + npsfh + 1), int(jpixavg[it] - npsfh): int(jpixavg[it] + npsfh + 1)].flatten()
                    if dolarge:
                        invVARlarge = imlarge.flatten()
                else:
                    invVAR = np.hstack((invVAR, invVARfits[0].data[int(ipixavg[it] - npsfh): int(ipixavg[it] + npsfh + 1), int(jpixavg[it] - npsfh): int(jpixavg[it] + npsfh + 1)].flatten()))
                    if dolarge:
                        invVARlarge = np.hstack((invVARlarge, imlarge.flatten()))
            del invVARfits
    
            # science image
            if showunsubtracted:
    
                fileidx = re.findall(diffmatch, difffile)[0][0:2]
                fits2 = fits.open("%s/%s/%s/%s_%s_%s_image_crblaster_grid%02i_%s.fits" % (indir, field, CCD, field, CCD, fileidx, fileref, resampling))
    
                for it in range(len(ipixavg)):
                    if dolarge:
                        imlarge = fits2[0].data[int(max(0, ipixavg[it] - nploth)): int(min(ny, ipixavg[it] + nploth + 1)), int(max(0, jpixavg[it] - nploth)): int(min(nx, jpixavg[it] + nploth + 1))]
                    if im2 is None:
                        im2 = fits2[0].data[int(ipixavg[it] - npsfh): int(ipixavg[it] + npsfh + 1), int(jpixavg[it] - npsfh): int(jpixavg[it] + npsfh + 1)].flatten()
                        if dolarge:
                            im2large = imlarge.flatten()
                    else:
                        im2 = np.hstack((im2, fits2[0].data[int(ipixavg[it] - npsfh): int(ipixavg[it] + npsfh + 1), int(jpixavg[it] - npsfh): int(jpixavg[it] + npsfh + 1)].flatten()))
                        if dolarge:
                            im2large = np.hstack((im2large, imlarge.flatten()))
                del fits2
    
        # loop among candidates and plot relevant information
        print "len(ipixavg)", len(ipixavg)
        for it in range(len(ipixavg)):
    
            if verbose:
                print "    Recomputing fluxes and plotting candidate in pixel %i, %i..." % (ipixavg[it], jpixavg[it])
    
    
            if it == 0:
                # numpify
                MJDnew = np.array(MJDnew)
                diffstring = np.array(diffstring)
    
                # sort data by date
                idxsort = np.argsort(MJDnew)
                MJDnew = MJDnew[idxsort]
                diffstring = diffstring[idxsort]
    
            # extract images corresponding to given transient candidate
            imdiffcand = None
            for iMJD in range(len(MJDnew)):
                
                idx = iMJD * len(ipixavg) + it
                idx0 = idx * npsf * npsf
                idx1 = (idx + 1) * npsf * npsf
                if dolarge:
                    idx0large = poslarge[idx]
                    idx1large = idx0large + diffshapes[idx][0] * diffshapes[idx][1]
                           
                if imdiffcand is None:
                    imdiffcand = imdiff[idx0: idx1]
                    invVARcand = invVAR[idx0: idx1]
                    if showunsubtracted:
                        im2cand = im2[idx0: idx1]
                    if dolarge:
                        imdifflargecand = imdifflarge[idx0large: idx1large].reshape(diffshapes[idx])
                        invVARlargecand = invVARlarge[idx0large: idx1large].reshape(diffshapes[idx])
                        if showunsubtracted:
                            im2largecand = im2large[idx0large: idx1large].reshape(diffshapes[idx])
                else:
                    imdiffcand = np.dstack((imdiffcand, imdiff[idx0: idx1]))
                    invVARcand = np.dstack((invVARcand, invVAR[idx0: idx1]))
                    if showunsubtracted:
                        im2cand = np.dstack((im2cand, im2[idx0: idx1]))
                    if dolarge:
                        imdifflargecand = np.dstack((imdifflargecand, imdifflarge[idx0large: idx1large].reshape(diffshapes[idx])))
                        invVARlargecand = np.dstack((invVARlargecand, invVARlarge[idx0large: idx1large].reshape(diffshapes[idx])))
                        if showunsubtracted:
                            im2largecand = np.dstack((im2largecand, im2large[idx0large: idx1large].reshape(diffshapes[idx])))
    
            
            # reference image
            if showunsubtracted:
                im1cand = im1[it * npsf * npsf: (it + 1) * npsf * npsf]
                im1candorig = im1orig[it * npsf * npsf: (it + 1) * npsf * npsf]
                if dolarge:
                    idx0large = pos1large[it]
                    if len(ipixavg) > 1:
                        idx1large = idx0large + im1shapes[it][0] * im1shapes[it][1]
                        im1largecand = im1large[idx0large: idx1large].reshape(im1shapes[it])
                        im1largecandorig = im1largeorig[idx0large: idx1large].reshape(im1shapes[it])
                    else:
                        idx1large = idx0large + im1shapes[0] * im1shapes[1]
                        im1largecand = im1large[idx0large: idx1large].reshape(im1shapes)
                        im1largecandorig = im1largeorig[idx0large: idx1large].reshape(im1shapes)
    
                # plot reference image
                fig, ax = plt.subplots(ncols = 3, sharex = True, sharey = True, figsize = (3.5, 1.))
                ax[0].set_ylabel("im1orig", fontsize = 8, labelpad = 0)
                ax[1].set_ylabel("im1", fontsize = 8, labelpad = 0)
                ax[2].set_ylabel("im1orig - im1", fontsize = 8, labelpad = 0)
                ax[0].yaxis.label.set_color("white")
                ax[1].yaxis.label.set_color("white")
                ax[2].yaxis.label.set_color("white")
                ax[0].set_xticks([])
                ax[0].set_yticks([])
                ax[1].set_xticks([])
                ax[1].set_yticks([])
                ax[2].set_xticks([])
                ax[2].set_yticks([])
                ax[0].imshow(im1candorig.reshape(npsf, npsf), interpolation = 'nearest', cmap = cmap)
                ax[1].imshow(im1cand.reshape(npsf, npsf), interpolation = 'nearest', cmap = cmap)
                ax[2].imshow(im1candorig.reshape(npsf, npsf) - im1cand.reshape(npsf, npsf), interpolation = 'nearest', cmap = cmap)
                plt.savefig("%s/%s/%s/CANDIDATES/final_%s_%s_%04i-%04i_grid%02i_%s_orig_images.png" % (webdir, field, CCD, field, CCD, ipixavg[it], jpixavg[it], fileref, resampling), pad_inches = 0.01, facecolor = 'black', bbox_inches = 'tight')

                # plot large reference image
                fig, ax = plt.subplots(ncols = 3, sharex = True, sharey = True, figsize = (3.5, 1.))
                ax[0].set_ylabel("im1orig", fontsize = 8, labelpad = 0)
                ax[1].set_ylabel("im1", fontsize = 8, labelpad = 0)
                ax[2].set_ylabel("im1orig - im1", fontsize = 8, labelpad = 0)
                ax[0].yaxis.label.set_color("white")
                ax[1].yaxis.label.set_color("white")
                ax[2].yaxis.label.set_color("white")
                ax[0].set_xticks([])
                ax[0].set_yticks([])
                ax[1].set_xticks([])
                ax[1].set_yticks([])
                ax[2].set_xticks([])
                ax[2].set_yticks([])
                ax[0].imshow(im1largecandorig, interpolation = 'nearest', cmap = cmap)
                ax[1].imshow(im1largecand, interpolation = 'nearest', cmap = cmap)
                ax[2].imshow(im1largecandorig - im1largecand, interpolation = 'nearest', cmap = cmap)
                plt.savefig("%s/%s/%s/CANDIDATES/final_%s_%s_%04i-%04i_grid%02i_%s_orig_images_large.png" % (webdir, field, CCD, field, CCD, ipixavg[it], jpixavg[it], fileref, resampling), pad_inches = 0.01, facecolor = 'black', bbox_inches = 'tight')

            # sort images chronologically
            imdiffcand = imdiffcand[0][:, idxsort]
            invVARcand = invVARcand[0][:, idxsort]
            if showunsubtracted:
                im2cand = im2cand[0][:, idxsort]
            if dolarge:
                imdifflargecand = imdifflargecand[:, :, idxsort]
                invVARlargecand = invVARlargecand[:, :, idxsort]
                if showunsubtracted:
                    im2largecand = im2largecand[:, :, idxsort]
    
            # new fluxes to compute
            fluxnew = np.zeros(len(MJDnew))
            e_fluxnew = np.zeros(len(MJDnew))
    
            # start plot of candidates images and loop among all available files
            if showunsubtracted:
                fig, ax = plt.subplots(nrows = 4, ncols = len(MJDnew), sharex = True, sharey = True, figsize = (1. * len(MJDnew), 4.))
                i0 = 2
            else:
                fig, ax = plt.subplots(nrows = 2, ncols = len(MJDnew), sharex = True, sharey = True, figsize = (1. * len(MJDnew), 2.))
                i0 = 0
    
            # horizontal labels
            ax[i0, 0].set_ylabel("im2 - im1", fontsize = 8, labelpad = 0)
            ax[i0 + 1, 0].set_ylabel("inv var.", fontsize = 8, labelpad = 0)
            ax[i0, 0].yaxis.label.set_color("white")
            ax[i0 + 1, 0].yaxis.label.set_color("white")
            if showunsubtracted:
                ax[1, 0].set_ylabel("im1", fontsize = 8, labelpad = 0)
                ax[0, 0].set_ylabel("im2", fontsize = 8, labelpad = 0)
                ax[0, 0].yaxis.label.set_color("white")
                ax[1, 0].yaxis.label.set_color("white")
    
            # compute fluxes and plot
            for iMJD in range(len(MJDnew)):
                
                print "---->", iMJD, diffstring[iMJD]

                # load psf to do consistent photometry
                try:
                    psfref = np.load("%s/%s/%s/CALIBRATIONS/psf_%s_%s_%s_grid%02i_%s.npy" % (sharedir, field, CCD, field, CCD, diffstring[iMJD], fileref, resampling))
                    psfrefflat = psfref.flatten()
                    print np.sum(psfrefflat)
    
                    # mask values below fcut from maximum
                    psfrefflat[psfrefflat < np.max(psfrefflat) * fcut] = 0

                except:
                    print "\n\nWARNING: Cannot find psf file %s/%s/%s/CALIBRATIONS/psf_%s_%s_%s_grid%02i_%s.npy" % (sharedir, field, CCD, field, CCD, diffstring[iMJD], fileref, resampling)
                    sys.exit(26)
                        

                    # do optimal photometry with given psf
                aux = psfrefflat * invVARcand[:, iMJD]
                fluxnew[iMJD] = np.sum(aux * imdiffcand[:, iMJD])
                e_fluxnew[iMJD] = np.sum(aux * psfrefflat)
                fluxnew[iMJD] = fluxnew[iMJD] / e_fluxnew[iMJD] # flux
                e_fluxnew[iMJD] = np.sqrt(1. / e_fluxnew[iMJD]) # sqrt(variance)
    
                # find scaling factors
                astrocal = np.load("%s/%s/%s/CALIBRATIONS/match_%s_%s_%s.npy" % (sharedir, field, CCD, field, CCD, diffstring[iMJD].replace("t", "")))
                aflux = astrocal[0]
                if diffstring[iMJD][2] != 't':
                    fluxnew[iMJD] = fluxnew[iMJD] / aflux
                    e_fluxnew[iMJD] = e_fluxnew[iMJD] / aflux
               
                # plot difference and inverse variance images
                ax[i0, iMJD].imshow(imdiffcand[:, iMJD].reshape(npsf, npsf), interpolation = 'nearest', cmap = cmap)
                ax[i0 + 1, iMJD].imshow(invVARcand[:, iMJD].reshape(npsf, npsf), interpolation = 'nearest', cmap = cmap)
                if showunsubtracted:
                    ax[1, iMJD].imshow(im1cand.reshape(npsf, npsf), interpolation = 'nearest', cmap = cmap)
                    ax[0, iMJD].imshow(im2cand[:, iMJD].reshape(npsf, npsf), interpolation = 'nearest', cmap = cmap)
    
                ax[0, iMJD].set_title("%4.2fd" % (MJDnew[iMJD] - MJDref), fontsize = 6, color = 'white')
    
                ax[0, iMJD].set_xticks([])
                ax[0, iMJD].set_yticks([])
                ax[1, iMJD].set_xticks([])
                ax[1, iMJD].set_yticks([])
                if showunsubtracted:
                    ax[2, iMJD].set_xticks([])
                    ax[2, iMJD].set_yticks([])
                    ax[3, iMJD].set_xticks([])
                    ax[3, iMJD].set_yticks([])
    
            # save images plot
            plt.subplots_adjust(wspace = 0, hspace = 0)
            plt.savefig("%s/%s/%s/CANDIDATES/final_%s_%s_%04i-%04i_grid%02i_%s_images.png" % (webdir, field, CCD, field, CCD, ipixavg[it], jpixavg[it], fileref, resampling), pad_inches = 0.01, facecolor = 'black', bbox_inches = 'tight')
    
            # start plot of candidates large images and loop among all available files
            if dolarge:
    
                # start plot of candidates images and loop among all available files
                if showunsubtracted:
                    fig, ax = plt.subplots(nrows = 4, ncols = len(MJDnew), sharex = True, sharey = True, figsize = (1. * len(MJDnew), 4.))
                    i0 = 2
                else:
                    fig, ax = plt.subplots(nrows = 2, ncols = len(MJDnew), sharex = True, sharey = True, figsize = (1. * len(MJDnew), 2.))
                    i0 = 0
                    
                ax[i0, 0].set_ylabel("im2 - im1", fontsize = 8, labelpad = 0)
                ax[i0 + 1, 0].set_ylabel("inv. var.", fontsize = 8, labelpad = 0)
                ax[i0, 0].yaxis.label.set_color("white")
                ax[i0 + 1, 0].yaxis.label.set_color("white")
                if showunsubtracted:
                    ax[1, 0].set_ylabel("im1", fontsize = 8, labelpad = 0)
                    ax[0, 0].set_ylabel("im2", fontsize = 8, labelpad = 0)
                    ax[0, 0].yaxis.label.set_color("white")
                    ax[1, 0].yaxis.label.set_color("white")
                
                for iMJD in range(len(MJDnew)):
                    
                    # plot difference and inverse variance images
                    ax[i0, iMJD].imshow(imdifflargecand[:, :, iMJD], interpolation = 'nearest', cmap = cmap)
                    ax[i0 + 1, iMJD].imshow(invVARlargecand[:, :, iMJD], interpolation = 'nearest', cmap = cmap)
                    if showunsubtracted:
                        ax[1, iMJD].imshow(im1largecand, interpolation = 'nearest', cmap = cmap)
                        ax[0, iMJD].imshow(im2largecand[:, :, iMJD], interpolation = 'nearest', cmap = cmap)
                    
                    ax[0, iMJD].set_title("%4.2fd" % (MJDnew[iMJD] - MJDref), fontsize = 6, color = 'white')
    
                    ax[0, iMJD].set_xticks([])
                    ax[0, iMJD].set_yticks([])
                    ax[1, iMJD].set_xticks([])
                    ax[1, iMJD].set_yticks([])
                    if showunsubtracted:
                        ax[2, iMJD].set_xticks([])
                        ax[2, iMJD].set_yticks([])
                        ax[3, iMJD].set_xticks([])
                        ax[3, iMJD].set_yticks([])
                        
                # save large images plot
                plt.subplots_adjust(wspace = 0, hspace = 0)
                plt.savefig("%s/%s/%s/CANDIDATES/final_%s_%s_%04i-%04i_grid%02i_%s_images_large.png" % (webdir, field, CCD, field, CCD, ipixavg[it], jpixavg[it], fileref, resampling), pad_inches = 0.01, facecolor = 'black', bbox_inches = 'tight')
    
            # save new fluxes
            fileout = "%s/%s/%s/CANDIDATES/final_%s_%s_%04i-%04i_grid%02i_%s.npy" % (sharedir, field, CCD, field, CCD, ipixavg[it], jpixavg[it], fileref, resampling)
            np.save(fileout, np.vstack([MJDnew, fluxnew, e_fluxnew]))
    
            # compute Lomb Scargle periodogram and best folded light curve
            times = MJDnew - MJDref
            fluxes = fluxnew
            e_fluxes = e_fluxnew
            times = np.concatenate(([0], times))
            fluxes = np.concatenate(([0], fluxes))
            e_fluxes = np.concatenate(([0], e_fluxes))
            (fx, fy, nout, jmax, prob) = lomb.fasper(times, fluxes, 20., 1.5)
    
            bestperiod = 1. / fx[jmax]
    
            timesfold = np.mod(times, bestperiod)
            idx = np.argsort(timesfold)
            timescycle = timesfold[idx]
            fluxcycle = fluxes[idx]
            e_fluxcycle = e_fluxes[idx]
    
            timesfold = timescycle
            fluxesfold = fluxcycle
            e_fluxesfold = e_fluxcycle
            nperiod = 1.
            while max(timesfold) < max(times):
                timesfold = np.concatenate([timesfold, nperiod * 1. / fx[jmax] + timescycle])
                fluxesfold = np.concatenate([fluxesfold, fluxcycle])
                e_fluxesfold = np.concatenate([e_fluxesfold, e_fluxcycle])
                nperiod += 1
            maskplot = timesfold <= max(times)

            # SNR masks
            SNRnew = np.abs(fluxes / e_fluxes)
            print "SNRnew:", SNRnew
            SNRlt3 = np.array(SNRnew < 3, dtype = bool)
            SNR3to5 = np.array((SNRnew >= 3) & (SNRnew < 5), dtype = bool)
            SNRgt5 = np.array(SNRnew >= 5, dtype = bool)

            # compute magnitudes
            (magsplot, e1_magsplot, e2_magsplot) = ADU2mag(fluxes, e_fluxes, CCD, exptime, airmass)
            print "fluxes:", fluxes
            print "e_fluxes:", e_fluxes
            print "magsplot:", magsplot
            print "e1_magsplot:", e1_magsplot
            print "e2_magsplot:", e2_magsplot
            
            # save new fluxes
            print "Saving revisit fluxes"

            fileout = "%s/%s/%s/CANDIDATES/final_%s_%s_%04i-%04i_grid%02i_%s_summary.npy" % (sharedir, field, CCD, field, CCD, ipixavg[it], jpixavg[it], fileref, resampling)
            finalstack = np.vstack([times + MJDref, fluxes, e_fluxes, magsplot, e1_magsplot, e2_magsplot])
            np.save(fileout, finalstack)
            with open((fileout.replace(".npy", ".json")).replace(sharedir, webdir), 'w') as jsonfile:
                jsonfile.write(json.dumps(finalstack.tolist()))

            # plot fluxes

            if doplottimeseriesflux:

                print "Plotting forced photometry flux"

                if np.max(fluxes) > 0:
                    fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (24, 4))
                    ax0 = ax[0]
                else:
                    fig, ax = plt.subplots(figsize = (12, 4))
                    ax0 = ax

                # plot original, revisited and folded light curves
                ax0.set_xlabel("MJD - MJD_ref [days]")
                ax0.set_ylabel("flux - flux_ref [ADU ref]")
                
                # folded fluxes
                if bestperiod < 2. and prob < 0.3:# and not (bestperiod > 0.3 and bestperiod < 0.366):
                    ax0.errorbar(timesfold[maskplot], fluxesfold[maskplot], yerr = e_fluxesfold[maskplot], c = 'gray', alpha = 0.5, ls = ':')
                    ax0.set_title("Best period: %5.3f days (p-value: %5.2e)" % (bestperiod, prob), fontsize = 8)
                    np.save("%s/%s/%s/CANDIDATES/Period_%s_%s_%04i-%04i_grid%02i_%s.npy" % (sharedir, field, CCD, field, CCD, ipixavg[it], jpixavg[it], fileref, resampling), np.concatenate([np.array([bestperiod, prob, len(timescycle)]), timescycle, fluxcycle]))
                
                # new fluxes
                if np.sum(SNRlt3) > 0:
                    ax0.errorbar(times[SNRlt3], fluxes[SNRlt3], yerr = e_fluxes[SNRlt3], c = 'r', lw = 0, elinewidth = 1, marker = 'o', alpha = 0.3, label = "SNR < 3")
                if np.sum(SNR3to5) > 0:
                    ax0.errorbar(times[SNR3to5], fluxes[SNR3to5], yerr = e_fluxes[SNR3to5], c = 'orange', lw = 0, elinewidth = 1, marker = 'o', alpha = 0.5, label = "3 <= SNR < 5")
                if np.sum(SNRgt5) > 0:
                    ax0.errorbar(times[SNRgt5], fluxes[SNRgt5], yerr = e_fluxes[SNRgt5], c = 'b', lw = 0, elinewidth = 1, marker = 'o', alpha = 0.7, label = "SNR >= 5")

                # detection fluxes
                ax0.errorbar(MJD[posflux[it]: posflux[it] + npt[it]] - MJDref, flux[posflux[it]: posflux[it] + npt[it]], \
                                          yerr = e_flux[posflux[it]: posflux[it] + npt[it]], \
                                          marker = '*', markersize = 20, lw = 0, elinewidth = 1, c = 'b', alpha = 0.3)
                ax0.axhline(y = 0, color = 'gray')
                ax0.axvline(x = 0, color = 'gray')

                plt.legend(fancybox = False, prop = {'size':8}, loc = 2)
                plt.tight_layout()
                plt.grid(True)

                # magnitudes
                if np.max(fluxes) > 0:
                    if verbose:
                        print "Plotting flux evolution in magnitudes"

                    if np.sum(SNRlt3) > 0:
                        print "\n lt3", times[SNRlt3], magsplot[SNRlt3], e1_magsplot[SNRlt3], e2_magsplot[SNRlt3]
                        ax[1].errorbar(times[SNRlt3], magsplot[SNRlt3], yerr = (e1_magsplot[SNRlt3]+1e-3, e2_magsplot[SNRlt3]+1e-3), c = 'r', lw = 0, elinewidth = 1, marker = 'o', alpha = 0.3, label = "SNR < 3")
                    if np.sum(SNR3to5) > 0:
                        print "\n 3to5", times[SNR3to5], magsplot[SNR3to5], e1_magsplot[SNR3to5], e2_magsplot[SNR3to5]
                        ax[1].errorbar(times[SNR3to5], magsplot[SNR3to5], yerr = (e1_magsplot[SNR3to5]+1e-3, e2_magsplot[SNR3to5]+1e-3), c = 'orange', lw = 0, elinewidth = 1, marker = 'o', alpha = 0.5, label = "3 <= SNR < 5")
                    if np.sum(SNRgt5) > 0:
                        print "\n gt5", times[SNRgt5], magsplot[SNRgt5], e1_magsplot[SNRgt5], e2_magsplot[SNRgt5]
                        # new magnitudes
                        ax[1].errorbar(times[SNRgt5], magsplot[SNRgt5], yerr = (e1_magsplot[SNRgt5]+1e-3, e2_magsplot[SNRgt5]+1e-3), c = 'b', lw = 0, elinewidth = 1, marker = 'o', alpha = 0.7, label = "SNR >= 5")
                        # detection magnitudes
                        ax[1].errorbar(MJD[posflux[it]: posflux[it] + npt[it]] - MJDref, mags[posflux[it]: posflux[it] + npt[it]], \
                                   yerr = (e1_mags[posflux[it]: posflux[it] + npt[it]], e2_mags[posflux[it]: posflux[it] + npt[it]]), \
                                   c = 'b', lw = 0, elinewidth = 1, marker = '*', markersize = 20, alpha = 0.3)

                    ax[1].axvline(x = 0, color = 'gray')
                    ax[1].set_xlabel("MJD - MJD_ref [days]")
                    ax[1].set_ylabel("mag %s" % filtername)
                    ax[1].set_ylim(min(26, np.max(magsplot[magsplot != 30]) + 1.5), min(magsplot) - 0.5)
            
                    plt.legend(fancybox = False, prop = {'size':8}, loc = 2)
                    plt.savefig("%s/%s/%s/CANDIDATES/final_%s_%s_%04i-%04i_grid%02i_%s_flux.png" % (webdir, field, CCD, field, CCD, ipixavg[it], jpixavg[it], fileref, resampling), bbox_inches = 'tight')
            
            if doplotperiodogram:
                # plot periodogram
                fig, ax = plt.subplots(figsize = (12, 4))
                ax.plot(fx, fy)
                ax.set_xlabel("Frequency [1 / days]")
                ax.set_ylabel("Power")
                ax.set_title("Best period: %5.3f days (p-value: %5.2e)" % (bestperiod, prob), fontsize = 8)
                plt.savefig("%s/%s/%s/CANDIDATES/final_%s_%s_%04i-%04i_grid%02i_%s_periodogram.png" % (webdir, field, CCD, field, CCD, ipixavg[it], jpixavg[it], fileref, resampling), bbox_inches = 'tight')

        if dohtml:

            print "Doing html"

            # plot coordinates of selected targets together with other catalogues
            fig, ax = plt.subplots(figsize = (10, 5))
        
            ax.scatter(RAavg, DECavg, marker = 'o', alpha = 0.5, s = 400, c = 'g', edgecolors = 'none')
            
            for i in range(len(RAavg)):
                if RAavg[i] >= RAmin and RAavg[i] <= RAmax and DECavg[i] >= DECmin and DECavg[i] <= DECmax:
                    ax.text(RAavg[i], DECavg[i], "%i %i (%i times)" % (ipixavg[i], jpixavg[i], npoints[i]), fontsize = 6)
                
        # plot catalogue information and matches
        jsonname = "%s/%s/%s/CANDIDATES/final_map_%s_%s_%02i.json" % (webdir, field, CCD, field, CCD, fileref)
        if dohtml:
            ijcoord = plotcatalogue(ax, jsonname, RAavg, DECavg, ipixavg, jpixavg, np.ones(len(RAavg), dtype = bool), posflux, probpairmany, labelsmany, RAmin, RAmax, DECmin, DECmax, sharedir, field, CCD, False)
        else:
            ijcoord = plotcatalogue(None, jsonname, RAavg, DECavg, ipixavg, jpixavg, np.ones(len(RAavg), dtype = bool), posflux, probpairmany, labelsmany, RAmin, RAmax, DECmin, DECmax, sharedir, field, CCD, False)

        # if candidates were plotted, prepare webpage
        if not (ijcoord is None):
            (icoord, jcoord, ngood) = ijcoord

            if dohtml:
                # create webpage with information
                doweb(webdir, field, CCD, "final", (ngood > 0), np.sum(fluxmax) > 0, fileref, resampling, RAavg, DECavg, ipixavg, jpixavg, icoord, jcoord, verbose)

        if dohtml:
            # nice ticks
            plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(HMS))
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(HMS))
            
            ax.set_xlim(RAmax, RAmin)
            ax.set_ylim(DECmin, DECmax)
            ax.set_xlabel('RA [hr]')
            ax.set_ylabel('DEC [deg]')
            plt.grid(True)
            plt.savefig("%s/%s/%s/CANDIDATES/final_map_%s_%s_%02i.png" % (webdir, field, CCD, field, CCD, fileref), pad_inches = 0.01, bbox_inches = 'tight')
        
    else:
        if verbose:
            print "    No repeated candidates."

# delete objects in memory

if verbose:
    print "\n\nCleaning objects in memory..."

#if os.path.exists("%s/LISTHEAD/%s_%s_%02i_listhead.txt" % (sharedir, field, CCD, fileref)):
#    command = "rm %s/%s_%s_%02i_listhead.txt" % (sharedir, field, CCD, fileref)
#    print command
#    os.system(command)
#
#if os.path.exists("%s/LISTHEAD/%s_%s_%02i_listhead.txt" % (sharedir, field, CCD, filesci)):
#    command = "rm %s/LISTHEAD/%s_%s_%02i_listhead.txt" % (sharedir, field, CCD, filesci)
#    print command
#    os.system(command)

# print the local time
printtime("END")
