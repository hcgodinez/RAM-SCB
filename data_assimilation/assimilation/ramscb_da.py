import numpy
import pickle
import datetime
import copy
import itertools
import pdb

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import pylab
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import scipy
from scipy import integrate
from scipy import interpolate
from scipy import spatial

import spacepy
from spacepy import pycdf
from spacepy import datamodel

import interpolation

# -----------------------------------------------------------------
#       get RAM dimension and coordinates
# -----------------------------------------------------------------

def get_ram_coordinates(ramscb):
    '''
    Parameters
    ==========
        ramscb : spacepy HDF5 datamodel object
            spacepy HDF5 object with the ramscb data, which is usually read as

            >>> import spacepy
            >>> from spacepy import datamodel
            >>> ramscb = datamodel.fromHDF5('ram-scbe-restart-file-name.nc')

    Returns
    =======
        species : dictionary
            dictionary with species name
        Lshell : numpy array
            array with Lshell values used in the RAM-SCBE grid. As of now the
            values are hardcoded into the RAM-SCBE code, hence they are also
            hardcoded here. The values range from L=1.75 to L=6.5. The number
            of L-shell can be determined from the RAM-SCBE restart file key
            'nR'
        MLT : numpy array
            array with local magnetic time use in the RAM-SCBE grid. As with
            L-shell, these are also hardcoded. The number of MLT can be
            determined from the RAM-SCBE restart file key 'nT'
        energy : numpy array
            array with energy bins use in the RAM-SCBE grid. As with
            L-shell, these are also hardcoded. The number of MLT can be
            determined from the RAM-SCBE restart file key 'nE'
        pitch_angle : numpy array
            array with pitch angle's use in the RAM-SCBE grid. As with
            L-shell, these are also hardcoded. The number of MLT can be
            determined from the RAM-SCBE restart file key 'nPa'

    Description
    ===========
    This routine gets the coordinates for RAM. NOTE: the coordinate are hardcoded into the RAM-SCB code.

    Examples
    ========

    >>> import spacepy
    >>> from spacepy import datamodel
    >>> ramscb = datamodel.fromHDF5('ram-scbe-restart-file-name.nc')
    >>> species,Lshell,MLT,energy,pitch_angle = get_ram_coordinates(ramscb)

    '''
    # ----------------------------
    #   get variables from restart
    # ----------------------------

    # L-Shell coordinates are hardcoded into RAM-SCB
    NL = ramscb['nR'].shape[0]
    Lmin = 1.75
    Lmax = 6.5
    Lshell = numpy.linspace(Lmin,Lmax,NL)

    # magnetic local time is also hardcoded into RAM-SCB
    NMLT = ramscb['nT'].shape[0]
    MLTmin = 0.0
    MLTmax = 24.0
    MLT = numpy.linspace(MLTmin,MLTmax,NMLT)

    # get the energy grid
    NE = ramscb['nE'].shape[0]
    energy = numpy.array(ramscb['EnergyGrid'])

    # get the pitch angle grid
    NPA = ramscb['nPa'].shape[0]
    pitch_angle = numpy.array(ramscb['PitchAngleGrid'])

    species = {
            0: 'FluxE',
            1: 'FluxH',
            2: 'FluxHe',
            3: 'FluxO'
            }

    return species,Lshell,MLT,energy,pitch_angle

# -----------------------------------------------------------------
#       compute flux from RAM-SCB restart files
# -----------------------------------------------------------------

def compute_restart_flux(ramscb):

    '''
    Parameters
    ==========
        ramscb : spacepy HDF5 datamodel object
            spacepy HDF5 object with the ramscb data, which is usually read as

            >>> import spacepy
            >>> from spacepy import datamodel
            >>> ramscb = datamodel.fromHDF5('ram-scbe-restart-file-name.nc')

    Returns
    =======
        species : dictionary
            dictionary with species name
        Lshell : numpy array
            array with Lshell values used in the RAM-SCBE grid. As of now the
            values are hardcoded into the RAM-SCBE code, hence they are also
            hardcoded here. The values range from L=1.75 to L=6.5. The number
            of L-shell can be determined from the RAM-SCBE restart file key
            'nR'
        MLT : numpy array
            array with local magnetic time use in the RAM-SCBE grid. As with
            L-shell, these are also hardcoded. The number of MLT can be
            determined from the RAM-SCBE restart file key 'nT'
        energy : numpy array
            array with energy bins use in the RAM-SCBE grid. As with
            L-shell, these are also hardcoded. The number of MLT can be
            determined from the RAM-SCBE restart file key 'nE'
        pitch_angle : numpy array
            array with pitch angle's use in the RAM-SCBE grid. As with
            L-shell, these are also hardcoded. The number of MLT can be
            determined from the RAM-SCBE restart file key 'nPa'
        Flux : numpy array
            computed flux from the RAM-SCBE restart file. To compute the flux
            the restart file must also write out FNHS and FFactor arrays. Both
            these arrays are written in the restart file for the data
            assimilation branch of the RAM-SCBE code.

    Description
    ===========
    This routine computes the flux for all species from the RAM-SCBE restart files.

    Examples
    ========

    >>> import spacepy
    >>> from spacepy import datamodel
    >>> ramscb = datamodel.fromHDF5('ram-scbe-restart-file-name.nc')
    >>> species,Lshell,MLT,energy,pitch_angle,Flux = compute_restart_flux(ramscb)

    '''

    # ----------------------------
    #   get variables from restart
    # ----------------------------

###    # L-Shell coordinates are hardcoded into RAM-SCB
###    NL = ramscb['nR'].shape[0]
###    Lmin = 1.75
###    Lmax = 6.5
###    Lshell = numpy.linspace(Lmin,Lmax,NL)
###
###    # magnetic local time is also hardcoded into RAM-SCB
###    NMLT = ramscb['nT'].shape[0]
###    MLTmin = 0.0
###    MLTmax = 24.0
###    MLT = numpy.linspace(MLTmin,MLTmax,NMLT)
###
###    # get the energy grid
###    NE = ramscb['nE'].shape[0]
###    energy = numpy.array(ramscb['EnergyGrid'])
###
###    # get the pitch angle grid
###    NPA = ramscb['nPa'].shape[0]
###    pitch_angle = numpy.array(ramscb['PitchAngleGrid'])

    # get information from ramscb
    species,Lshell,MLT,energy,pitch_angle = get_ram_coordinates(ramscb)

    # number of L-shells
    NL = ramscb['nR'].shape[0]
    # number of magnetic local time
    NMLT = ramscb['nT'].shape[0]
    # number of energy channels
    NE = ramscb['nE'].shape[0]
    # number of pitch angles
    NPA = ramscb['nPa'].shape[0]
    # number of species
    NS = ramscb['nS'].shape[0]

    # get FFactor array
    FFactor = ramscb['FFactor']
    tmp_FNHS = numpy.array(ramscb['FNHS'])

    FNHS = numpy.zeros((NPA,NMLT,NL),dtype=float)
    FNHS[:,:,0] = tmp_FNHS[:,:,0]
    FNHS[:,:,1] = tmp_FNHS[:,:,0]
    for iL in numpy.arange(2,NL):
        FNHS[:,:,iL] = tmp_FNHS[:,:,iL-1]

    # ----------------------------
    #   compute fluxes
    # ----------------------------

###    species = {
###            0: 'FluxE',
###            1: 'FluxH',
###            2: 'FluxHe',
###            3: 'FluxO'
###            }

    # initialize flux array
    Flux = numpy.zeros((NS,NL,NMLT,NE,NPA),dtype=float)

    for iS,iL,iMLT,iE,iPA in itertools.product(numpy.arange(NS),
            numpy.arange(NL),numpy.arange(NMLT),numpy.arange(NE),
            numpy.arange(NPA)):

        # compute the flux
        Flux[iS,iL,iMLT,iE,iPA] = (ramscb[species[iS]][iPA,iE,iMLT,iL]/
                (FFactor[iPA,iE,iL,iS]*FNHS[iPA,iMLT,iL]))

    return species,Lshell,MLT,energy,pitch_angle,Flux

# -----------------------------------------------------------------
#       compute flux from RAM-SCB restart files
# -----------------------------------------------------------------

def compute_restart_flux_old(filename):

    '''
    Parameters
    ==========
        filename : string
            name, including path, of the RAM-SCBE restart file

    Returns
    =======
        species : dictionary
            dictionary with species name
        Lshell : numpy array
            array with Lshell values used in the RAM-SCBE grid. As of now the
            values are hardcoded into the RAM-SCBE code, hence they are also
            hardcoded here. The values range from L=1.75 to L=6.5. The number
            of L-shell can be determined from the RAM-SCBE restart file key
            'nR'
        MLT : numpy array
            array with local magnetic time use in the RAM-SCBE grid. As with
            L-shell, these are also hardcoded. The number of MLT can be
            determined from the RAM-SCBE restart file key 'nT'
        energy : numpy array
            array with energy bins use in the RAM-SCBE grid. As with
            L-shell, these are also hardcoded. The number of MLT can be
            determined from the RAM-SCBE restart file key 'nE'
        pitch_angle : numpy array
            array with pitch angle's use in the RAM-SCBE grid. As with
            L-shell, these are also hardcoded. The number of MLT can be
            determined from the RAM-SCBE restart file key 'nPa'
        Flux : numpy array
            computed flux from the RAM-SCBE restart file. To compute the flux
            the restart file must also write out FNHS and FFactor arrays. Both
            these arrays are written in the restart file for the data
            assimilation branch of the RAM-SCBE code.

    Description
    ===========
    This routine computes the flux for all species from the RAM-SCBE restart files.

    Examples
    ========

    >>> filename = 'ram-scbe-restart-file-name.nc'
    >>> species,Lshell,MLT,energy,pitch_angle,Flux = compute_restart_flux(filename)

    '''

    # ----------------------------
    #   load restart file
    # ----------------------------

    ramscb = datamodel.fromHDF5(filename)

    # ----------------------------
    #   get variables from restart
    # ----------------------------

    # L-Shell coordinates are hardcoded into RAM-SCB
    NL = ramscb['nR'].shape[0]
    Lmin = 1.75
    Lmax = 6.5
    Lshell = numpy.linspace(Lmin,Lmax,NL)

    # magnetic local time is also hardcoded into RAM-SCB
    NMLT = ramscb['nT'].shape[0]
    MLTmin = 0.0
    MLTmax = 24.0
    MLT = numpy.linspace(MLTmin,MLTmax,NMLT)

    # get the energy grid
    NE = ramscb['nE'].shape[0]
    energy = numpy.array(ramscb['EnergyGrid'])

    # get the pitch angle grid
    NPA = ramscb['nPa'].shape[0]
    pitch_angle = numpy.array(ramscb['PitchAngleGrid'])

    # get number of species
    NS = ramscb['nS'].shape[0]

    # get FFactor array
    FFactor = ramscb['FFactor']
    tmp_FNHS = numpy.array(ramscb['FNHS'])

    FNHS = numpy.zeros((NPA,NMLT,NL),dtype=float)
    FNHS[:,:,0] = tmp_FNHS[:,:,0]
    FNHS[:,:,1] = tmp_FNHS[:,:,0]
    for iL in numpy.arange(2,NL):
        FNHS[:,:,iL] = tmp_FNHS[:,:,iL-1]

    # ----------------------------
    #   compute fluxes
    # ----------------------------

    species = {
            0: 'FluxE',
            1: 'FluxH',
            2: 'FluxHe',
            3: 'FluxO'
            }

    # initialize observation operator array
    H = numpy.zeros((NS,NL,NMLT,NE,NPA),dtype=float)

    # initialize flux array
    Flux = numpy.zeros((NS,NL,NMLT,NE,NPA),dtype=float)

    for iS,iL,iMLT,iE,iPA in itertools.product(numpy.arange(NS),
            numpy.arange(NL),numpy.arange(NMLT),numpy.arange(NE),
            numpy.arange(NPA)):

        # compute the flux
        Flux[iS,iL,iMLT,iE,iPA] = (ramscb[species[iS]][iPA,iE,iMLT,iL]/
                (FFactor[iPA,iE,iL,iS]*FNHS[iPA,iMLT,iL]))

        # populate observation array
        H[iS,iL,iMLT,iE,iPA] = 1.0/(FFactor[iPA,iE,iL,iS]*FNHS[iPA,iMLT,iL])

    return ramscb,species,Lshell,MLT,energy,pitch_angle,Flux

# -----------------------------------------------------------------
#       compute RAM-SCB omnidirectional flux
# -----------------------------------------------------------------

def compute_omnidirectional_flux(species,Lshell,MLT,energy,pitch_angle,Flux):

    '''
    Parameters
    ==========
        species : dictionary
            dictionary with species name
        Lshell : numpy array
            array with Lshell values used in the RAM-SCBE grid.
            'nR'
        MLT : numpy array
            array with local magnetic time use in the RAM-SCBE grid.
        energy : numpy array
            array with energy bins use in the RAM-SCBE grid.
        pitch_angle : numpy array
            array with pitch angle's use in the RAM-SCBE grid.
        Flux : numpy array
            RAM-SCBE flux

    Returns
    =======
        ramscb_OF : numpy array
            computed omnidirectional flux

    Description
    ===========
    This routine computes the omnidirectional flux for all species given in the
    Flux array. The omnidirectional flux is computed using the following
    formula:
                 
       OF = 2*pi \int{ J(\alpha) \sin{\alpha} } d \alpha,
    
    where J(\alpha) is the flux at the pitch angle \alpha.

    Examples
    ========

    >>> filename = 'ram-scbe-restart-file-name.nc'
    >>> species,Lshell,MLT,energy,pitch_angle,Flux = compute_restart_flux(filename)
    >>> ramscb_OF = compute_omnidirectional_flux(species,Lshell,MLT,energy,pitch_angle,Flux)

    '''

    # ----------------------------
    #   get variables
    # ----------------------------

    NL = Lshell.shape[0]
    NMLT = MLT.shape[0]
    NE = energy.shape[0]
    NPA = pitch_angle.shape[0]
    NS = len(species)

    # compute MLT in radians
    rad = numpy.pi/12.0*MLT

    # ----------------------------
    #   compute omnidirectional flux
    # ----------------------------
    # the omnidirectional flux is computed using the following formula:
    #             
    #   OF = 2*pi \int{ J(\alpha) \sin{\alpha} } d \alpha,
    #
    # where J(\alpha) is the flux at the pitch angle \alpha. The omnidirectional
    # flux will be compute for each energy and L-shell

    ramscb_OF = numpy.zeros((NS,NL,NMLT,NE),dtype=float)

    # iterative loop for the flux
    for iS,iL,iMLT,iE in itertools.product(numpy.arange(NS),
            numpy.arange(NL),numpy.arange(NMLT),numpy.arange(NE)):

        # get x and y for the integral
        x = numpy.pi/180.0*pitch_angle
        y = numpy.sin(x)*Flux[iS,iL,iMLT,iE,:]

        # reorder from smallest to biggest
        idx = x.argsort()
        x = x[idx]
        y = y[idx]

        # compute integral
        ramscb_OF[iS,iL,iMLT,iE] = 2.0*numpy.pi*integrate.cumtrapz(y,x)[-1]

    return ramscb_OF

# -----------------------------------------------------------------
#       compute RBSP omnidirectional flux
# -----------------------------------------------------------------

def compute_omnidirectional_flux_RBSP(rbsp,species,start,stop):

    '''
    Parameters
    ==========
        rbsp : spacepy CDF object
            spacepy CDF object with the RBSP data, which is usually read as
            >>> import spacepy
            >>> from spacepy import pycdf
            >>> rbsp = pycdf.CDF('name_rbsp_file_to_read.nc')

        species : string
            species from which to compute the omnidirectional flux, the values
            are either:
            - FPDU : protons
            - FEDU : electrons

        start : datetime
            start time to consider observations

        stop : datetime
            stop time to consider observations
            
    Returns
    =======
        flux : numpy array
            flux for the specified species and within the time interval
            indicated by the start and stop datetime objects. 
        Lshell : numpy array
            array with Lshell values in the RBSP data.
        MLT : numpy array
            array with local magnetic time in the RBSP data.
        energy : numpy array
            array with energy bins use in the RBSP data.
        pitch_angle : numpy array
            array with pitch angle's use in the RBSP data.
        dates : datetime numpy array
            dates of the RBSP data within the time interval specified by the
            start and stop datatime objects
        rbsp_OF : numpy array
            computed omnidirectional flux for the RBSP data, as a funciton of
            the date, MLT, L-shell and energy

    Description
    ===========
    This routine computes the omnidirectional flux for RBSP for the specified
    species and whitin the time interval given by start and stop datetime
    objects. The omnidirectional flux is computed using the following formula:
                 
       OF = 2*pi \int{ J(\alpha) \sin{\alpha} } d \alpha,
    
    where J(\alpha) is the flux at the pitch angle \alpha.

    Examples
    ========

    >>> import numpy
    >>> import datetime
    >>> import spacepy
    >>> from spacepy import pycdf
    >>> rbsp = pycdf.CDF('name_rbsp_file_to_read.nc')
    >>> start = datetime.datetime(2015,3,7)
    >>> stop = datetime.datetime(2015,3,17)
    >>> species = 'FPDU'
    >>> Lshell,MLT,energy,dates,rbsp_flux,rbsp_OF = compute_omnidirectional_flux_RBSP(rbsp,species,start,stop)

    '''

    # ----------------------------
    #       read data
    # ----------------------------

    # read RBSP data file
    #rbsp = pycdf.CDF(filename)

    # ----------------------------
    #   get energy and pitch angles
    # ----------------------------

    # get number of pitch angles
    #pitch_angle = rbsp['FPDU_Alpha'][:]
    pitch_angle = rbsp[species+'_Alpha'][:]
    idx_pitch_angle = numpy.where(pitch_angle >= 0.0)[0]
    pitch_angle = pitch_angle[idx_pitch_angle]

    # get Energy spectrum between 1.0 KeV and 400.0 KeV 
    # for both satellites
    #energy = rbsp['FPDU_Energy'][:]
    energy = rbsp[species+'_Energy'][:]
    idx_E = numpy.where(energy >= 0.0)[0]
    energy = energy[idx_E]

    # ----------------------------
    #   get dates
    # ----------------------------

    #dates = rbsp['FPDU_Epoch'][:]
    dates = rbsp[species+'_Epoch'][:]
    idx_start = numpy.where(dates >= start)[0]
    idx_stop = numpy.where(dates <= stop)[0]
    idx_time = numpy.intersect1d(idx_start,idx_stop)

    if (len(dates) == 0):
        raise ValueError('no RBSP dates available within the time interval specified')

    # select only dates within time interval
    dates = dates[idx_time]

    # ----------------------------
    #   get L-shell and MLT for the time interval
    # ----------------------------

    # get L-shell
    Lshell = rbsp['L'][:]
    Lshell = Lshell[idx_time]

    # get MLT
    MLT = rbsp['MLT'][:]
    MLT = MLT[idx_time]

    # ----------------------------
    #   compute omnidirectional flux
    # ----------------------------
    # the omnidirectional flux is computed using the following formula:
    #             
    #   OF = 2*pi \int{ J(\alpha) \sin{\alpha} } d \alpha,
    #
    # where J(\alpha) is the flux at the pitch angle \alpha. The omnidirectional
    # flux will be compute for each energy and L-shell

    # first filter out the invalid values
    #flux_rbsp = rbsp['FPDU'][:]
    rbsp_flux = rbsp[species][:]
    rbsp_flux = rbsp_flux[idx_time,:,:]
    rbsp_flux = rbsp_flux[:,idx_pitch_angle,:]
    rbsp_flux = rbsp_flux[:,:,idx_E]

    # initialize the omnidirectional flux array
    rbsp_OF = numpy.zeros((dates.shape[0],energy.shape[0]),dtype=float)

    # loop over dates and energy spectrum
    for idate,date in enumerate(dates):
        for iE,Ev in enumerate(energy):

            idx_valid_flux = numpy.where(rbsp_flux[idate,:,iE] > 0.0)[0]

            if len(idx_valid_flux) > 1:

                # get independent and dependent variables for integral
                x = numpy.pi/180.0*pitch_angle[idx_valid_flux]
                y = numpy.sin(x)*rbsp_flux[idate,idx_valid_flux,iE]

                # reorder from smallest to biggest
                idx = x.argsort()
                x = x[idx]
                y = y[idx]

                # compute integral
                rbsp_OF[idate,iE] = integrate.cumtrapz(y,x)[-1]

    return Lshell,MLT,energy,pitch_angle,dates,rbsp_flux,rbsp_OF

# -----------------------------------------------------------------
#       get SM coordinates of RBSP satellite from emphemeris file
# -----------------------------------------------------------------

def get_SM_RBSP(rbsp_dates,rbsp_ephem):

    '''
    Parameters
    ==========
        rbsp_dates : numpy array of datetimes
            numpy array of datetime objects for which we want the position

        rbsp_ephem : spacepy HDF5 datamodel object
            spacepy HDF5 object that includes the three dimensional position of
            the RBSP spacecraft in SM coordinates. This is usually done by loading the ephemerus RBSP files:

            >>> import spacepy
            >>> from spacepy import datamodel
            >>> rbsp_ephem = datamodel.fromHDF5('rbsp_ephemeris_file.h5')

            the ephemeris files can be found in:

            https://rbsp-ect.newmexicoconsortium.org/data_pub/rbspa/MagEphem/
            https://rbsp-ect.newmexicoconsortium.org/data_pub/rbspb/MagEphem/
            
    Returns
    =======
        rbsp_sm_position : numpy array
            three dimensional position of the RBSP spacecraft in SM coordinates

    Description
    ===========
        This subroutine computes the SM coordinates of the RBSP flux observations by using the RBSP ephemeris file

    Examples
    ========

    >>> import numpy
    >>> import spacepy
    >>> from spacepy import pycdf
    >>> from spacepy import datamodel
    >>> rbsp = pycdf.CDF('name_rbsp_file_to_read.nc')
    >>> rbsp_ephem = datamodel.fromHDF5('name_rbsp_ephemeir_file_to_read.h5')
    >>> rbsp_dates = numpy.array(rbsp['Epoch'])
    >>> rbsp_sm_position = ramscb_da.get_SM_RBSP(rbsp_dates,rbsp_ephem)

    '''

    # ----------------------------
    #   convert to datetime numpy array
    # ----------------------------

    # get datetime for ephemeris file
    rbsp_ephem_dates = []
    for idate in rbsp_ephem['IsoTime']:

        # decode from bites str to str
        idate = idate.decode()

        # convert to datetime object and append to list
        rbsp_ephem_dates.append(datetime.datetime.strptime(idate,'%Y-%m-%dT%H:%M:%SZ'))

    # convert to numpy array
    rbsp_ephem_dates = numpy.array(rbsp_ephem_dates)

    # ----------------------------
    #   loop over the rbsp flux file dates and compute the position using
    #   ephemeris file 
    # ----------------------------

    # initialize the rbsp position array
    rbsp_sm_position = []

    #pdb.set_trace()

    for idate in numpy.arange(rbsp_dates.shape[0]):

        # get date from RBSP flux file
        rbsp_date_k = rbsp_dates[idate]

        # find date in RBSP ephemeris file
        idx = numpy.argmin(numpy.abs(rbsp_ephem_dates-rbsp_date_k))

        # find time interval
        if (rbsp_date_k > rbsp_ephem_dates[idx]):

            # get time interval
            t0 = rbsp_ephem_dates[idx]
            t1 = rbsp_ephem_dates[idx+1]

            # get coordinates for t0
            p0 = numpy.array(rbsp_ephem['Rsm'][idx,:])
            p1 = numpy.array(rbsp_ephem['Rsm'][idx+1,:])

        else:

            # get time interval
            t0 = rbsp_ephem_dates[idx-1]
            t1 = rbsp_ephem_dates[idx]

            # get coordinates for t0
            p0 = numpy.array(rbsp_ephem['Rsm'][idx-1,:])
            p1 = numpy.array(rbsp_ephem['Rsm'][idx,:])

        #print('date rbsp flux file: '+rbsp_date_k.strftime('%Y-%m-%dT%H:%M:%S:%f'))
        #print('index: '+str(idx))
        #print('t0: '+t0.strftime('%Y-%m-%dT%H:%M:%S:%f'))
        #print('t1: '+t1.strftime('%Y-%m-%dT%H:%M:%S:%f'))
        #pdb.set_trace()

        # compute difference between rbsp flux date and initial date in interval
        dt = rbsp_date_k-t0
        dt = dt.total_seconds()

        # compute difference between date interval
        dtt = t1-t0
        dtt = dtt.total_seconds()

        # compute proportion
        pt = dt/dtt

        # compute distance between the two points
        v = p1-p0

        # compute the desired distance
        d = numpy.linalg.norm(v)*pt

        # compute interpolation between the points
        pd = p0+d/numpy.linalg.norm(v)*v

        # append to rbsp position array
        rbsp_sm_position.append(copy.copy(pd))

    # convert to numpy array
    rbsp_sm_position = numpy.array(rbsp_sm_position)

    return rbsp_sm_position

# -----------------------------------------------------------------
#       get coordinate index of the SCB coordinates that match SM coordinates
#       from the RBSP satellite position
# -----------------------------------------------------------------

def find_SM_point_SCB_grid(rbsp_sm_position,ramscb):

    '''
    Parameters
    ==========
        rbsp_sm_position : numpy array
            three dimensional position of the RBSP spacecraft in SM coordinates

        ramscb : spacepy HDF5 datamodel object
            spacepy HDF5 object with the ramscb data, which is usually read as

            >>> import spacepy
            >>> from spacepy import datamodel
            >>> ramscb = datamodel.fromHDF5('ram-scbe-restart-file-name.nc')

    Returns
    =======
        scb_coor_index : numpy array
            array which stores the index of the SCB coordinates that is closest
            to the RBSP satellite position provided

    Description
    ===========
        This subroutine provides the index of the SCB coordinates that is
        closests to the SM coordinates from the RBSP satellite position

    Examples
    ========

    >>> import numpy
    >>> import spacepy
    >>> from spacepy import pycdf
    >>> from spacepy import datamodel
    >>> import ramscb_da
    >>> rbsp = pycdf.CDF('name_rbsp_file_to_read.nc')
    >>> rbsp_dates = numpy.array(rbsp['Epoch'])
    >>> rbsp_ephem = datamodel.fromHDF5('rbsp_ephemeris_file.h5')
    >>> rbsp_sm_position = ramscb_da.get_SM_RBSP(rbsp_dates,rbsp_ephem)
    >>> ramscb = datamodel.fromHDF5('ram-scbe-restart-file-name.nc')
    >>> scb_coor_index = ramscb_da.find_SM_point_SCB_grid(rbsp_sm_position,ramscb)

    '''

    # ----------------------------
    #   get SM coordinates from both SCB and RBSP
    # ----------------------------

    # get magnetic field coordinates from ram-scb
    scb_sm_x = numpy.array(ramscb['x']).flatten()
    scb_sm_y = numpy.array(ramscb['y']).flatten()
    scb_sm_z = numpy.array(ramscb['z']).flatten()

    # get single 2D array
    scb_sm = numpy.array([scb_sm_x, scb_sm_y, scb_sm_z]).T

    # ----------------------------
    #   locate the closest point using 
    #   KDTree object from spatial package in scipy
    # ----------------------------

    # declare KDTree object to find closest points
    ramscb_kdtree = spatial.KDTree(scb_sm)

    # initialize the index array
    idx_rbsp_ramscb = []

    # initialize the distance array
    dist_rbsp_ramscb = []

    # coordinate for loop
    for coord in rbsp_sm_position:

        # find closest point using query from KDTree 
        dist,idx = ramscb_kdtree.query(coord)

        # append the distance
        dist_rbsp_ramscb.append(dist)

        # append the index
        idx_rbsp_ramscb.append(idx)

    # convert to numpy array
    idx_rbsp_ramscb = numpy.array(idx_rbsp_ramscb)

    # convert to the 3D array in the scb coordinates
    scb_coor_index = []
    for idx in idx_rbsp_ramscb:
        idx_scb_coor = numpy.unravel_index(idx,ramscb['x'].shape)
        scb_coor_index.append(idx_scb_coor)

    # convert to numpy array
    scb_coor_index = numpy.array(scb_coor_index)

    return scb_coor_index

# -----------------------------------------------------------------
#       get coordinate index of the SCB coordinates that match SM coordinates
#       from the RBSP satellite position
# -----------------------------------------------------------------

def relabel_pitchangle(ramscb):

    '''
    Parameters
    ==========
        ramscb : spacepy HDF5 datamodel object
            spacepy HDF5 object with the ramscb data, which is usually read as

            >>> import spacepy
            >>> from spacepy import datamodel
            >>> ramscb = datamodel.fromHDF5('ram-scbe-restart-file-name.nc')

    Returns
    =======
        scb_pa : numpy array
            array which stores the index of the available pitch angles for the off-equator flux

    Description
    ===========
        This subroutine provides the index of the available pitch angles valid
        for the off-equator flux. 

    Examples
    ========

    >>> import numpy
    >>> import spacepy
    >>> from spacepy import datamodel
    >>> import ramscb_da
    >>> ramscb = datamodel.fromHDF5('ram-scbe-restart-file-name.nc')
    >>> scb_pa = ramscb_da.relabel_pitchangle(ramscb)

    '''

    # ----------------------------
    #   get RAM pitch angles
    # ----------------------------

    # convert RAM pitch angle into radians
    PA = numpy.array(ramscb['PitchAngleGrid'])*numpy.pi/180.0

    # number of pitch angles
    nPA = PA.shape[0]

    # get cosine of the PA, only for convenience to re-label pitch angles in
    # the SCB 3D grid
    cosPA = numpy.cos(PA)

    # set cos(pi/2) = 0.0
    cosPA[0] = 0.0

    # ----------------------------
    # get SCB magnetic field
    # ----------------------------

    # get magnetic field on the SCB grid
    Bx = numpy.array(ramscb['Bx'])
    By = numpy.array(ramscb['By'])
    Bz = numpy.array(ramscb['Bz'])
    B = numpy.sqrt(Bx**2+By**2+Bz**2)

    # ----------------------------
    # compute flux in SCB grid-points
    # ----------------------------

    # get the dimension
    nx,ny,nz = Bx.shape

    # initialize the pitch angle index array
    scb_pa = -1.0*numpy.ones((nx,ny,nz,nPA),dtype=int)

    # square of the pitch angle values
    cosPA2 = cosPA**2

    # massive for loop for the SCB grid
    for i in numpy.arange(nx):
        for j in numpy.arange(ny):

            #pdb.set_trace()

            # get equatorial magnetic minimum value
            Beq = numpy.min(numpy.abs(B[i,j,:]))

            # compute the equivalent equatorial alpha
            Bloc = B[i,j,:]

            # test with numpy.broadcast_to
            Bloc = numpy.broadcast_to(Bloc,(cosPA2.shape[0],nz)).T
            alpha_eq = 1.0-Beq/Bloc*(1.0-cosPA2)

            # apply cos inverse and convert to degrees
            #alpha_eq = numpy.rad2deg(numpy.arcsin(numpy.sqrt(alpha_eq)))-90.0
            alpha_eq = numpy.rad2deg(numpy.arccos(numpy.sqrt(alpha_eq)))

            # assigned re-labeled pitch angles
            scb_pa[i,j,:,:] = alpha_eq[...]

###            # for loop along the magnetic field
###            for k in numpy.arange(nz):
###                Bloc = B[i,j,k] 
###
###                #pdb.set_trace()
###
###                # pitch angle loop
###                for l,alpha in enumerate(cosPA):
###
###                    # compute the equivalent equatorial alpha
###                    alpha_eq = 1.0-Beq/Bloc*(1.0-alpha**2)
###
###                    if (alpha_eq >= 0):
###
###                        # get the square root of the pitch angle
###                        alpha_eq = numpy.sqrt(alpha_eq)
###
###                        # find appropriate equatorial alpha
###                        # Because of the magnetic field strength, only a small set
###                        # of unique pitch angles are present. That is to say, the
###                        # flux is not present in all pitch angles from 0 to pi/2
###                        # rad.
###                        idx = numpy.argmin(numpy.abs(cosPA-alpha_eq))
###                        scb_pa[i,j,k,l] = idx

    return scb_pa

# -----------------------------------------------------------------
#       interpolate the flux from RAM to SCB grid
# -----------------------------------------------------------------

def interpolate_RAM_flux_to_SCB(ramscb,ispecies,scb_pa,ram_flux=[]):
    '''
    Parameters
    ==========
        ramscb : spacepy HDF5 datamodel object
            spacepy HDF5 object with the ramscb data, which is usually read as

            >>> import spacepy
            >>> from spacepy import datamodel
            >>> ramscb = datamodel.fromHDF5('ram-scbe-restart-file-name.nc')

        ispecies : int
            species index to use in the interpolation. The species index is
            specified in the PARAM.in file, the default index order is the
            following:
                0 : H+
                1 : O+
                2 : He
                3 : e
            The default order can be found in ModRamGrids.f90 under the
            "NameVar" character variable

        scb_pa : numpy array
            array which stores the index of the available pitch angles for the
            off-equator flux

        ram_flux : numpy array, optional
            array with the computed flux from ram, if not provided we compute
            the flux using the ramscb object

    Returns
    =======

        scb_flux : numpy array
            array which stores the flux of the available pitch angles for the
            off-equator SCB grid-points

    Description
    ===========
        This subroutine provides the index of the available pitch angles valid
        for the off-equator flux. 

    Examples
    ========

    >>> import numpy
    >>> import spacepy
    >>> from spacepy import datamodel
    >>> import ramscb_da
    >>> ramscb = datamodel.fromHDF5('ram-scbe-restart-file-name.nc')
    >>> species,Lshell,MLT,energy,pitch_angle,Flux = compute_restart_flux(ramscb)
    >>> scb_pa = ramscb_da.relabel_pitchangle(ramscb)
    >>> ispecies = 0
    >>> scb_flux = ramscb_da.interpolate_RAM_flux_to_SCB(ramscb,ispecies,scb_pa,ram_flux=Flux)

    '''

    # ----------------------------
    #   get RAM coordinates and flux
    # ----------------------------

    if (len(ram_flux) == 0):

        species,Lshell,MLT,energy,pitch_angle,ram_flux = compute_restart_flux(ramscb)

    else:

        # get information from ramscb
        species,Lshell,MLT,energy,pitch_angle = get_ram_coordinates(ramscb)

    # get number of energy bins and pitch angles
    nE = energy.shape[0]
    nPA = pitch_angle.shape[0]
    nS = len(species)
    nLshell = Lshell.shape[0]
    nMLT = MLT.shape[0]

    # ----------------------------
    #   get RAM grid coordinates to cartesian coordinates
    # ----------------------------

    # convert MLT to radians
    rho = numpy.pi/12.0*MLT - numpy.pi/2.0

    # initialize the X and Y cartesian arrays
    X = numpy.zeros((nMLT,nLshell),dtype=float)
    Y = numpy.zeros((nMLT,nLshell),dtype=float)

    # convert to cartesian
    for j,L in enumerate(Lshell):
        for i,r in enumerate(rho):
            X[i,j] = L*numpy.cos(r)
            Y[i,j] = L*numpy.sin(r)

    # reshape array for KD tree function
    ram_grid = numpy.array([X.flatten(),Y.flatten()]).T

    # get KD tree object
    ram_kdtree = spatial.KDTree(ram_grid)

    # ----------------------------
    # get SCB grid coordinates
    # ----------------------------

    scbx = numpy.array(ramscb['x'])
    scby = numpy.array(ramscb['y'])
    scbz = numpy.array(ramscb['z'])

    # get the dimension
    nx,ny,nz = scbx.shape

    # get SCB gridpoint array
    scb_sm = numpy.array([scbx.flatten(), scby.flatten(), scbz.flatten()]).T

    # declare KDTree object to find closest points
    scb_kdtree = spatial.KDTree(scb_sm)

    # ----------------------------
    #   inerpolate flux to SCB grid
    # ----------------------------

    # only do the first species
    ispecies = 0

    # compute the pitch angles in the SCB grid
    scb_pa = relabel_pitchangle(ramscb)

    # initialize SCB flux array
    scb_flux = numpy.zeros((nS,nx,ny,nz,nE,nPA),dtype=float)

    # get SCB equatorial grid-points coordinates
    scb_grid = []
    scb_grid_index_flat_full = []
    scb_grid_index_flat = []

    # main loop to identify equatorial SCB grid-point
    for i in numpy.arange(nx):
        for j in numpy.arange(ny):

            # get equatorial SCB z-coordinate index
            k = numpy.argmin(numpy.abs(scbz[i,j,:]))

            # get x and y coordinates for the equatorial SCB grid-point
            scb_eq_x = scbx[i,j,k]
            scb_eq_y = scby[i,j,k]

            # ----------------------------
            #   test to see if the equatorial SCB grid-point is away from innter
            #   and outer boundary of RAM grid
            # ----------------------------
            scb_rho = numpy.arctan2(scb_eq_y,scb_eq_x)
            scb_Lshell = numpy.sqrt(scb_eq_x**2+scb_eq_y**2)

            if ( (scb_Lshell > Lshell[0]) and (scb_Lshell < Lshell[-1]) ):

                scb_grid.append([scb_eq_x,scb_eq_y])
                scb_grid_index_flat_full.append(numpy.ravel_multi_index((i,j,k),(nx,ny,nz)))
                scb_grid_index_flat.append(numpy.ravel_multi_index((i,j),(nx,ny)))

    # convert the point to numpy array
    scb_grid = numpy.array(scb_grid)

    # number of equatorial SCB grid-points
    num_eq_scb_grid = len(scb_grid_index_flat_full)

    # initialize the equatorial SCB flux
    scb_flux_eq_flat = numpy.zeros((nE,nPA,num_eq_scb_grid),dtype=float)

    # initialize scb flux in equator array
    scb_flux_eq = numpy.zeros((nE,nPA,nx,ny),dtype=float)

    # main for loop to interpolate the SCB flux along the equatorial plane
    # for all available energy channels and pitch-angles
    for iE in numpy.arange(nE):
        for iPA in numpy.arange(nPA):

            # get target ram flux
            ram_flux_EPA = ram_flux[ispecies,:,:,iE,iPA].T
            ram_flux_EPA = ram_flux_EPA.flatten()
            
            # need to mask the negative flux
            idx = numpy.where(ram_flux_EPA>0.0)[0]
            ram_flux_EPA = ram_flux_EPA[idx]
            ram_grid_tmp = ram_grid[idx,:]

            # get values and convert to log scale
            ram_flux_EPA = numpy.log(ram_flux_EPA)

            # interpolate RAM flux to equatorial SCB grid-points for the
            # particular energy and pitch angle
            scb_flux_eq_flat[iE,iPA,:] = \
                    numpy.exp(interpolate.griddata(ram_grid_tmp,ram_flux_EPA,scb_grid))

            # for equatorial SCB grid-points that are outside the convex hull
            # of the RAM grid-points, just get the nearest neighbor
            idx_nan = numpy.where(numpy.isnan(scb_flux_eq_flat[iE,iPA,:]))[0]
            if (len(idx_nan) > 0):
                scb_flux_eq_flat[iE,iPA,idx_nan] = numpy.exp(interpolate.griddata(ram_grid_tmp,ram_flux_EPA,scb_grid[idx_nan,:],method='nearest'))

            # place in second array
            flux_tmp2 = -1.0e+4*numpy.ones((nx*ny),dtype=float)
            flux_tmp2[scb_grid_index_flat] = scb_flux_eq_flat[iE,iPA,:]
            scb_flux_eq[iE,iPA,:,:] = flux_tmp2.reshape((nx,ny))


    return scb_flux_eq

# -----------------------------------------------------------------
#       get coordinate index of the SCB coordinates that match SM coordinates
#       from the RBSP satellite position
# -----------------------------------------------------------------

def interpolate_flux_RAMSCB_RBSP(rbsp_sm_position,ramscb):
    '''
    Parameters
    ==========
        rbsp_sm_position : numpy array
            three dimensional position of the RBSP spacecraft in SM coordinates

        ramscb : spacepy HDF5 datamodel object
            spacepy HDF5 object with the ramscb data, which is usually read as

                >>> import spacepy
                >>> from spacepy import datamodel
                >>> ramscb = datamodel.fromHDF5('ram-scbe-restart-file-name.nc')

    Returns
    =======
        ramscb_rbsp_flux : numpy array
            array which stores the interpolated RAM-SCB flux onto the satellite
            location.

    Description
    ===========
        This subroutine provides the interpolated directional flux of the
        RAM-SCB model onto the given SM coordinates of a satellite location.

    Examples
    ========

    >>> import numpy
    >>> import spacepy
    >>> from spacepy import pycdf
    >>> from spacepy import datamodel
    >>> import ramscb_da
    >>> ramscb = datamodel.fromHDF5('ram-scbe-restart-file-name.nc')
    >>> rbsp = pycdf.CDF('name_rbsp_file_to_read.nc')
    >>> rbsp_dates = numpy.array(rbsp['Epoch'])
    >>> rbsp_ephem = datamodel.fromHDF5('rbsp_ephemeris_file.h5')
    >>> rbsp_sm_position = ramscb_da.get_SM_RBSP(rbsp_dates,rbsp_ephem)
    >>> rbsp_flux = ramscb_da.interpolate_flux_RAMSCB_RBSP(rbsp_sm_position,ramscb)

    '''

    # get information of the ramscb object
    ramscb_species,ramscb_Lshell,ramscb_MLT, \
    ramscb_energy,ramscb_pitch_angle, \
    ramscb_flux = compute_restart_flux(ramscb)

    # get index of closes SCB grid-point to the RBSP satellite location
    scb_coor_index = find_SM_point_SCB_grid(rbsp_sm_position,ramscb)

    # get SCB grid coordinates that are on the equatorial plane
    scbx = numpy.array(ramscb['x'])
    scby = numpy.array(ramscb['y'])
    scbz = numpy.array(ramscb['z'])

    # ----------------------------
    #   get RAM grid coordinates to cartesian coordinates
    # ----------------------------

    # convert MLT to radians
    rho = numpy.pi/12.0*ramscb_MLT - numpy.pi/2.0

    X = numpy.zeros((ramscb['nT'].shape[0],ramscb['nR'].shape[0]),dtype=float)
    Y = numpy.zeros((ramscb['nT'].shape[0],ramscb['nR'].shape[0]),dtype=float)

    # convert to cartesian
    for j,L in enumerate(ramscb_Lshell):
        for i,r in enumerate(rho):
            X[i,j] = L*numpy.cos(r)
            Y[i,j] = L*numpy.sin(r)

    # reshape array for KD tree function
    ram_grid = numpy.array([X.flatten(),Y.flatten()]).T

    # get KD tree object
    ram_kdtree = spatial.KDTree(ram_grid)

    # main for loop to from the satellite SM coordinate location
    for idx_rbsp in scb_coor_index:

        # convert indexes to tuple
        idx = tuple(idx_rbsp)

        # ----------------------------
        # get SCB grid coordinates that are on the equatorial plane
        # ----------------------------

        # find the SCB grid-point that is closest to the equator
        idx_scb_min_z = numpy.argmin(numpy.abs(scbz[idx[0],idx[1],:]))

        # equatorial SCB grid-point along the magnetic field line that is
        # closest to the satellite location
        idx_scb_eq = tuple((idx[0],idx[1],idx_scb_min_z))

        # get x and y coordinates for the equatorial SCB grid-point
        scb_eq_x = scbx[idx_scb_eq]
        scb_eq_y = scby[idx_scb_eq]

        # ----------------------------
        # get RAM grid coordinates that are the closest to the SCB gridpoint
        # ----------------------------

        # get the 10 closes RAM spatial grid-point for the SCB grid-point at
        # the equator
        dist, idx_ram_scb = ram_kdtree.query(numpy.array([scb_eq_x,scb_eq_y]),10)

        pdb.set_trace()

        # unravel index for 2-D arrays
        idx_ram_scb_xy = []
        for ii in idx_ram_scb:
            idx_ram_scb_xy.append(numpy.unravel_index(ii,X.shape))

# -----------------------------------------------------------------
#       compute flux from RAM-SCB restart files
# -----------------------------------------------------------------

### def process_observations(filename,species,Lshell,MLT,energy,pitch_angle):
### 
###     '''
###     Parameters
###     ==========
###         filename : string
###             name, including path, of the RBSP observation level 3 files. These
###             files should include the proton and electron derived fluxes.
###         species : dictionary
###             dictionary with species name from RAM-SCBE
###         Lshell : numpy array
###             array with Lshell values used in the RAM-SCBE grid.
###             'nR'
###         MLT : numpy array
###             array with local magnetic time use in the RAM-SCBE grid.
###         energy : numpy array
###             array with energy bins use in the RAM-SCBE grid.
###         pitch_angle : numpy array
###             array with pitch angle's use in the RAM-SCBE grid.
### 
###     Returns
###     =======
###         rbsp_dates : datetime numpy array
###             dates for RBSP observations
###         rbsp_pitch_angle : numpy array
###             pitch angle for RBSP observations
###         rbsp_energy : numpy array
###             energy for RBSP observations
###         rbsp_flux : numpy array
###             flux from RBSP observations
###         rbsp_oflux : numpy array
###             computed omnidirectional flux for RBSP level 3 derived fluxes
###         H : numpy array
###             observation operator for assimilation
### 
###     Description
###     ===========
###     This routine processes the RBSP observations to compute the omnidirectional flux observed from the level 3 derived proton or electron fluxes. The routine also estimates what is the observation operator H for the data assimilation algorithm.
### 
###     Examples
###     ========
### 
###     >>> filename = 'ram-scbe-restart-file-name.nc'
###     >>> species,Lshell,MLT,energy,pitch_angle,Flux = compute_restart_flux(filename)
###     >>> filename = 'rbspa_rel04_ect-mageis-L3_20170907_v8.4.0.cdf'
###     >>> rbsp_dates,rbsp_pitch_angle,rbsp_energy,rbsp_flux,rbsp_oflux,H = 
###     >>>         process_observations(filename,species,Lshell,MLT,energy,pitch_angle)
### 
###     '''
### 
###     # ----------------------------
###     #       read data
###     # ----------------------------
### 
###     # read RBSP data file
###     rbsp = pycdf.CDF(filename)
### 
###     # ----------------------------
###     #   get energy and pitch angles
###     # ----------------------------
### 
###     # get number of valid pitch angles
###     PitchAngle = rbsp['FPDU_Alpha'][:]
###     idx_PA = numpy.where(PitchAngle >= 0.0)[0]
###     rbsp_pitch_angle = PitchAngle[idx_PA]
### 
###     # get valid Energy spectrum
###     Energy = rbsp['FPDU_Energy'][:]
###     idx_E = numpy.where(Energy >= 0.0)[0]
###     rbsp_energy = Energy[idx_E]
### 
###     rbsp_L = rbsp['L']
### 
###     rbsp_MLT = rbsp['MLT']
### 
###     # ----------------------------
###     #   get dates
###     # ----------------------------
### 
###     rbsp_dates = rbsp['FPDU_Epoch'][:]
### 
###     # -------------------------
###     #   get valid flux for RBSP-A
###     # -------------------------
### 
###     # first filter out the invalid values
###     rbsp_flux = rbsp['FPDU'][:]
###     rbsp_flux = rbsp_flux[:,idx_PA,:]
###     rbsp_flux = rbsp_flux[:,:,idx_E]
### 
###     # -------------------------
###     #   interpolate to closest model grid-point
###     # -------------------------
### 
###     # get number of grid-points for RBSP obs
###     Ndates = rbsp_dates.shape[0]
###     NE = rbsp_energy.shape[0]
###     NPA = rbsp_pitch_angle.shape[0]
### 
###     # get number of grid-points for RAM-SCB
###     NS_model = len(species)
###     NL_model = Lshell.shape[0]
###     NMLT_model = MLT.shape[0]
###     NE_model = energy.shape[0]
###     NPA_model = pitch_angle.shape[0]
### 
###     # initialize observation operator array
###     obs_to_model_grid = numpy.zeros((NL_model,
###         NMLT_model,NE_model,NPA_model),dtype=int)
### 
###     # initialize observation array
###     obs = []
### 
###     # initialize observations coordinates array
###     obs_coordinates = []
### 
###     # initialize observations dates array
###     obs_dates = []
### 
###     # initialize number of observations in model grid-point
###     number_obs_model_grid = []
### 
###     # initialize observation counter
###     iobs = 0
###     for idate in numpy.arange(Ndates):
###         for iE in numpy.arange(NE):
###             for iPA in numpy.arange(NPA):
###                 if ( rbsp_pitch_angle[iPA] < numpy.max(pitch_angle) and
###                         rbsp_energy[iE] < numpy.max(energy) ):
###                     if (rbsp_flux[idate,iPA,iE] > 0.0):
### 
###                         # find index in model array
###                         idx_L = numpy.argmin(numpy.abs(Lshell-rbsp_L[idate]))
###                         idx_MLT = numpy.argmin(numpy.abs(MLT-rbsp_MLT[idate]))
###                         idx_energy = numpy.argmin(numpy.abs(energy-rbsp_energy[iE]))
###                         idx_pitch_angle = numpy.argmin(numpy.abs(pitch_angle-rbsp_pitch_angle[iPA]))
### 
###                         if (obs_to_model_grid[idx_L,idx_MLT,idx_energy,idx_pitch_angle] > 0):
### 
###                             pdb.set_trace()
### 
###                             # there is already an observation available at the grid-point, sum up to take the average 
###                             obs_idx = obs_to_model_grid[idx_L,idx_MLT,idx_energy,idx_pitch_angle]
###                             obs[obs_idx] = obs[obs_idx] + rbsp_flux[idate,iPA,iE]
###                             number_obs_model_grid[obs_idx] = number_obs_model_grid[obs_idx]+1.0
### 
###                         else:
### 
###                             # append the observation
###                             obs.append(rbsp_flux[idate,iPA,iE])
### 
###                             # append the observation coordinate
###                             coord = numpy.array([rbsp_L[idate], rbsp_MLT[idate],
###                                 rbsp_energy[iE], rbsp_pitch_angle[iPA]])
###                             obs_coordinates.append(coord)
### 
###                             # append the observation date
###                             obs_dates.append(rbsp_dates[idate])
### 
###                             # get obs index
###                             obs_to_model_grid[idx_L,idx_MLT,idx_energy,idx_pitch_angle] = iobs
###                             # get number of observations in the current grid
###                             number_obs_model_grid.append(1.0)
### 
###                             # increase counter
###                             iobs = iobs+1

# -----------------------------------------------------------------
#       compute observation operator
# -----------------------------------------------------------------

def get_observation_operator(
        species,ramscb_Lshell,ramscb_MLT,ramscb_energy,ramscb_pitch_angle,ramscb_flux,
        satobs_Lshell,satobs_MLT,satobs_energy,satobs_pitch_angle,satobs_flux):

    # -------------------------
    #   we are assuming the the dimensions of the satellite observations are as
    #   follows:
    #       satobs(dates,pitch_angle,energy)
    # -------------------------

    # -------------------------
    #   get dimensions from observations
    # -------------------------

    # number of observation dates within the given time interval
    Ndates = satobs_flux.shape[0]

    # valid energy ranges
    idx0 = numpy.where(satobs_energy >= numpy.min(ramscb_energy))[0]
    idx1 = numpy.where(satobs_energy <= numpy.max(ramscb_energy))[0]
    idxE = numpy.intersect1d(idx0,idx1)
    valid_satobs_energy = satobs_energy[idxE]

    # valid energy ranges
    idx0 = numpy.where(satobs_pitch_angle >= numpy.min(ramscb_pitch_angle))[0]
    idx1 = numpy.where(satobs_pitch_angle <= numpy.max(ramscb_pitch_angle))[0]
    idxPA = numpy.intersect1d(idx0,idx1)
    valid_satobs_pitch_angle = satobs_pitch_angle[idxPA]

    # get valid sat obs
    valid_satobs_flux = satobs_flux[:,:,idxE]
    valid_satobs_flux = valid_satobs_flux[:,idxPA,:]

    # initialize the array
    satobs_OF = numpy.zeros((Ndates,valid_satobs_energy.shape[0]),dtype=float)

    # -------------------------
    #   compute omnidirectional flux
    # -------------------------

    for iobs,obs_flux in enumerate(valid_satobs_flux):

        # identify index in L-shell
        idx_L = numpy.argmin(numpy.abs(ramscb_Lshell-satobs_Lshell[iobs]))

        # get index for closest ram-scb MLT
        idx_MLT = numpy.argmin(numpy.abs(ramscb_MLT-satobs_MLT[iobs]))

        for iE,E in enumerate(valid_satobs_energy):

            idx_valid_flux = numpy.where(valid_satobs_flux[iobs,:,iE] > 0.0)[0]

            if len(idx_valid_flux) > 1:

                # get index for closest ram-scb MLT
                idx_E = numpy.argmin(numpy.abs(ramscb_energy-E)) 

                # initialize PA array
                model_to_obs_pitch_angle = []

                # loop over the pitch angle
                for iPA,PA in enumerate(valid_satobs_pitch_angle):

                    # get index for closest ram-scb MLT
                    idx_PA = numpy.argmin(numpy.abs(ramscb_pitch_angle-PA)) 
                    model_to_obs_pitch_angle.append(idx_PA)

                # convert to numpy array
                model_to_obs_pitch_angle = numpy.array(model_to_obs_pitch_angle)
                model_to_obs_pitch_angle = numpy.unique(model_to_obs_pitch_angle)
                
                # -------------------------
                # integrate over the pitch angle for omnidirectional flux
                # -------------------------

                # compute delta in pitch angle
                dx = valid_satobs_pitch_angle[1:]-valid_satobs_pitch_angle[:-1]
                dx = numpy.pi/180.0*dx

                # form the function to integrate
                f = numpy.sin(numpy.pi/180.0*valid_satobs_pitch_angle)
                f = f*valid_satobs_flux[iobs,:,iE]

                pdb.set_trace()
                tmpx = numpy.pi/180.0*valid_satobs_pitch_angle
                tmp = integrate.cumtrapz(f,tmpx)[-1]

                f[0] = 1.0/2.0*dx[0]*f[0]
                for k in numpy.arange(1,valid_satobs_pitch_angle.shape[0]-1):
                    f[k] = 1.0/2.0*(dx[k-1]+dx[k])*f[k]
                f[-1] = 1.0/2.0*dx[-1]*f[-1]

                integral_trapezoid = numpy.sum(f)

                pdb.set_trace()

    return Ndates


# -----------------------------------------------------------------
#       interpolate the ramscb omnidirectional flux to RBSP omni. flux
# -----------------------------------------------------------------

def interpolate_omnidirectional_flux(ramscb_Lshell,ramscb_MLT,ramscb_energy,ramscb_OF,
                                     satobs_Lshell,satobs_MLT,satobs_energy,satobs_OF):

    #pdb.set_trace()

    # valid energy ranges
    idx0 = numpy.where(satobs_energy >= numpy.min(ramscb_energy))[0]
    idx1 = numpy.where(satobs_energy <= numpy.max(ramscb_energy))[0]
    idxE = numpy.intersect1d(idx0,idx1)
    valid_satobs_energy = satobs_energy[idxE]

    # number of observation whithin the time interval
    ndates = satobs_OF.shape[0]

    ramscb_NE = ramscb_energy.shape[0]
    satobs_NE = satobs_energy.shape[0]

    # initialize interpolation arrays
    interpolated_OF1 = numpy.zeros((ndates,ramscb_NE),dtype=float)
    interpolated_OF = numpy.zeros((ndates,satobs_NE),dtype=float)

    # main observation loop
    for iobs,obs_OF in enumerate(satobs_OF):

        # ------------------------------
        #   interpolation for L-shell and MLT
        # ------------------------------

        # identify index in L-shell
        idx_L = numpy.argmin(numpy.abs(ramscb_Lshell-satobs_Lshell[iobs]))

        # get index for closest ram-scb MLT
        idx_MLT = numpy.argmin(numpy.abs(ramscb_MLT-satobs_MLT[iobs]))

        # determine where point is located in radius
        if (satobs_Lshell[iobs] > ramscb_Lshell[idx_L]):
            L1 = idx_L
            L2 = idx_L+1
        else:
            L1 = idx_L-1
            L2 = idx_L

        # max radius
        if ( (L2 <= ramscb_Lshell.shape[0]-1) 
                and (L1 >= 0) ):

            # determine where point is located in angle
            if (satobs_MLT[iobs] > ramscb_MLT[idx_MLT]):
                MLT1 = idx_MLT
                MLT2 = idx_MLT+1
            else:
                MLT1 = idx_MLT-1
                MLT2 = idx_MLT
            # cyclic angle
            if (MLT2 > ramscb_MLT.shape[0]-1):
                MLT1 = ramscb_MLT.shape[0]-1
                MLT2 = 0
            if (MLT1 < 0):
                MLT1 = ramscb_MLT.shape[0]-1
                MLT2 = 0

            # first coordinate (L1,MLT1)
            OF11 = ramscb_OF[species,L1,MLT1,:]

            # second coordinate (L1,MLT2)
            OF12 = ramscb_OF[species,L1,MLT2,:]

            # third coordinate (L2,MLT1)
            OF21 = ramscb_OF[species,L2,MLT1,:]

            # fourth coordinate (L2,MLT2)
            OF22 = ramscb_OF[species,L2,MLT2,:]

            # L-shell array
            Lshell = numpy.array([ramscb_Lshell[L1],ramscb_Lshell[L2]],
                    dtype=float)

            # MLT array
            MLT = numpy.array([ramscb_MLT[MLT1],ramscb_MLT[MLT2]],
                    dtype=float)

            for iE,E in enumerate(ramscb_energy):

                OF = numpy.array([OF11[iE],OF12[iE],OF21[iE],OF22[iE]],dtype=float)
                interpolated_OF1[iobs,iE] = interpolation.BilinearInterpolation(Lshell,MLT,OF,satobs_Lshell[iobs],satobs_MLT[iobs])

                # ---------------------------
                #       DEBUG
                '''
                fig = pylab.figure()
                ax = fig.gca(projection='3d')
                X,Y = numpy.meshgrid(Lshell,MLT)
                Z = numpy.array([[OF11[iE],OF21[iE]],[OF12[iE],OF22[iE]]])
                surf = ax.plot_surface(X,Y,Z,cmap=cm.coolwarm,
                       linewidth=0,antialiased=False,alpha=0.5)
                spoint = ax.scatter(satobs_Lshell[iobs],satobs_MLT[iobs],interpolated_OF1[iobs,iE],color='green')
                ax.set_xlabel('L-shell')
                ax.set_ylabel('MLT')
                ax.set_zlabel('omni. flux')
                pylab.show()
                '''
                # ---------------------------

        # ------------------------------
        #   interpolation for energy
        # ------------------------------

        #pdb.set_trace()

        for iE,E in enumerate(satobs_energy):

            if (satobs_OF[iobs,iE] > 0.0):

                # check we are within a valid energy domain for RAM-SCB
                if ( (E >= numpy.min(ramscb_energy)) and 
                        (E <= numpy.max(ramscb_energy)) ):

                    # find closest ram energy
                    idx_E = numpy.argmin(numpy.abs(ramscb_energy-E))

                    if (E > ramscb_energy[idx_E]):
                        idx_E1 = idx_E
                        idx_E2 = idx_E+1
                    else:
                        idx_E1 = idx_E-1
                        idx_E2 = idx_E

                    # get left and right energy grid-points at
                    # the RAM-SCBE grid
                    E1 = ramscb_energy[idx_E1]
                    E2 = ramscb_energy[idx_E2]

                    # get omnidirectional flux at left and right energy grid-points at
                    # the RAM-SCBE grid
                    OF1 = interpolated_OF1[iobs,idx_E1]
                    OF2 = interpolated_OF1[iobs,idx_E2]

                    # do interpolation in log space (?)
                    lE1 = numpy.log(E1)
                    lE2 = numpy.log(E2)
                    lOF1 = numpy.log(OF1)
                    lOF2 = numpy.log(OF2)
                    lE = numpy.log(E)

                    x = numpy.array([lE1,lE2])
                    f = numpy.array([lOF1,lOF2])

                    interpolated_OF[iobs,iE] = numpy.exp(numpy.interp(lE,x,f))

        # ---------------------------
        #       DEBUG
        '''
        fig_size=(16.0,6.0)
        figparams = {
                  'backend': 'ps',
                  'font.size' : 28,
                  'axes.labelsize': 28,
                  'font.size': 28,
                  'xtick.labelsize': 24,
                  'ytick.labelsize': 24,
                  'legend.fontsize' : 'large',
                  'legend.labelspacing' : 0.3,
                  'text.usetex': True,
                  'figure.figsize': fig_size
                  }
        pylab.rcParams.update(figparams)
        fig = pylab.figure()

        ax1 = fig.add_subplot(121)
        idx = numpy.where(interpolated_OF[iobs,:]>0.0)[0]
        ax1.plot(ramscb_energy,ramscb_OF[species,idx_L,idx_MLT,:],'b-')
        ax1.scatter(satobs_energy[idx],interpolated_OF[iobs,idx],color='blue')
        ax1.scatter(satobs_energy[idx],satobs_OF[iobs,idx],color='green')
        ax1.set_ylim(1.0e-1,1.0e+8)
        ax1.set_yscale('log')
        ax1.set_xlabel('energy')
        ax1.set_ylabel('omni. flux')

        ax2 = fig.add_subplot(122)
        ax2.plot(ramscb_energy,ramscb_OF[species,idx_L,idx_MLT,:],'b-')
        ax2.scatter(satobs_energy[idx],interpolated_OF[iobs,idx],color='blue')
        ax2.scatter(satobs_energy[idx],satobs_OF[iobs,idx],color='green')
        ax2.set_xlabel('energy')
        ax2.set_ylabel('omni. flux')

        pylab.show()
        pdb.set_trace()
        '''
        # ---------------------------

    return interpolated_OF
