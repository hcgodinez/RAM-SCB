import numpy
import pdb
import pylab
import copy

def EnKF(xm,yo,R,H,S=[],inflation=0.0):

    # dimensions
    Nvar,Nens = xm.shape
    Nobs = yo.shape[0]

    # resample observation perturbations
    yo_pert = numpy.random.multivariate_normal(yo,R,Nens)
    yo_pert = yo_pert.transpose()

    # get ensemble in observation space
    hm = numpy.dot(H,xm)

    # model covariance matrix
    P = numpy.cov(xm)

    # apply localization through S
    if (len(S)>0):
        P = P*S

    # inflation
    if (inflation < 0.0):
        # do apaptive inflation
        inflation = adaptive_inflation(hm,yo,R)
        if (abs(inflation) < 1.0):
            inflation = 1
        P = inflation*P
    elif (inflation > 0.0):
        P = inflation*P

    # matrices for Kalman gain matrix
    PHt = numpy.dot(P,H.T)
    HPHt = numpy.dot(H,PHt)

    # compute the Kalman gain matrix
    C = HPHt+R
    D = yo_pert-hm
    z = numpy.linalg.solve(C,D)

    # analysis
    xa = xm + numpy.dot(PHt,z)

    return xa

def EnKF2(xm,hm,yo,R,Q=[],inflation=0):

    '''

    Parameters: 

    xm : numpy array of floats, dimension is nvar x nens
         ensemble matrix where each row are the variables and each column is an
         ensemble member

    hm : numpy array of floatx, dimension is nobs x nens
         ensemble matrix that is model in observation space, each row are the
         variables and each column is an ensemble member

    yo : numpy array of floats, dimension is nobs x 1
         vector of observations

    R :  numpy array of floats, dimension is nobs x nobs
         observations covariance matrix

    Q :  (optional) numpy array of floats, dimension nvar x nvar
         mask array for the model covariance matrix

    inflation : (optional) float
                inflation factor for the model covariance matrix, 
                if inflation=-1 then we use adaptive inflation

    Returns:

    xa : numpy array of floats, dimension is nvar x nens
         analysis ensemble matrix from the ensemble Kalman filter (EnKF)

    '''

    # dimensions
    Nvar,Nens = xm.shape
    Nobs = yo.shape[0]

    # resample observation perturbations
    yo_pert = numpy.random.multivariate_normal(yo,R,Nens)
    yo_pert = yo_pert.transpose()

    # do inflation
    if inflation < 0:
        # do apaptive inflation
        inflation = adaptive_inflation(hm,yo,R)
        if (abs(inflation) < 1.0):
            inflation = 1
    #endif

    # form matrices
    xm_avg = numpy.average(xm,axis=1)
    hm_avg = numpy.average(hm,axis=1)
    PHt = numpy.zeros((Nvar,Nobs),dtype=float)
    HPHt = numpy.zeros((Nobs,Nobs),dtype=float)
    for iens in range(Nens):
        d1 = xm[:,iens]-xm_avg
        d2 = hm[:,iens]-hm_avg
        PHt = PHt + (1.0/(float(Nens)-1.0))*numpy.outer(d1,d2)
        HPHt = HPHt + (1.0/(float(Nens)-1.0))*numpy.outer(d2,d2)

    #pdb.set_trace()

    if (inflation != 0):
        PHt = inflation*PHt
        HPHt = inflation*HPHt

    # compute the Kalman gain matrix
    C = HPHt+R
    D = yo_pert-hm
    K = numpy.linalg.solve(C,D)
    z = numpy.dot(PHt,K)

    # analysis
    xa = xm + z

    return xa

def EnKF_hm(xm,hm,yo,R):

    # dimensions
    Nvar,Nens = xm.shape
    Nobs = yo.shape[0]

    # resample observation perturbations
    yo_pert = numpy.random.multivariate_normal(yo,R,Nens)
    yo_pert = yo_pert.transpose()

    # form matrices
    xm_avg = numpy.average(xm,axis=1)
    hm_avg = numpy.average(hm,axis=1)
    PHt = numpy.zeros((Nvar,Nobs),dtype=float)
    HPHt = numpy.zeros((Nobs,Nobs),dtype=float)
    for iens in range(Nens):
        d1 = xm[:,iens]-xm_avg
        d2 = hm[:,iens]-hm_avg
        PHt = PHt + (1.0/(float(Nens)-1.0))*numpy.outer(d1,d2)
        HPHt = HPHt + (1.0/(float(Nens)-1.0))*numpy.outer(d2,d2)

    # REVIEW!!!!!!!!!!!!!!!
    # compute the Kalman gain matrix
    C = HPHt+R
    D = yo_pert-hm
    z = numpy.linalg.solve(C,D)

    # analysis
    xa = xm + numpy.dot(PHt,z)

    return xa

def adaptive_inflation(hm,yo,R,yo_weights=[]):

    # compute innovation
    d = yo-numpy.average(hm,axis=1)

    # compute covanriance in observation space
    HPHt = numpy.cov(hm)

    if (len(yo_weights)>0):
        d = d*yo_weights
        R = R/yo_weights
        HPHt = numpy.dot(HPHt,numpy.diag(yo_weights))

    alpha = (numpy.dot(d,d)-numpy.trace(R))/numpy.trace(HPHt)

    return alpha


# ====================================================================

# Compute the fifth-order support function value from Gaspari & Cohn paper:
#
# Gaspari, G. and Cohn, S. E. (1999), Construction of correlation functions in
# two and three dimensions. Q.J.R. Meteorol. Soc., 125: 723-757
#
# Authors: Humberto C. Godinez
#     hgodinez@lanl.gov
#     T-5: Applied Mathematics and Plasma Physics
#     Los Alamos National Laboratory
#
def gasparicohn(a,b):

    if (b < 0.0):
        print('distance is less than zero, '+str(b))
        GC = -1.0

    if (b <= a):
        GC = -1.0/4.0*(b/a)**5 + 1.0/2.0*(b/a)**4 + \
                5.0/8.0*(b/a)**3 - 5.0/3.0*(b/a)**2 + 1.0
    elif ( (a < b) and (b < 2.0*a)):
        # at the very end, when b is very close to 2a, we get wiggles that dip
        # below zero. Hence, we take the absolute value of the function
        GC = abs(1.0/12.0*(b/a)**5 - 1.0/2.0*(b/a)**4 + \
                5.0/8.0*(b/a)**3 + 5.0/3.0*(b/a)**2 - \
                5.0*(b/a) + 4.0 - 2.0/3.0*(b/a)**(-1))
    else:
        GC = 0.0

    return GC


def EnKFSVD(xm,yo,R,H,Ns=4,inflation=0,taper=False):

    # get dimensions
    Nvar,Nens = xm.shape
    Nobs = yo.shape[0]

    # get ensemble in observation space
    hm = numpy.dot(H,xm)

    # do inflation
    if inflation < 0:
        # do apaptive inflation
        xm_avg = numpy.average(xm,axis=1)
        hm_avg = numpy.average(hm,axis=1)
        inflation = adaptive_inflation(hm,yo,R)
    #endif

    # compute singular value decomposition
    U,S,V = numpy.linalg.svd(xm,full_matrices=False)

    # get basis
    B = 1.0/numpy.sqrt(Nens-1)*numpy.dot(U[:,:Ns],numpy.diag(S[:Ns]))

    # get transformation matrix
    T = numpy.linalg.solve(numpy.dot(B.T,B),B.T)

    # get weights
    w = numpy.dot(T,xm)

    # project onto basis
    HB = numpy.dot(H,B)

    # get weights in obs space
    hw = numpy.dot(HB,w)

    #pdb.set_trace()

    # resample observation perturbations
    yo_pert = numpy.random.multivariate_normal(yo,R,Nens)
    yo_pert = yo_pert.transpose()

    # compute covariance matrix of weights for the EnKF
    P = numpy.cov(w)
    if (inflation != 0):
        P = inflation*P

    # matrices
    HP = numpy.dot(HB,P)
    PHt = numpy.dot(P,HB.T)
    HPHt = numpy.dot(HB,PHt)

    if taper:
        for iobs in numpy.arange(Nobs):
            R[iobs,iobs] = R[iobs,iobs]/yo_weights[iobs]

    # compute Kalman gain matrix
    C = HPHt+R
    C = numpy.linalg.solve(C,HP)
    K = C.T
    D = yo_pert-hw

    # analysis weight
    wa = w + numpy.dot(K,D)

    # get analysis in state space
    xa = numpy.dot(B,wa)

    return xa

def LETKF(xm,yo,R,H,inflation=0):

    #pdb.set_trace()

    # get dimensions of problem
    Nvar = xm.shape[0]
    Nens = xm.shape[1]
    Nobs = yo.shape[0]

    # set analysis to background
    xa = copy.copy(xm)

    # get ensemble in observation space
    hm = numpy.dot(H,xm)

    # define Xb
    Xb = xm - numpy.average(xm,axis=1)[:,numpy.newaxis]

    # define Yb
    Yb = hm - numpy.average(hm,axis=1)[:,numpy.newaxis]

    # ----------------------------------------
    # compute C = (Yb)^T*R^(-1)
    # ----------------------------------------

    C = numpy.dot(Yb.T,numpy.linalg.inv(R))

    # ----------------------------------------
    # inflation
    # ----------------------------------------

    if (inflation < 0.0):

        # do apaptive inflation
        inflation = adaptive_inflation(hm,yo,R)

        if (abs(inflation) < 1.0):
            inflation = 1.0

        # localized inflation
        rho = inflation

    elif (inflation > 0.0):

        # localized inflation
        rho = inflation

    # ----------------------------------------
    # compute Pa = [ (Nens-1)*I/rho + C Yb ]^(-1)
    #   use eigendecomposition or SVD
    # ----------------------------------------

    Pa = (Nens-1.0)/rho*numpy.eye(Nens) + numpy.dot(C,Yb)

    # compute SVD
    U,S,Vt = numpy.linalg.svd(Pa,full_matrices=True)

    # compute inverse of singular value matrix
    Sinv = numpy.diag(1.0/S)

    # compute inverse with SVD
    Pa = numpy.dot(U,Sinv)
    Pa = numpy.dot(Pa,Vt)

    # ----------------------------------------
    # compute Wa = [ (Nens-1)*Pa ]^(1/2)
    # ----------------------------------------

    # compute square root using singular vectors
    alpha = numpy.sqrt(Nens-1)
    Ssqrt = numpy.sqrt(Sinv)
    Wa = numpy.dot(U,Ssqrt)
    Wa = alpha*numpy.dot(Wa,Vt)

    # ----------------------------------------
    # compute the analysis weight vector
    # wa_avg = Pa*C*(yo_local - yo_ens_avg_local)
    # ----------------------------------------

    # compute deviations
    z = yo - numpy.average(hm,axis=1)

    # compute average weight
    wa_avg = numpy.dot(Pa,C)
    wa_avg = numpy.dot(wa_avg,z)

    # compute ensemble weights
    for iens in numpy.arange(Nens):
        Wa[:,iens] = Wa[:,iens]+wa_avg

    # ----------------------------------------
    # compute the analysis
    # ----------------------------------------

    # compute average background state
    xm_avg = numpy.average(xm,axis=1)

    # initialize the analysis array
    xa = numpy.zeros_like(xm)

    # loop to compute analysis
    for iens in numpy.arange(Nens):
        xa[:,iens] = xm_avg + numpy.dot(Xb,Wa[:,iens])

    return xa
