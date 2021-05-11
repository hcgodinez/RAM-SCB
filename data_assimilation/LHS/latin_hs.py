# s=latin_hs(xmean,xsd,nsample,nvar)
# LHS from normal distribution, no correlation
# method of Stein
# Stein, M. 1987. Large Sample Properties of Simulations Using Latin Hypercube Sampling. 
#                 Technometrics 29:143-151
# Input:
#   xmean   :  mean of data (1,nvar)
#   xsd     : std.dev of data (1,nvar)
#   nsample : no. of samples
#   nvar    : no. of variables
# Output:
#   s       : random sample (nsample,nvar)
#
# Uses Peter Acklam inverse normal CDF
#
#   Budiman (2003)
# References:
# Iman, R. L., and W. J. Conover. 1980. Small Sample Sensitivity Analysis Techniques for Computer Models, 
# with an Application to Risk Assessment.Communications in Statistics: Theory and Methods A9: 1749-1874
# McKay, M. D., W. J. Conover and R. J. Beckman. 1979.A Comparison of Three Methods for Selecting Values
# of Input Variables in the Analysis of Output from a Computer Code. Technometrics 21: 239-245
#

# libraries needed
import numpy
import ltqnorm

def latin_hs(xmean,xsd,nsample,nvar):

    ran = numpy.random.rand(nsample,nvar)
    s = numpy.zeros((nsample,nvar),dtype=float)

    idx = numpy.arange(1,nsample+1)

    # method of Stein
    for j in range(nvar):
        numpy.random.shuffle(idx)
        P = (idx - ran[:,j])/numpy.float(nsample)
        print P
        s[:,j] = xmean[j] + ltqnorm.ltqnorm(P) * xsd[j]
    #end for

    return s

