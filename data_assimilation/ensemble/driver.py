import numpy
import os
import parameter
import ensembles
import msubscript

# declare ensemble object
ens = ensembles.ensemble()

# read ensemble imput information
ens.read_input()

# create ensemble directory structure
ens.create_directories()

# read and generate ensemble input files
ens.generate_parameters()
