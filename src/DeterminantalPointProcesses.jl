######################################################################
# DeterminantalPointProcesses.jl
# Determinantal Point Processes in Julia
# http://github.com/alshedivat/DeterminantalPointProcesses.jl
# MIT Licensed
######################################################################

__precompile__(true)

module DeterminantalPointProcesses

using LinearAlgebra
using Random: rand, bitrand, AbstractRNG, MersenneTwister
using SharedArrays
using Distributed
import Base: rand

export
    # point process types and aliases
    DeterminantalPointProcess, DPP,

    # mehtods
    logpmf,             # log probability mass function
    pmf,                # probability mass function
    rand,               # generate samples
    randmcmc            # generate samples using MCMC


### source files

# types
include("types.jl")

# methods
include("fit.jl")
include("prob.jl")
include("rand.jl")

# utilities
include("utils.jl")

end # module
