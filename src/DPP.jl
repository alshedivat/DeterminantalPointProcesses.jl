__precompile__(true)

module DPP

import Base.LinAlg: Eigen, Symmetric
import Base.Random: rand

export
    # point process types
    DeterminantalPointProcess,

    # mehtods
    logpmf,             # log probability mass function
    pmf,                # probability mass function
    rand,               # generate samples
    randmcmc            # generate samples using MCMC


### source files

include("types.jl")
include("methods.jl")
include("utils.jl")

end # module
