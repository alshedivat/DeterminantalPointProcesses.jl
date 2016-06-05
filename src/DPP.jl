__precompile__(true)

module DPP

import Base.Random
import Base.LinAlg: Eigen, Symmetric
import Distributions: Multinomial, rand

export
    # point process types
    DeterminantalPointProcess,

    # mehtods
    logpmf,             # log probability mass
    logpmf!,            # evaluate log pmf to provided storage
    pmf,                # probability mass function (DiscreteDistribution)
    # sampler,            # create a Sampler object for efficient samples
    rand


### source files

include("types.jl")
include("methods.jl")
include("utils.jl")

end # module
