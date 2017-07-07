using HamiltonianABC
using Base.Test
using Parameters
using Distributions
using Cubature
using StatsBase

import HamiltonianABC: logdensity, simulate!

# use consistent random seed for unit tests
srand(UInt32[0x8c736cc0, 0x63556b2b, 0x808f767c, 0xc912a668])

include("utilities.jl")
include("test-mcmc.jl")
include("test-toyexponential.jl")
include("EM_GaussianMixtureModel.jl")
