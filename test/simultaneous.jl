using ArgCheck
using Distributions
using Parameters
using DynamicHMC
using StatsBase
using StatPlots                 # for kernel density
import DynamicHMC: logdensity, loggradient, length
using Base.Test
using ForwardDiff
using ReverseDiff
using Plots
using ContinuousTransformations
using ProfileView
import ForwardDiff: Dual

plotlyjs()                      # Tamas likes this, optional

############################################################################
## Simple simultaneous equations
############################################################################

## True model:
##  C_t = β * Y_t + us_t where  us_t ∼ N(0, τ)   (1)
##  Y_t = C_t + X_t                              (2)

## C_t is the consumption at time t
## X_t is the non-consumption at time t
## Y_t is the output at time t
## So output is used for consumption and non-consumption

## We have simultaneous equations as c_t depends on Y_t
## And Y_t depends on C_t as well


## Only X_t is exogenous in this model, Y_t and C_t are endogenous

##############################################################################
## Functions needed for the model
##############################################################################

"an array of ForwardDiff.Dual, for testing type stability"
duals(dims...) = fill(Dual(1.0, 2.0), dims...)

"""
    simulate_simultaneous(β, X, us)

Take in the parameter β, X and errors us, give back the endogenous variables of the system (Y and C).
"""
function simulate_simultaneous(β, X, us)
    N = length(us)
    C = Vector{eltype(X)}(N)
    Y = (X .+ us) / (1 - β)
    C = β * Y .+ us
    return (C, Y)
end


struct ToySimultaneousModel{T, Prior_β, Dist_x, Dist_us}
    "observed consumption"
    Cs::Vector{T}
    "observed output"
    Ys::Vector{T}
    "non-comsumption"
    Xs::Vector{T}
    "distribution of Xs"
    dist_x::Dist_x
    "prior for β"
    prior_β::Prior_β
    "Normal(0,τ) draws for simulation, where τ is fixed"
    us::Vector{T}
    "distribution of us"
    dist_us::Dist_us
end


"""
    ToySimultaneousModel(Cs, Ys, prior_β, dist_x, dist_us, M)

Convenience constructor for ToySimultaneousModel.
Take in the observed data, the prior, and number of simulations (M).
"""
function ToySimultaneousModel(Cs, Ys, prior_β, dist_x, dist_us, M)
    ToySimultaneousModel(Cs, Ys, rand(dist_x, M), dist_x, prior_β, rand(dist_us, M), dist_us)
end

"""
    OLS(y, x)

Take in the dependant variable (y) and the regressor (x), give back the estimated coefficients (β) and the variance (σ_2).
"""
function OLS(y, x)
    β = x \ y
    err = y - x * β
    σ_2 = mean(abs2, err)
    β, σ_2
end

"""
    simulate!(pp::ToySimultaneousModel)

Updates the shock and the non-consumption variable of the model.
"""
function simulate!(pp::ToySimultaneousModel)
    @unpack Xs, us, dist_x, dist_us = pp
    rand!(dist_x, Xs)
    rand!(dist_us, us)
end



bridge_trans(dist::Uniform) = bridge(ℝ, Segment(dist.a, dist.b))

## logdensity uses the following auxiliary model:
## C_t ∼  N(β_1 + β_2 * X_t, √σ_2)

function logdensity(pp::ToySimultaneousModel, β)
    @unpack Cs, Ys, Xs, prior_β, us = pp
    trans = bridge_trans(prior_β)
    value_and_logjac = trans(β, LOGJAC)
    β = first(value_and_logjac)
    logprior = logpdf(prior_β, β[1]) + last(value_and_logjac)
    Ones = ones(length(us))
    ## Generating the data
    C, Y = simulate_simultaneous(β[1], Xs, us)
    # OLs estimatation, regressing C on [1 X]
    XX = hcat(Ones, Y-C)
    est, σ_2 = OLS(C, XX)

    log_likelihood = sum(logpdf.(Normal(0, √σ_2), Cs - [ones(length(Cs)) Ys-Cs] * est))
    return(logprior + log_likelihood)
end


#################################################################################
## Toymodel
#################################################################################

β = 0.9
C, Y = simulate_simultaneous(β, rand(Normal(100, 3), 100), rand(Normal(0, 5), 100))
pp = ToySimultaneousModel(C, Y, Uniform(0, 1), Normal(100, 3), Normal(0, 5), 1000)
β₀ = inv(bridge_trans(pp.prior_β))(β)


## loggradient with ForwardDiff
loggradient(pp::ToySimultaneousModel, x) = ForwardDiff.derivative(y->logdensity(pp, y), x)
loggradient(pp, β₀)
## with ReverseDiff
#loggradient_(pp::ToySimultaneousModel, x) = ReverseDiff.deriv(y->logdensity(pp, y), x)
#loggradient_(pp, [β])

Base.length(::ToySimultaneousModel) = 1
## defining RNG
const RNG = srand(UInt32[0x23ef614d, 0x8332e05c, 0x3c574111, 0x121aa2f4])

sample, tuned_sample = NUTS_tune_and_mcmc(RNG, pp, 1000)

NN = ceil(Int, length(sample) - (size(sample,1)) * 0.5 )
sample_β = Vector(NN)
for i in 1:NN
sample_β[i] = [t(param) for (t,param) in zip(bridge_trans(pp.prior_β), sample[i+NN].q)]
end

plt = plot(density(sample_β), label = "posterior", title = "β")
plot!(plt, a -> pdf(pp.prior_β, a), linspace(0, 1, 100), label = "prior")
vline!(plt, [β], label = "true value")



#################################################################################
## type-stability
#################################################################################


# type-stable, works fine
@code_warntype simulate_simultaneous(0.8, ones(5), randn(5))
@code_warntype simulate_simultaneous(0.8, duals(5), duals(5))

@code_warntype OLS(ones(3), ones(3,3))
@code_warntype OLS(duals(3), duals(3,3))

## type-stable
@code_warntype logdensity(pp, β)
