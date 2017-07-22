using Distributions
using Parameters
using HamiltonianABC
using ContinuousTransformations
using Plots
using StatsBase
using StatPlots                 # for kernel density
import HamiltonianABC: logdensity, simulate!
using Base.Test
using PlotlyJS
plotlyjs()

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


"""
    simulate_simultaneous(β, X, us)

Take in the prameter (β), X and errors us, give back the endogenous variables of the system (Y and C).
"""
function simulate_simultaneous(β, X, us)
    N = length(us)
    C = Vector{Float64}
    Y = (X .+ us) / (1 - β)
    C = β * Y .+ us
    return (C, Y)
end

struct ToySimultaneousModel
    "observed consumption"
    Cs::Vector{Float64}
    "observed output"
    Ys::Vector{Float64}
    "non-comsumption"
    Xs::Vector{Float64}
    "distribution of Xs"
    dist_x
    "prior for β"
    prior_β
    "Normal(0,τ) draws for simulation, where τ is fixed"
    us::Vector{Float64}
    "distribution of us"
    dist_us
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
    err = (y - x * β)
    σ_2 = mean(abs2, err)
    return β, σ_2
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

## logdensity uses the following auxiliary model:
## C_t ∼  N(β_1 + β_2 * X_t, √σ_2)
function logdensity(pp::ToySimultaneousModel, β)
    @unpack Cs, Ys, Xs, prior_β, us = pp
    logprior = logpdf(prior_β, β[1])

    ## Generating the data
    C, Y = simulate_simultaneous(β[1], Xs, us)
    # OLs estimatation, regressing C on [1 X]
    est, σ_2 = OLS(C, [ones(length(us)) (Y - C)])
    β_1, β_2 = est

    log_likelihood = 0
    for i in 1:length(Cs)
        log_likelihood += logpdf(Normal(β_1 + β_2 * (Ys[i] - Cs[i]), √(σ_2)), Cs[i])
    end

    return(logprior + log_likelihood)
end


# Trial
β = 0.9
C, Y = simulate_simultaneous(β, rand(Normal(100, 3), 100), rand(Normal(0, 5), 100))
pp = ToySimultaneousModel(C, Y, Uniform(0, 1), Normal(100, 3), Normal(0, 5), 1000)
# variance is hand-tuned to get a ≈ 0.6
chain, a = mcmc(RWMH(diagm([4e-7])), pp, [0.1], 5000)
result = vcat(chain[2500:end]'...)

# Analysis with plotting
plt = plot(density(result), label = "posterior", title = "β")
plot!(plt, a -> pdf(pp.prior_β, a), linspace(0, 1, 100), label = "prior")
vline!(plt, [β], label = "true value")

@testset "simultaneous equation" begin
    @test mean(result) ≈ β rtol = 0.05
end
