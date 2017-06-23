using HamiltonianABC
using Base.Test
using Parameters
using Distributions
using Cubature
using StatsBase

import HamiltonianABC: logdensity, simulate!

# use consistent random seed for unit tests
srand(UInt32[0x8c736cc0, 0x63556b2b, 0x808f767c, 0xc912a668])

######################################################################
# utility functions
######################################################################

"""
A wrapper for sampling from a given log density.

Useful for testing MCMC.
"""
struct DensityWrapper{T}
    "Log density function."
    ℓ::T
end

logdensity(dw::DensityWrapper, θ) = dw.ℓ(θ)

simulate!(::DensityWrapper) = nothing

"""
    normalized_density(logdensity, x, xmin, xmax)

Return a density function from an unnormalize `logdensity`.

`x` should be a point where the density had a value that is not far
form the typical region (eg the mode or the mean will be fine). This
is used to correct the value *before* calculating the integral, for
better accuracy.

`xmin` and `xmax` are integration bounds.
"""
function normalized_density(logdensity, x, xmin, xmax)
    c = logdensity(x)
    C, _ = hquadrature(x -> exp(logdensity(x)-c), xmin, xmax)
    c += log(C)
    x -> exp(logdensity(x)-c)
end

"""
    test_cdf(density, xs, xmin[; ps, atol])

Compare the random values `xs` to the given `density` function, which
has support srting at `xmin`. Comparison is made at `ps`, with
absolute tolerance `atol` (in probability space). Useful for testing
distributions.
"""
function test_cdf(density, xs, xmin; ps = 0.1:0.1:0.9, atol = 0.05)
    for (p, x) in zip(ps, quantile(xs, ps))
        p′, _ = hquadrature(density, xmin, x)
        @test p ≈ p′ atol = atol
    end
end

######################################################################
# test counter and mcmc with a standard normal
######################################################################

"Normal(0,1) density. Keeps track of simulations. For unit tests."
mutable struct StandardNormalModel
    counter::Int
end

logdensity(::StandardNormalModel, θ) = -0.5*θ[1]^2

simulate!(p::StandardNormalModel) = p.counter += 1

@testset "standard normal" begin
    p = StandardNormalModel(0)
    chain, a = mcmc(RWMH(diagm([0.5])), p, [0.0], 1000)
    @test p.counter == 1000
    xs = first.(chain)
    @test mean(xs) ≈ 0 atol = 0.1
    @test std(xs) ≈ 1 atol = 0.2
end

######################################################################
# test mcmc with a transformed Γ
######################################################################

@testset "log transformed Γ" begin
    dist = Gamma(5.0, 1.0)
    # sample log(x), where x ∼ distribution, apply the Jacobian correction
    p = DensityWrapper(θ -> logpdf(dist, exp(θ[1])) + θ[1])

    chain, a = mcmc(RWMH(diagm([2.0])), p, [log(5.0)], 10000)
    xs = exp.(first.(chain))

    @test mean(xs) ≈ mean(dist) atol = 0.2
    for q in 0.1:0.1:0.9
        @test quantile(xs, q) ≈ quantile(dist, q) atol = 0.2
    end
end

######################################################################
# Exponential with normal auxiliary model
######################################################################

"""
The true model is y ∼ Exponential(λ), IID, where λ is the scale.

The auxiliary model is y ∼ N(μ, σ²), with statistics ϕ = (μ, σ).

The prior is λ ∼ Uniform(A,B) prior.
"""
struct ToyExponentialModel
    "observed data"
    y::Vector{Float64}
    "lower boundary of the uniform prior distribution"
    A::Float64
    "upper boundary of the uniform prior distribution"
    B::Float64
    """
    Uniform(0,1) draws, mapped to exponential draws using the
    CDF. For the empty vector, the limiting statistics are used.
    """
    ϵ::Vector{Float64}
end

"""
Convenience constructor for ToyExponentialModel, true value `λ`, `N`
draws, `M` simulated values.
"""
function ToyExponentialModel(λ::Float64, N::Int, M::Int)
    ToyExponentialModel(rand(Exponential(λ), N), 0.0, 2*λ, rand(M))
end
 
function logdensity(p::ToyExponentialModel, θ)
    @unpack y, A, B, ϵ = p
    logλ, = θ
    λ = exp(logλ)
    # log Jacobian is ``log(|exp(logλ)|) = logλ``, hence the ``+ logλ``.
    logprior = logpdf(Uniform(A, B), λ) + logλ
    dist = Exponential(λ)
    if isempty(ϵ)
        μ = mean(dist)
        σ² = var(dist)
    else
        μ, σ² = mean_and_var(quantile.(dist, ϵ); corrected = false)
    end
    loglikelihood = sum(logpdf.(Normal(μ, √σ²), y))
    loglikelihood + logprior
end

simulate!(p::ToyExponentialModel) = rand!(p.ϵ)

"Analytical distribution for the problem."
function analytical_distribution(p::ToyExponentialModel)
    @unpack y, A, B = p
    Truncated(InverseGamma(length(y), sum(y)), A, B)
end

function limit_posterior(p::ToyExponentialModel)
    @unpack y, A, B = p
    N = length(y)
    ℓ(λ) = -N*log(λ) - sum((y./λ-1).^2)/2 # log posterior
    normalized_density(ℓ, mean(y), A, B)
end
                   
@testset "toy exponential limit" begin
    λ = 2.0
    p = ToyExponentialModel(λ, 100, 0)
    posterior = limit_posterior(p)
    
    @test hquadrature(posterior, p.A, p.B)[1] ≈ 1
    
    chain, a = mcmc(RWMH(diagm([0.02])), p, [log(λ)], 10000)
    λs = exp.(first.(chain))

    @test mean(λs) ≈ mean(p.y) rtol = 0.15
    
    test_cdf(limit_posterior(p), λs, p.A)
end

@testset "toy exponential 10x replication" begin
    λ = 3.0
    p = ToyExponentialModel(λ, 100, 1000)
    
    chain, a = mcmc(RWMH(diagm([0.02])), p, [log(λ)], 10000)
    λs = exp.(first.(chain))

    @test mean(λs) ≈ mean(p.y) rtol = 0.15
    
    test_cdf(limit_posterior(p), λs, p.A; atol = 0.1)
end
