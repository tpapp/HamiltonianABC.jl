using Distributions
using StatsBase

using HamiltonianABC
import HamiltonianABC: logprior, simulate_ϵ, generate_x, estimate_ϕ, data_loglikelihood

"""
The true model is ``y ∼ Exponential(λ), IID``.

The auxiliary model is ``y ∼ N(μ, σ²)``, with statistics ϕ = (μ, σ).

We use a ``λ ∼ Γ(α,β)`` prior.

The `ϵ` are Uniform(0,1) draws, mapped to exponential draws using the CDF.
"""
struct ToyExponentialProblem
    "observed data"
    y::Vector{Float64}
    "lower boundary of the uniform prior distribution"
    A::Float64
    "upper boundary of the uniform prior distribution"
    B::Float64
    "number of draws for simulated data"
    M::Int
end

"""
Helper function, verifies that `θ` has a single element, this is is
taken as `log(λ)`, transformed to `λ` and returned.
"""
get_λ(θ) = ((logλ,) = θ; exp(logλ))

function logprior(pp::ToyExponentialProblem, θ)
    # log Jacobian is ``log(|exp(logλ)|) = logλ``, hence the ``+ θ[1]``.
    logpdf(Uniform(pp.A,pp.B), get_λ(θ)) + θ[1]
end

simulate_ϵ(pp::ToyExponentialProblem) = rand(pp.M)

generate_x(pp::ToyExponentialProblem, θ, ϵ) = quantile.(Exponential(get_λ(θ)), ϵ)

estimate_ϕ(pp::ToyExponentialProblem, x) = mean_and_std(x; corrected = false)

data_loglikelihood(pp::ToyExponentialProblem, ϕ) = sum(logpdf.(Normal(ϕ...), pp.y))

@testset "toy exponential" begin

    λ = 5.0
    pp = ToyExponentialProblem(rand(Exponential(λ), 100), 0.0, 10.0, 1000)
    state = simulate_state(pp, [log(λ)])
    chain, a = mcmc(RWMH(fill(0.05,1,1)), pp, state, 10000)
    
    λs = get_λ.(get_θ.(chain))

    @test mean(λs) ≈ mean(pp.y) atol = 0.5
end
