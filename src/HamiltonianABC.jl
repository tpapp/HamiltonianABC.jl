module HamiltonianABC

using Distributions

export
    logdensity,
    simulate!,
    propose,
    mcmc,
    RWMH

"""
    logdensity(model, θ)

Return the log density for `model` at parameter `θ`.

A *model* is an object that can behave as a probability distribution over some
space, possibly by simulation (see [`simulate`](@ref)).
"""
function logdensity end

"""
    simulate!(model)

Re-generate the simulated shocks for `model`. For Monte Carlo calculation of
integrals, `model` should save appropriate shocks, independent of
parameters. Evaluations of [`logdensity`](@ref) then transform these using the
parameters.
"""
function simulate! end

"""
Random walk Metropolis-Hastings.

In this library, it is provided purely for testing and pedagogical purposes. See
[`NUTS`](@ref) for a more practical alternative.
"""
struct RWMH{S <: AbstractMatrix}
    "Covariance matrix for proposals."
    Σ::S
end

"""
    propose(transition, model, θ)

Generate a Metropolis-Hastings proposal using the given `transition` algorithm
and `model`, starting from parameters `θ`.

Return `θ′, a` where `θ′` is the proposed new parameter and `a` is the
acceptance probability.

The method `logdensity(model, θ)` needs to be supported by the arguments.
"""
function propose(transition::RWMH, model, θ)
    θ′ = rand(MvNormal(θ, transition.Σ))
    a = min(1, exp(logdensity(model, θ′) - logdensity(model, θ)))
    θ′, a
end

"""
    mcmc(transition, model, θ, N)

Runs a Markov Chain Monte Carlo algorithm by calling [`propose`](@ref)
repeatedly `N` times. After each iteration, re-simulate the shocks for model
using `simulate!`.

Return the vector of parameters, and the average acceptance rate.

The following interface needs to be supported by the arguments: -
`logposterior(model, θ)`, `simulate!(model)`.
"""
function mcmc{T}(transition, model, θ::T, N)
    chain = Vector{T}(N)
    as = Vector{Float64}(N)
    for i in 1:N
        θ′, a = propose(transition, model, θ)
        if rand() ≤ a
            θ = θ′
        end
        as[i] = a
        chain[i] = θ
        simulate!(model)
    end
    chain, mean(as)
end

end # module
