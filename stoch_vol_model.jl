using Distributions
using Parameters
using HamiltonianABC
using ContinuousTransformations
using Plots
using StatsBase
import HamiltonianABC: logdensity, simulate!


function simulate_stochastic(ρ, σ_v, ϵs)
    N, K = size(ϵs)
    x_0 = rand(Normal(0, 1 / (1 - ρ^2)))
    xs = Vector{Float64}(N)
    ys = Vector{Float64}(N)
    xs[1] = ρ * x_0 + σ_v * ϵs[1, 2]
    ys[1] = xs[1] + log(ϵs[1, 1]) + 1.27
    for i in 2 : N
        xs[i] = ρ * xs[i-1] + σ_v * ϵs[i, 2]
        ys[i] = xs[i] + log(ϵs[i, 1]) + 1.27
    end
    ys, xs
end

struct Toy_Vol_Problem
    "observed data"
    ys
    "prior for ρ (persistence)"
    prior_ρ
    "prior for σ_v (volatility of volatility)"
    prior_σ_v
    "χ^2 draws for simulation"
    ϵ::Vector{Float64}
    "Normal(0,1) draws for simulation"
    ν::Vector{Float64}
end

simulate!(pp::Toy_Vol_Problem) = rand!(pp.ϵ, pp.ν)

function logdensity(pp::Toy_Vol_Problem, θ)
    @unpack ys, ϵ, prior_ρ, prior_σ_v, ν = pp
    ρ, σ_v = θ

    logprior = logpdf(prior_ρ, ρ) + logpdf(prior_σ_v, σ_v)

    # Generating xs, which is the latent volatility process
    if ρ > 1 - eps()
        return - Inf
    else
        xs = simulate_stochastic(ρ, σ_v, [ϵ ν])[2]
        β_1, β_2 = mean_and_std(xs, corrected = false)
        println(xs)
        log_likelihood = 0
        for i in 2:length(xs)
            log_likelihood += logpdf(Normal(β_1 * xs[i-1] + β_2 * ν[i-1], √(π^2 / 2)), ys[i])
        end
        #log_likelihood = sum(logpdf(Normal(β_1, β_2 + π^2 /2), ys))
        logprior + log_likelihood
    end

end

y = simulate_stochastic(0.8, 1, [rand(Chisq(1), 100) randn(100)])[1]
pp = Toy_Vol_Problem(y, Uniform(-1, 1), InverseGamma(1, 1), rand(Chisq(1), 100), randn(100))
θ = [0.8, 1]
chain, a = mcmc(RWMH(diagm([0.02, 0.02])), pp, θ, 100)
