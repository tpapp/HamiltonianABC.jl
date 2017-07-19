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

function simulate!(pp::Toy_Vol_Problem)
    N = length(pp.ϵ)
    pp = Toy_Vol_Problem(pp.ys, pp.prior_ρ, pp.prior_σ_v, rand(Chisq(1), N), randn(N))
end

function OLS(y,x)
    N, K = size(x)
    β = inv(x' * x) * (x' * y)
    err = (y - x * β)
    σ_2 = (err' * err) / (N) + eps()
    return β, σ_2
end

function logdensity(pp::Toy_Vol_Problem, θ)
    @unpack ys, ϵ, prior_ρ, prior_σ_v, ν = pp
    ρ, σ_v = θ
    N = length(ϵ)
    #println(ϵ[1])

    if (ρ > 1 - eps() || σ_v ≤ 0 )
        return - Inf
    else

    logprior = logpdf(prior_ρ, ρ) + logpdf(prior_σ_v, σ_v)

    # Generating xs, which is the latent volatility process

        xs = simulate_stochastic(ρ, σ_v, [ϵ ν])[2]
        X = [ones(N-2) xs[2:(end-1)] xs[3:end]]
        β, σ_2 = OLS(xs[3:end], X)
        β_1, β_2, β_3 = β
        log_likelihood = 0
        for i in 3:length(ys)
            log_likelihood += logpdf(Normal(β_1 + β_2 * ys[i-1] + β_3 * ys[i-2], √(σ_2)), ys[i])
        end
        logprior + log_likelihood
    end

end

y = simulate_stochastic(0.8, 1, [rand(Chisq(1), 1000) randn(1000)])[1]
pp = Toy_Vol_Problem(y, Uniform(-1, 1), InverseGamma(1, 1), rand(Chisq(1), 10000), randn(10000))
θ = [0.8, 1]
logdensity(pp, θ)
chain, a = mcmc(RWMH(diagm([0.02, 0.02])), pp, θ, 5000)
mean(chain)
