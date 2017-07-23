using Distributions
using Parameters
using HamiltonianABC
using ContinuousTransformations
using Plots
using StatsBase
using StatPlots                 # for kernel density
import HamiltonianABC: logdensity, simulate!
using Base.Test
plotlyjs()                      # Tamas likes this, optional

"""
    simulate_stochastic(ρ, σ_v, ϵs, ηs)

Take in the parameter values (ρ, σ_v) for the latent volatility process, the errors ϵs used for the measurement equation and the errors ηs used for the transition equation.

The discrete-time version of the Ornstein-Ulenbeck Stochastic - volatility model:

    y_t = x_t + ϵs_t where ϵs_t ∼ χ^2(1)
    x_t = ρ * x_(t-1) + σ_v * ηs_t  where ηs_t ∼ N(0, 1)

"""
function simulate_stochastic(ρ, σ_v, ϵs, ηs)
    N = length(ϵs)
    x_0 = rand(Normal(0, 1 / (1 - ρ^2)))
    xs = Vector{Float64}(N)
    ys = Vector{Float64}(N)
    xs[1] = ρ * x_0 + σ_v * ηs[1]
    ys[1] = xs[1] + log(ϵs[1]) + 1.27
    for i in 2 : N
        xs[i] = ρ * xs[i-1] + σ_v * ηs[i]
        ys[i] = xs[i] + log(ϵs[i]) + 1.27
    end
    ys
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


"""
    Toy_Vol_Problem(ys, prior_ρ, prior_σ_v, M)

Convenience constructor for Toy_Vol_Problem.
Take in the observed data, the priors, and number of simulations (M).
"""
function Toy_Vol_Problem(ys, prior_ρ, prior_σ_v, M)
    Toy_Vol_Problem(ys, prior_ρ, prior_σ_v, rand(Chisq(1), M), randn(M))
end

"""
    simulate!(pp::Toy_Vol_Problem)

Updates the shocks of the model.
"""
function simulate!(pp::Toy_Vol_Problem)
    @unpack ϵ, ν = pp
    rand!(Chisq(1), ϵ)
    randn!(ν)
end

"""
    OLS(y, x)

Take in the dependant variable (y) and the regressor (x), give back the estimated coefficients (β) and the variance (σ_2).
"""
function OLS(y, x)
    β = x \ y
    err = (y - x * β)
    σ_2 = mean(abs2, err) + eps()
    return β, σ_2
end


## In this form, we use an AR(2) process of the first differences with an intercept as the auxiliary model.

"""
    lag(xs, n, K)

Lag-`n` operator on vector `xs` (maximum `K` lags).
"""
lag(xs, n, K) = xs[((K+1):end)-n]

lag_matrix(xs, ns, K = maximum(ns)) = hcat([lag(xs, n, K) for n in ns]...)

@test lag_matrix(1:5, 1:3) == [3 2 1; 4 3 2]

"""
    y, X = diff_yX(zs, K)

Take the first differences of `zs`, then return conformable `y` and `X` matrices for OLS, where `X` is the first `K -1` lags of the first differences  plus a vector of ones and `y` starts at time `K+1`.
"""
function diff_yX(zs, K)
    Δs = diff(zs)
    lag(Δs, 0, K), hcat(lag_matrix(Δs, (K:-1:1), K), ones(length(Δs)-2))
end

function logdensity(pp::Toy_Vol_Problem, θ)
    @unpack ys, ϵ, prior_ρ, prior_σ_v, ν = pp
    ρ, σ_v = θ
    N = length(ϵ)

    if (abs(ρ) > 1 - eps() || σ_v ≤ 0 )
        return -Inf
    else

        logprior = logpdf(prior_ρ, ρ) + logpdf(prior_σ_v, σ_v)

        # Generating xs, which is the latent volatility process

        zs = simulate_stochastic(ρ, σ_v, ϵ, ν)
        y, X = diff_yX(zs, 2)

        # We work with first differences
        β, σ_2 = OLS(y, X)
        yy, yX = diff_yX(ys, 2)
        log_likelihood = sum(logpdf.(Normal(0, √(σ_2)), yy - yX * β))
        logprior + log_likelihood
    end
end

# Trial
θ = [0.8, 1.2]
y = simulate_stochastic(θ[1], θ[2], rand(Chisq(1), 1000), randn(1000))
pp = Toy_Vol_Problem(y, Uniform(-1, 1), InverseGamma(1, 1), 1000)

# visualize posterior
ρ_grid = linspace(-0.9, 0.9, 60)
σ²_grid = linspace(0.1, 2, 50)
logdensity_on_grid = ((ρ, σ²)->logdensity(pp, [ρ, σ²])).(ρ_grid, σ²_grid')
logdensity_on_grid .-= maximum(logdensity_on_grid)
heatmap(ρ_grid, σ²_grid, logdensity_on_grid', xlab = "ρ", ylab = "σ²",
        title = "log posterior")

heatmap(ρ_grid, σ²_grid, exp.(logdensity_on_grid'), xlab = "ρ", ylab = "σ²",
        title = "posterior")


logdensity(pp, θ)
chain, a = mcmc(RWMH(diagm([0.02, 0.1])), pp, θ, 10000) # Σ was hand-tuned

# Analysis with some plotting

posterior = vcat(chain[5000:end]'...)
plot(posterior)

cov(posterior, 1)    * 2/3           # used about 2/3 * this for Σ above

plt = plot(density(posterior[:, 1]), label = "posterior", title = "rho")
plot!(plt, a -> pdf(pp.prior_ρ, a), linspace(-1, 1, 100), label = "prior")
vline!(plt, [θ[1]], label = "true value")

plt = plot(density(posterior[:, 2]), label = "posterior", title = "σ")
plot!(plt, a -> pdf(pp.prior_σ_v, a), linspace(0, 5, 100), label = "prior")
vline!(plt, [θ[2]], label = "true value")

mean(posterior, 1)     # not bad, could be better

@testset "first difference" begin
    @test mean(posterior, 1)[1] ≈ θ[1] rtol = 0.30
    @test mean(posterior, 1)[2] ≈ θ[2] rtol = 0.30
end
