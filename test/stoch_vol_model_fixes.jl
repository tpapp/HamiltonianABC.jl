using ArgCheck
using Distributions
using Parameters
using HamiltonianABC
using StatsBase
using StatPlots                 # for kernel density
import HamiltonianABC: logdensity, simulate!
using Base.Test

using Plots
plotlyjs()                      # Tamas likes this, optional

"""
    simulate_stochastic(ρ, σ_v, ϵs, νs)

Take in the parameter values (ρ, σ) for the latent volatility process, the errors ϵs used for the measurement equation and the errors νs used for the transition equation.

The discrete-time version of the Ornstein-Ulenbeck Stochastic - volatility model:

    y_t = x_t + ϵ_t where ϵ_t ∼ χ^2(1)
    x_t = ρ * x_(t-1) + σ * ν_t  where ν_t ∼ N(0, 1)

"""
function simulate_stochastic(ρ, σ, ϵs, νs)
    N = length(ϵs)
    @argcheck N == length(νs)
    xs = Vector{Float64}(N)
    for i in 1:N
        xs[i] = (i == 1) ? (νs[1]*σ*(1 - ρ^2)^(-0.5)) : (ρ*xs[i-1] + σ*νs[i])
    end
    xs + log.(ϵs) + 1.27
end

simulate_stochastic(ρ, σ, N) = simulate_stochastic(ρ, σ, rand(Chisq(1), N), randn(N))

struct Toy_Vol_Problem
    "observed data"
    ys
    "prior for ρ (persistence)"
    prior_ρ
    "prior for σ_v (volatility of volatility)"
    prior_σ
    "χ^2 draws for simulation"
    ϵ::Vector{Float64}
    "Normal(0,1) draws for simulation"
    ν::Vector{Float64}
end

"""
    Toy_Vol_Problem(ys, prior_ρ, prior_σ, M)

Convenience constructor for Toy_Vol_Problem.
Take in the observed data, the priors, and number of simulations (M).
"""
function Toy_Vol_Problem(ys, prior_ρ, prior_σ, M)
    Toy_Vol_Problem(ys, prior_ρ, prior_σ, rand(Chisq(1), M), randn(M))
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
    β, v = OLS(y, x)

Take in the dependant variable (y) and the regressor (x), give back the estimated coefficients (β) and the variance (v).
"""
function OLS(y, x)
    β = x \ y
    err = (y - x * β)
    v = mean(abs2, err)
    return β, v
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
    y, X = yX1(zs, K)
"""
function yX1(zs, K)
    Δs = diff(zs)
    lag(Δs, 0, K), hcat(lag_matrix(Δs, 1:K, K), ones(length(Δs)-K), lag(zs, 1, K+1))
end

function yX2(zs, K)
    lag(zs, 0, K), hcat(ones(length(zs)-K), lag_matrix(zs, 1:K, K))
end

function logdensity(pp::Toy_Vol_Problem, θ)
    @unpack ys, prior_ρ, prior_σ, ν, ϵ = pp
    ρ, σ = θ
    N = length(ϵ)

    if (abs(ρ) > 1 - eps() || σ ≤ 0 )
        return -Inf
    else

        logprior = logpdf(prior_ρ, ρ) + logpdf(prior_σ, σ)

        # Generating xs, which is the latent volatility process

        zs = simulate_stochastic(ρ, σ, ϵ, ν)
        β₁, v₁ = OLS(yX1(zs, 2)...)
        β₂, v₂ = OLS(yX2(zs, 2)...)

        # We work with first differences
        y₁, X₁ = yX1(ys, 2)
        log_likelihood1 = sum(logpdf.(Normal(0, √v₁), y₁ - X₁ * β₁))
        y₂, X₂ = yX2(ys, 2)
        log_likelihood2 = sum(logpdf.(Normal(0, √v₂), y₂ - X₂ * β₂))
        logprior + log_likelihood1 + log_likelihood2
    end
end

# Trial
ρ = 0.8
σ = 1.2
y = simulate_stochastic(ρ, σ, 10000)
pp = Toy_Vol_Problem(y, Uniform(-1, 1), InverseGamma(1, 1), 10000)

# visualize posterior
ρ_grid = linspace(0.75, 0.9, 50)
σ_grid = linspace(0.9, 1.4, 30)
logdensity_on_grid = ((ρ, σ)->logdensity(pp, [ρ, σ])).(ρ_grid', σ_grid)
logdensity_on_grid .-= maximum(logdensity_on_grid)
heatmap(ρ_grid, σ_grid, exp.(logdensity_on_grid), xlab = "ρ", ylab = "σ",
        title = "posterior")

chain, a = mcmc(RWMH(diagm([0.02, 0.1])), pp, [ρ, σ], 10000) # Σ was hand-tuned
posterior = vcat(chain[5000:end]'...)

# convergence diagnostics (visual)
plot(posterior, label = ["ρ" "σ²"])

# marginal posterior densities vs prior
plt = plot(density(posterior[:, 1]), label = "posterior", title = "ρ")
plot!(plt, a -> pdf(pp.prior_ρ, a), linspace(0.6, 1, 100), label = "prior")
vline!(plt, [ρ], label = "true value")

plt = plot(density(posterior[:, 2]), label = "posterior", title = "σ²")
plot!(plt, a -> pdf(pp.prior_σ, a), linspace(0, 5, 100), label = "prior")
vline!(plt, [σ], label = "true value")

@testset "first difference" begin
    ρ̄, σ̄ = vec(mean(posterior, 1))
    @test ρ̄ ≈ ρ rtol = 0.02
    @test σ̄ ≈ σ rtol = 0.02
end
