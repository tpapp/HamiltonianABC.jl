using Distributions
using Parameters
using HamiltonianABC
using ContinuousTransformations
using Plots
using Gadfly    # fail to precompile Gadfly for me
using StatsBase
import HamiltonianABC: logdensity, simulate!
using Base.Test

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
    N, K = size(x)
    β = x \ y
    err = (y - x * β)
    σ_2 = mean(abs2, err) + eps()
    return β, σ_2
end

## In this form, we use an AR(2) regression as the auxiliary model.

function logdensity(pp::Toy_Vol_Problem, θ)
    @unpack ys, ϵ, prior_ρ, prior_σ_v, ν = pp

    ρ, σ_v = θ
    N = length(ϵ)

    if (abs(ρ) > 1 - eps() || σ_v ≤ 0 )
        return -Inf
    else

        logprior = logpdf(prior_ρ, ρ) + logpdf(prior_σ_v, σ_v)

    # Generating xs 
        xs = simulate_stochastic(ρ, σ_v, ϵ, ν)
        X = [ones(N-2) xs[2:(end-1)] xs[3:end]]
        β, σ_2 = OLS(xs[3:end], X)
        # println(σ_2)
        β_1, β_2, β_3 = β
        log_likelihood = 0
        for i in 3:length(ys)
            log_likelihood += logpdf(Normal(β_1 + β_2 * ys[i-1] + β_3 * ys[i-2], √(σ_2)), ys[i])
        end
        return(logprior + log_likelihood)
    end

end

## Trial
y = simulate_stochastic(0.8, 1, rand(Chisq(1), 100), randn(100))
pp = Toy_Vol_Problem(y, Uniform(-1, 1), InverseGamma(1, 1), 1000)
θ = [0.8, 1]
logdensity(pp, θ)
chain, a = mcmc(RWMH(diagm([0.01, 0.01])), pp, θ, 5000)

## getting rid of the eps() in the OLS function breaks the code at line 77
## σ_2 is extremely small from the beginning, i.e 1.9830235061855857e-31
## this is an issue to be solved


# Analysis with some plotting

result_1 = vcat(chain[2500:end]'...)
plot(result_1)
hline!([θ[1]], line = (4, :red))
hline!([θ[2]], line = (4, :dash, :black))
ρ_sample_1 = result_1[:, 1]
σ_sample_1 = result_1[:, 2]

# histogram with the prior
histogram(σ_sample_1, color = (:lightgreen))
plot!(a -> pdf(pp.prior_σ_v, a), linspace(0.0, 20.0, 1000), line = (4, :dash, :black))
# histogram with the prior
histogram(ρ_sample_1, color = (:lightgreen))
plot!(a -> pdf(pp.prior_ρ, a), linspace(-1.0, 1.0, 1000), line = (4, :dash, :black))

mean(ρ_sample_1)
mean(σ_sample_1)

## Analysis
## the estimation of ρ is not that bad
## the estimation of σ is terrible
## the prior does not have a big influence

## this should work under Gadfly:
# plot(σ_sample_1, Geom.density, Geom.histogram(density=true))
# plot!(a -> pdf(pp.prior_σ_v, a), linspace(0.0, 20.0, 1000), line = (4, :dash, :black))
## the histogram should display the density

@testset "AR(2)" begin
    @test mean(ρ_sample_1) ≈ θ[1] rtol = 0.2
    @test mean(σ_sample_1) ≈ θ[2] rtol = 0.2
end

## In this form, we use an AR(2) process of the first differences as the auxiliary model.

function logdensity(pp::Toy_Vol_Problem, θ)
    @unpack ys, ϵ, prior_ρ, prior_σ_v, ν = pp
    ρ, σ_v = θ
    N = length(ϵ)

    if (abs(ρ) > 1 - eps() || σ_v ≤ 0 )
        return -Inf
    else

        logprior = logpdf(prior_ρ, ρ) + logpdf(prior_σ_v, σ_v)

    # Generating xs, which is the latent volatility process

        xs = simulate_stochastic(ρ, σ_v, ϵ, ν)

    # We work with first differences
        Δ_2 = diff(xs[1:end-2])
        Δ_1 = diff(xs[2:end-1])
        Δ = diff(xs[3:end])
        X = [Δ_1 Δ_2]
        β, σ_2 = OLS(Δ, X)
        β_2, β_3 = β
        log_likelihood = 0
        for i in 4:length(ys)
            log_likelihood += logpdf(Normal(β_2 * (ys[i-1] - ys[i-2]) + β_3 *(ys[i-2] - ys[i-3]), √(σ_2)), ys[i] - ys[i-1])
        end
        return(logprior + log_likelihood)
    end
end

# Trial
y = simulate_stochastic(0.8, 1, rand(Chisq(1), 100), randn(100))
pp = Toy_Vol_Problem(y, Uniform(-1, 1), InverseGamma(1, 1), 10000)
θ = [0.8, 1]
logdensity(pp, θ)
chain, a = mcmc(RWMH(diagm([0.01, 0.01])), pp, θ, 5000)

# Analysis with some plotting

result_2 = vcat(chain[2500:end]'...)
plot(result_2)
hline!([θ[1]], line = (4, :red))
hline!([θ[2]], line = (4, :dash, :black))
ρ_sample_2 = result_2[:, 1]
σ_sample_2 = result_2[:, 2]
# histogram with the prior
histogram(σ_sample_2, color = (:lightgreen))
plot!(a -> pdf(pp.prior_σ_v, a), linspace(0.0, 10.0, 1000), line = (4, :dash, :black))

# histogram with the prior
histogram(ρ_sample_2, color = (:lightgreen))
plot!(a -> pdf(pp.prior_ρ, a), linspace(-1.0, 1.0, 1000), line = (4, :dash, :black))
mean(ρ_sample_2)
mean(σ_sample_2)


@testset "first difference" begin
    @test mean(ρ_sample_2) ≈ θ[1] rtol = 0.25
    @test mean(σ_sample_2) ≈ θ[2] rtol = 0.25
end
