using ArgCheck
using Distributions
using Parameters
using DynamicHMC
using StatsBase
using StatPlots
import DynamicHMC: logdensity, loggradient, length
using Base.Test
using ForwardDiff
using ReverseDiff
using Plots
using ContinuousTransformations
using ProfileView
import ForwardDiff: Dual
using BenchmarkTools
plotlyjs()

###############################################################################################################
## Functions needed for the model
###################################################################################

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
    x₀ = νs[1]*σ*(1 - ρ^2)^(-0.5)
    xs = Vector{typeof(x₀)}(N)
    for i in 1:N
        xs[i] = (i == 1) ? x₀ : (ρ*xs[i-1] + σ*νs[i])
    end
    xs + log.(ϵs) + 1.27
end

simulate_stochastic(ρ, σ, N) = simulate_stochastic(ρ, σ, rand(Chisq(1), N), randn(N))

struct Toy_Vol_Problem{T, Prior_ρ, Prior_σ}
    "observed data"
    ys::Vector{T}
    "prior for ρ (persistence)"
    prior_ρ::Prior_ρ
    "prior for σ_v (volatility of volatility)"
    prior_σ::Prior_σ
    "χ^2 draws for simulation"
    ϵ::Vector{T}
    "Normal(0,1) draws for simulation"
    ν::Vector{T}
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
    err = y - x * β
    v = mean(abs2, err)
    β, v
end

"an array of ForwardDiff.Dual, for testing type stability"
duals(dims...) = fill(Dual(1.0, 2.0), dims...)


## In this form, we use an AR(2) process of the first differences with an intercept as the auxiliary model.

"""
    lag(xs, n, K)

Lag-`n` operator on vector `xs` (maximum `K` lags).
"""
lag(xs, n, K) = xs[((K+1):end)-n]

"""
    lag_matrix(xs, ns, K = maximum(ns))

Matrix with differently lagged xs.
"""
function lag_matrix(xs, ns, K = maximum(ns))
    M = Matrix{eltype(xs)}(length(xs)-K, maximum(ns))
    for i ∈ ns
        M[:, i] = lag(xs, i, K)
    end
    M
end


"first auxiliary regression y, X, meant to capture first differences"
function yX1(zs, K)
    Δs = diff(zs)
    lag(Δs, 0, K), hcat(lag_matrix(Δs, 1:K, K), ones(eltype(zs), length(Δs)-K), lag(zs, 1, K+1))
end

"second auxiliary regression y, X, meant to capture levels"
function yX2(zs, K)
    lag(zs, 0, K), hcat(ones(eltype(zs), length(zs)-K), lag_matrix(zs, 1:K, K))
end

bridge_trans(dist::Uniform) = bridge(ℝ, Segment(dist.a, dist.b))
bridge_trans(dist::InverseGamma) = bridge(ℝ, ℝ⁺)

parameter_transformations(pp::Toy_Vol_Problem) = bridge_trans.((pp.prior_ρ, pp.prior_σ))

function logdensity(pp::Toy_Vol_Problem, θ)
    @unpack ys, prior_ρ, prior_σ, ν, ϵ = pp
    trans = parameter_transformations(pp)

    value_and_logjac = map((t, raw_θ) -> t(raw_θ, LOGJAC), trans, θ)
    par = first.(value_and_logjac)
    ρ, σ = par

    logprior = logpdf(prior_ρ, ρ) + logpdf(prior_σ, σ) + sum(last, value_and_logjac)

    N = length(ϵ)

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

"""
    ρ̂ ₜ = Variogram(xs, j, var)

Give back the estimated autocorrelation given by equation (11.7) in Gelman et al. 2014, Bayesian Data Analysis, Third Edition.
"""
function Variogram(xs, j, var)
    xs_ = lag(xs, j, j)
    xs_t = lag(xs, 0, j)
    1 - mean(abs2,(xs_ - xs_t))/ (2 * var)
end


"""
    ESS(xs, var = var(xs))

Give back the estimated effective sample size given by equation (11.8) in Gelman et al.
If not specifically given, variance is calculated from xs.
"""
function  ESS(xs, var = var(xs))
    N = length(xs)
    # starting from lag 0
    ϕ_t = 1 + 2 * Variogram(xs, 1, var)
    J = 1
    ## continuing until the sum of autocorrelation estimates for two successive lags is negative
    ## following Gelman et al. at page 287.
    while J < (N -2)
        dif = Variogram(xs, (2*J), var) + Variogram(xs, (2*J+1), var)
        if dif < 0
            break
        else
            ϕ_t += 2 * dif
            J += 1
        end
    end
    # mn / (1 + 2*Σᵀ ρₜ) is the result
    N / ϕ_t
end

## with Forward mode AD
loggradient(pp::Toy_Vol_Problem, x) = ForwardDiff.gradient(y->logdensity(pp, y), x)
## with reverse mode AD, NOT WORKING RIGHT NOW
## ISSUE : there is a problem with line 139, i will look into it
loggradient_rev(pp::Toy_Vol_Problem, x) = ReverseDiff.gradient(y->logdensity(pp, y), x)

###############################################################################
## Toy model
##############################################################################

ρ = 0.8
σ = 0.6
y = simulate_stochastic(ρ, σ, 10000)
pp = Toy_Vol_Problem(y, Uniform(-1, 1), InverseGamma(1, 1), 10000)

θ₀ = [inv(t)(param) for (t,param) in zip(parameter_transformations(pp), [ρ, σ])]

## checking whether the two types of loggradient functions work,
## right now, reversediff.gradient does not work
loggradient(pp, θ₀)
#loggradient_rev(pp, θ₀)

## length, pp does not have an element such that i could get hold of the length of the parameter vector
Base.length(::Toy_Vol_Problem) = 2
## defining RNG
const RNG = srand(UInt32[0x23ef614d, 0x8332e05c, 0x3c574111, 0x121aa2f4])

sample, tuned_sample = NUTS_tune_and_mcmc(RNG, pp, 3000; q = θ₀)
## transfomring back the sample
NN = ceil(Int, length(sample) - (size(sample,1)) * 0.5 )
sample_ρ = Vector{Float64}(NN)
sample_σ = Vector{Float64}(NN)
for i in 1:NN
sample_ρ[i], sample_σ[i] = [t(param) for (t,param) in zip(parameter_transformations(pp), sample[i+NN].q)]
end

plt = plot(density(sample_ρ), label = "posterior", title = "ρ")
plot!(plt, a -> pdf(pp.prior_ρ, a), linspace(-1, 1, 100), label = "prior")
vline!(plt, [ρ], label = "true value")

lt = plot(density(sample_σ), label = "posterior", title = "σ")
plot!(lt, a -> pdf(pp.prior_σ, a), linspace(0, 2, 100), label = "prior")
vline!(lt, [σ], label = "true value")

## priors do not really have an impact on posterior.

## Effective sample size
ESS(sample_ρ)
ESS(sample_σ)


###############################################################################
##  Testing type-stability
###############################################################################

@code_warntype simulate_stochastic(0.8, 0.6, 10000)
@code_warntype simulate_stochastic(Dual(1.0,2.0), Dual(2.0,1.0), zeros(10), zeros(10))

@code_warntype OLS(ones(3), ones(3,3))

@code_warntype OLS(duals(3), duals(3,3))
## seems type-stable
@code_warntype lag(ones(5), 1,1)
## seems like that this lag_matrix function is type-stable now
@code_warntype lag_matrix(ones(5), 1:3)
@code_warntype lag_matrix(duals(3), 1:3)
lag_matrix(1:5, 1:3) == [3 2 1; 4 3 2]

## now the yX1 function is type-stable
@code_warntype yX1(simulate_stochastic(1.0, 2.0, zeros(10), zeros(10)), 2)
@code_warntype yX1(simulate_stochastic(Dual(1.0,2.0),Dual(2.0,1.0), zeros(10), zeros(10)), 2)


## yX2 is type-stable as well now
@code_warntype yX1(simulate_stochastic(1.0, 2.0, zeros(10), zeros(10)), 2)
@code_warntype yX1(simulate_stochastic(Dual(1.0,2.0),Dual(2.0,1.0), zeros(10), zeros(10)), 2)

## OLS(yX2()...) gives a type stable output
@code_warntype OLS(yX2(simulate_stochastic(1.0, 2.0, zeros(10), zeros(10)), 2)...)
@code_warntype OLS(yX2(simulate_stochastic(Dual(1.0,2.0),Dual(2.0,1.0), zeros(10), zeros(10)), 2)...)


@code_warntype bridge_trans(pp.prior_ρ)
@code_warntype bridge_trans(pp.prior_σ)

@code_warntype parameter_transformations(pp)


@code_native logdensity(pp, θ₀)
@code_warntype logdensity(pp, θ₀)


### testing the results

@testset "NUTS sample" begin
    @test mean(sample_ρ) ≈ ρ atol= 0.1
    @test mean(sample_σ) ≈ σ atol= 0.1
end

###############################################################################
## Profiling and Benchmarking
###############################################################################


## profiling
Profile.clear()
f(N = 10000) = for _ in 1:N logdensity(pp, θ₀) end
f(1)                            # don't capture compilation
@profile f(10000)
ProfileView.view()
Profile.clear()


### ISSUE
## We see a lot of red, but they point at different julia files, i.e. inference.jl, array.jl, etc...
