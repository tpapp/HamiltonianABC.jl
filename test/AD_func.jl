using Distributions
using Parameters
#using HamiltonianABC
using Plots
using StatsBase
using StatPlots                 # for kernel density
using Base.Test
using ForwardDiff
using ReverseDiff
using BenchmarkTools

## simple function and its gradient
func(a) = 2*a[1] + a[2]^2
a_ = [2.0, 4.0]

## implementing the two methods and benchmarking them
## ForwardDiff works better
@benchmark ForwardDiff.gradient(ARG -> func(ARG), a_)
@benchmark ReverseDiff.gradient(ARG -> func(ARG), a_)
ForwardDiff.gradient(ARG -> func(ARG), a_)
ReverseDiff.gradient(ARG -> func(ARG), a_)
## reversediff works here, could be a problem of types in the functions
## should look into it

f_(x) = sum(sqrt, x)
ff_(x) = 0.5 * x^(-0.5)
f_grid = linspace(1,2,51)
f_res = ForwardDiff.gradient(x-> f_(x),  f_grid)
@benchmark ForwardDiff.gradient(f_,  f_grid)
ff_res = map(ff_, f_grid)
@benchmark map(ff_, f_grid)

plot(f_grid, f_res)
plot!(f_grid, ff_res)
ress = f_res - ff_res
plot(f_grid, ress)

## does not work, problem with array
ReverseDiff.gradient(x-> f_(x), linspace(1,2,30))



function likelihood_calc(X, params)
    # N = how many columns X has
    N = size(X, 2)
    μ, σ_2 = params
    # Calculate likelihood (normal)
    _like = zeros(eltype(μ), N)
    for i ∈ 1:N
        _like[i] = exp(-1/(2 * σ_2) * (X[:, i] .- μ)' *  (X[:, i] .- μ))
    end

    # return log-likelihood
    return sum(log(_like))

end


X = randn(310, 1) # some random data
likelihood_calc(X, μ)

μ = [0.0, 1.0]

ForwardDiff.gradient(ARG -> likelihood_calc(X, ARG), μ)
## reversediff.gradient breaks, need to look at it more carefully
ReverseDiff.gradient(ARG -> likelihood_calc(X, ARG), μ)

function OLS(y, x)
    β = x \ y
    err = (y - x .* β)
    v = mean(abs2, err)
    return β, v
end


Xs = rand(Normal(5,1), 1000)
Ys = 2.1 * Xs + randn(1000)
OLS(Ys, Xs)
d_ = [2.1, 1.0]
likelihood_calc(Ys, d_)
ForwardDiff.gradient(dd -> likelihood_calc(Ys - Xs, dd), d_)




function likelihood_pdf(Y, X, params)
    # N = how many columns X has
    N = size(X, 2)
    μ, σ_2 = params
    # Calculate likelihood
    _like = zeros(eltype(μ), N)
    for i ∈ 1:N
        _like[i] = exp(-1/(2 * σ_2) * (Y[:, i] - X[:, i] * μ)' *  (Y[:, i] - X[:, i] * μ))
    end

    # return log-likelihood
    return sum(log(_like))

end

ForwardDiff.gradient(dd -> likelihood_pdf(Ys, Xs, dd), d_)
# reversediff breaks
ReverseDiff.gradient(dd -> likelihood_pdf(Ys, Xs, dd), d_)

## could be faster
@benchmark ForwardDiff.gradient(dd -> likelihood_pdf(Ys, Xs, dd), d_)
