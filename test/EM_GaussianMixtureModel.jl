using Distributions
using Plots
Plots.gr()
using Parameters
using Roots
using HamiltonianABC

using ArgCheck

"""
I am following the notations of Wikipedia while implementing the Expectation-Maximization algortihm of Mixture of Normals
link: https://en.wikipedia.org/wiki/Mixture_model#Expectation_maximization_.28EM.29
"""


"""
    unit_row_matrix(n, m)

Facilitator function, generates an nxm matrix, whose rows sum up to 1.
"""

function unit_row_matrix(n, m)
    M = rand(n, m)
    M .= M ./ sum(M, 2)
    M
end

@testset "unit row matrix" begin
    for _ in 1:10
        n, m = rand(3:10, 2)
        M = unit_row_matrix(n, m)
        @test size(M) == (n, m)
        @test sum(M, 2) ≈ ones(n)
        @test all(M .≥ 0)
    end
end

"""
    normal_mixture(μs, σs, weights)

Generate a mixture of normals with means 'μs', st. deviations `σs`
with probabilities 'weights'
"""
function normal_mixture(μs, σs, weights)
    # mixture of normals model
    p = sortperm([μs...])
    μs, σs, weights = μs[p], σs[p], weights[p]
    MixtureModel(Normal.(μs, σs), weights)
end

@testset "normal_mixture" begin
    μs = 1:3
    σs = fill(1, 3)
    weights = [0.1, 0.2, 0.7]
    simulated_mean = mean(rand(normal_mixture(μs, σs, weights), 10000))
    @test simulated_mean ≈ dot(μs, weights) atol = 0.1
end


"""
    normal_mixture_EM_parameters!(μs, σs, weights, hs, x)

Maximization step. Update the parameters 'μs','σs' and 'weigths',
given the posterior probabilities `hs` and data `x`.

`hs[i,j]` is the posterior probability of component `i` for
observation `j`.
"""
function normal_mixture_EM_parameters!(μs, σs, ws, hs, xs)
    m, n = size(hs)
    # @argcheck n == length(μs) == length(σs) == length(ws)
    for i in 1:n
        h = @view hs[:, i]
        ∑h = sum(h)
        ws[i] = ∑h / m
        μs[i] = dot(xs, h) / ∑h
        σs[i] = √(sum(h .* (xs - μs[i]).^2) / ∑h)
    end
    μs, σs, ws
end

@testset "not mixture" begin
    ## testing the normal_mixture_EM_parameters. If the input hs matrix
    ## contains ones in one column and zeros in the others, then it should
    ## give back the actual distribution'

    # building hs with 1s in the first column and zeros in the others
    hs = [ones(1000) zeros(1000, 2)]
    # drawing normally distributed numbers
    xs = rand(Normal(1.2, 0.4), 1000)
    μs = zeros(3)
    σs = zeros(3)
    ws = zeros(3)
    # estimating the parameters
    normal_mixture_EM_parameters!(μs, σs, ws, hs, xs)
    # density with estimated parameters
    @test ws[1] == 1
    @test mean(xs) ≈ μs[1]
    @test std(xs, corrected = false) ≈ σs[1]
end


"""
    normal_mixture_EM_posterior!(μs,σs,weights,hs,x)

Update the 'hs' matrix given the parameters 'μs','σs' and 'weigths' of the Gaussian Mixture Model.

Using the Expectation Maximization algorithm the hs matrix contains the posterior probabilities.
Return the logpdf of the Mixture Model.
"""

function normal_mixture_EM_posterior!(μs, σs, weights, hs, x)
    for k in 1:length(μs)
        dist = Normal(μs[k],σs[k])
        for t in 1:length(x)
            hs[t, k] = weights[k] * pdf(dist, x[t])
        end
    end
    mix_likelihood = sum(log.(sum(hs, 2)))
    broadcast!(/, hs, hs, sum(hs, 2))
    return(mix_likelihood)
end


"""
    normal_mixture_EMM(x, m, max_step=1000, tol=eps())

Take in the observed data 'x' and , return the approximated parameters.
Take in the observed data points 'x', the number of the normal mixture 'm', the maximum number of iteration steps, return the approximated parameters.

The function uses the Expectation Maximization algorithm  to update the parameters of
the Gaussian Mixture Model, namely 'μs, σs and the weights'
the function also gives back the loglikelihood of the Mixture Model with
the updated parameters.
"""

function normal_mixture_EMM(x, m, max_step = 1000, tol = eps())
    ℓ = NaN
    step = 1
    n = length(x)
    # initialize the parameters
    hs = unit_row_matrix(n, m)
    μs = fill(mean(x), m)
    σs = fill(std(x, corrected = false), m)
    weights = fill(1 / m, m)
    while step ≤ max_step
        normal_mixture_EM_parameters!(μs, σs, weights, hs, x)
        ℓ′ = normal_mixture_EM_posterior!(μs, σs, weights, hs, x)
        Δ, ℓ = abs(ℓ′ - ℓ), ℓ′
        if Δ ≤ tol
            break
        end
        step += 1
    end
    p = sortperm([μs...])
    μs, σs, weights = μs[p], σs[p], weights[p]
    ℓ, μs, σs, weights, hs, step
end
