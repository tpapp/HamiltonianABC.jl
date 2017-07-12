using Distributions
using Parameters
using Base.Test
using StatsBase
using ArgCheck

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

`hs[i, j]` is the posterior probability of component `j` for
observation `i`.
"""
function normal_mixture_EM_parameters!(μs, σs, ws, hs, xs)
    N, K = size(hs)
    @argcheck K == length(μs) && K == length(σs) && K == length(ws)
    @argcheck (N == length(xs))
    for j in 1:K
        h = @view hs[:, j]
        ∑h = sum(h)
        ws[j] = ∑h / N
        μs[j] = dot(xs, h) / ∑h
        σs[j] = √(sum(h .* (xs - μs[j]).^2) / ∑h)
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
    normal_mixture_EM_posterior!(μs, σs, ws, hs, xs)

Expectation step: update the `hs` matrix given the parameters `μs`,
`σs` and `ws` of the Gaussian Mixture Model. See
`normal_mixture_EM_parameters!` for variable name and index
conventions.

Return the (marginalized) log likelihood of the mixture model.
"""
function normal_mixture_EM_posterior!(μs, σs, ws, hs, xs)
    N, K = size(hs)
    @argcheck K == length(μs) && K == length(σs) && K == length(ws)
    @argcheck N == length(xs)
    for k in 1:K
        hs[:, k] .= ws[k] * pdf.(Normal(μs[k], σs[k]), xs)
    end
    row_sums = sum(hs, 2)
    mix_likelihood = sum(log.(row_sums))
    hs .= hs ./ row_sums
    mix_likelihood
end

@testset "testing posterior probabilities" begin
    # Testing the normal_mixture_EM_posterior with a *one* mixture normal.
    # It should give back hs with 1s.

    # starting with hs = zeros
    hs = zeros(100, 1)
    xs = rand(Normal(1.0, 0.2), 100)
    # the function updates the hs matrix
    ℓ = normal_mixture_EM_posterior!([1.2], [0.5], [1.0], hs, xs)
    # testing whether we got back a vector of ones or not
    @test hs == ones(length(hs), 1)
    @test ℓ ≈ sum(logpdf.(Normal(1.2, 0.5), xs))
end

function normal_mixture_crude_init(K, xs)
    N = length(xs)
    μ, σ = mean_and_std(xs)
    if K == 1
        μs = [μ]
    else
        μs = collect(μ + σ * linspace(-1, 1, K))
    end
    σs = fill(σ, K)
    ws = fill(1/K, K)
    hs = Array{Float64}(N, K)
    ℓ = normal_mixture_EM_posterior!(μs, σs, ws, hs, xs)
    μs, σs, ws, hs, ℓ
end

@testset "normal mixture EM iteration" begin
    N = 1000
    for _ in 1:100
        K = rand(3:6)
        dist = normal_mixture(randn(K), abs.(randn(K)) + 1, normalize(abs.(randn(K) + 1), 1))
        xs = rand(dist, N)
        μs, σs, ws, hs, ℓ = normal_mixture_crude_init(3, xs)
        for _ in 1:100
            normal_mixture_EM_parameters!(μs, σs, ws, hs, xs)
            ℓ′ = normal_mixture_EM_posterior!(μs, σs, ws, hs, xs)
            @test ℓ' ≥ ℓ            # test that loglikelihood is always increasing
            ℓ = ℓ′
        end
    end
end

"""
    normal_mixture_EM(xs, K; maxiter = 1000, tol=√eps())

Given observations `xs`, estimate a mixture of `K` normals.

Do at most `maxiter` iterations. Convergence stops when the
log-likelihood increases by less than `tol`.

Return `ℓ, μs, σs, ws, hs, iter`, where ℓ is the log likelihood

Implementing the algorithm as described in 
[[https://en.wikipedia.org/wiki/Mixture_model#Expectation_maximization_.28EM.29]].

The function uses the Expectation Maximization algorithm to update the
parameters of the Gaussian Mixture Model, namely 'μs, σs and the
weights' the function also gives back the loglikelihood of the Mixture
Model with the updated parameters.
"""
function normal_mixture_EM(xs, K; maxiter = 1000, tol = eps())
    μs, σs, ws, hs, ℓ = normal_mixture_crude_init(K, xs)
    normal_mixture_EM!(μs, σs, ws, hs, xs; maxiter = maxiter, tol = tol, ℓ = ℓ)
end

"""
Same as `normal_mixture_EM`, but with parameters provided as buffers.

Starts with the maximization step (updating the parameters), expects valid `hs`.

Can be used to save allocation time (with preallocated buffers), or start the
algorithm from a known value (eg in MCMC, the previous iteration).
"""
function normal_mixture_EM!(μs, σs, ws, hs, xs;
                            maxiter = 1000, tol = eps(),
                            ℓ = normal_mixture_EM_posterior!(μs, σs, ws, hs, xs))
    iter = 0
    while iter < maxiter
        normal_mixture_EM_parameters!(μs, σs, ws, hs, xs)
        ℓ′ = normal_mixture_EM_posterior!(μs, σs, ws, hs, xs)
        ℓ′ < ℓ && warn("decreasing likelihood, this should not happen")
        ℓ′-ℓ ≤ tol && break
        ℓ = ℓ′
        iter += 1
    end
    iter == maxiter &&
        warn("reached maximum number of iterations without convergence")
    p = sortperm(μs)
    ℓ, μs[p], σs[p], ws[p], hs, iter
end

@testset "one-component mixture of normal EM" begin
    xs = rand(Normal(2.5, 0.8), 10000) # just one component, K = 1
    ℓ, μs, σs, ws, hs, iter = normal_mixture_EM(xs, 1)
    @test mean(xs) ≈ μs[1]
    @test std(xs, corrected = false) ≈ σs[1]
    @test iter ≤ 5             # should get there quickly, no updates to hs
end

@testset "mixture of normal EM" begin
    # generating the real mixture density
    μs = [1.2, 3.7, 12.3]
    σs = [0.4, 1.1, 4.3]
    ws = [0.3, 0.1, 0.6]
    xs = rand(normal_mixture(μs, σs, ws), 100000)
    ℓ, μe, σe, we, _, iter = normal_mixture_EM(xs, 3)
    @test norm(μs-μe, 1) ≤ 0.1
    @test norm(σs-σe, 1) ≤ 0.1
    @test norm(ws-we) ≤ 0.1
    @test iter < 500
end
