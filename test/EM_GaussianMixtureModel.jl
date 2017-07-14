using Distributions
using Parameters
using Base.Test
using StatsBase
using ArgCheck
import StatsFuns: logsumexp

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
    ws = [0.1, 0.2, 0.7]
    simulated_mean = mean(rand(normal_mixture(μs, σs, ws), 10000))
    @test simulated_mean ≈ dot(μs, ws) atol = 0.1
end

"""
    is_compatible_arguments(μs, σs, logws, loghs, xs)

Test if the dimensions of the arguments are compatible.
"""
function is_compatible_arguments(μs, σs, logws, loghs, xs)
    N, K = size(loghs)
    (K == length(μs) == length(σs) == length(logws)) && (N == length(xs))
end

"""
    normal_mixture_EM_parameters!(μs, σs, logws, loghs, xs)

Maximization step. Update the means `μs`, standard deviations `σs` and
log weights `logws`, given the log posterior probabilities `loghs` and
data `xs`.

`loghs[i, j]` is the posterior probability of component `j` for
observation `i`.

Note that because of convergence problems with isolated and/or
repeated points (practically 0 variance), we correct σs by the machine
ϵ. For the description of the problem and other possible methods, see

Archambeau, Cédric, John Aldo Lee, and Michel Verleysen. "On
Convergence Problems of the EM Algorithm for Finite Gaussian
Mixtures." ESANN. Vol. 3. 2003.

Yang, Zheng Rong, and Sheng Chen. "Robust maximum likelihood training
of heteroscedastic probabilistic neural networks." Neural Networks
11.4 (1998): 739-747.
"""
function normal_mixture_EM_parameters!(μs, σs, logws, loghs, xs)
    N, K = size(loghs)
    @argcheck is_compatible_arguments(μs, σs, logws, loghs, xs)
    for j in 1:K
        logh = @view loghs[:, j]
        log∑h = logsumexp(logh)
        logws[j] = log∑h - log(N)
        weights = exp.(logh - log∑h)
        μs[j] = dot(xs, weights)
        σs[j] = √sum(weights .* (xs - μs[j]).^2) + eps()
    end
    nothing
end

@testset "not mixture" begin
    ## testing the normal_mixture_EM_parameters. If the input hs matrix
    ## contains ones in one column and zeros in the others, then it should
    ## give back the actual distribution'

    # building hs with 1s in the first column and zeros in the others
    loghs = log.([ones(1000) zeros(1000, 2)])
    # drawing normally distributed numbers
    xs = rand(Normal(1.2, 0.4), 1000)
    μs = zeros(3)
    σs = zeros(3)
    logws = zeros(3)
    # estimating the parameters
    normal_mixture_EM_parameters!(μs, σs, logws, loghs, xs)
    # density with estimated parameters
    @test logws[1] == 0
    @test mean(xs) ≈ μs[1]
    @test std(xs, corrected = false) ≈ σs[1]
end

"""
    normal_mixture_EM_posterior!(μs, σs, logws, loghs, xs)

Expectation step: update the `loghs` matrix given the parameters `μs`,
`σs` and `logws` of the Gaussian Mixture Model. See
`normal_mixture_EM_parameters!` for variable name and index
conventions.

Return the (marginalized) log likelihood of the mixture model.
"""
function normal_mixture_EM_posterior!(μs, σs, logws, loghs, xs)
    N, K = size(loghs)
    @argcheck is_compatible_arguments(μs, σs, logws, loghs, xs)
    for k in 1:K
        loghs[:, k] .= logws[k] + logpdf.(Normal(μs[k], σs[k]), xs)
    end
    log_likelihood = [logsumexp(loghs[i, :]) for i in indices(loghs, 1)]
    loghs .-= log_likelihood
    sum(log_likelihood)
end

@testset "testing posterior probabilities" begin
    # Testing the normal_mixture_EM_posterior with a *one* mixture normal.
    # It should give back hs with 1s.

    N = 100
    loghs = rand(N, 1)        # random matrix, to see if it is overwritten
    xs = rand(Normal(1.0, 0.2), N)
    # the function updates the hs matrix
    ℓ = normal_mixture_EM_posterior!([1.2], [0.5], [0.0], loghs, xs)
    @test loghs == zeros(N, 1)
    @test ℓ ≈ sum(logpdf.(Normal(1.2, 0.5), xs))
end

"""
    ℓ = normal_mixture_crude_init!(μs, σs, logws, loghs, xs)

Sensible but crude initialization for normal mixtures, modifies the
first four arguments, returns log likelihood.
"""
function normal_mixture_crude_init!(μs, σs, logws, loghs, xs)
    N, K = size(loghs)
    @argcheck is_compatible_arguments(μs, σs, logws, loghs, xs)
    μ, σ = mean_and_std(xs)
    if K == 1
        μs .= [μ]
    else
        μs .= μ + σ * linspace(-1, 1, K)
    end
    σs .= σ
    logws .= log(1/K)
    normal_mixture_EM_posterior!(μs, σs, logws, loghs, xs)
end

"""
    μs, σs, logws, loghs, ℓ = normal_mixture_crude_init(K, xs)

Sensible but crude initialization for normal mixtures. Allocates the
relevant arrays, see signature for return value.
"""
function normal_mixture_crude_init(K, xs)
    N = length(xs)
    μs = zeros(K)
    σs = zeros(K)
    logws = Vector{Float64}(K)
    loghs = Array{Float64}(N, K)
    ℓ = normal_mixture_crude_init!(μs, σs, logws, loghs, xs)
    μs, σs, logws, loghs, ℓ
end

@testset "normal mixture EM iteration" begin
    N = 1000
    for _ in 1:100
        K = rand(3:6)
        dist = normal_mixture(randn(K), abs.(randn(K)) + 1, normalize(abs.(randn(K) + 1), 1))
        xs = rand(dist, N)
        μs, σs, logws, loghs, ℓ = normal_mixture_crude_init(3, xs)
        for _ in 1:100
            normal_mixture_EM_parameters!(μs, σs, logws, loghs, xs)
            ℓ′ = normal_mixture_EM_posterior!(μs, σs, logws, loghs, xs)
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

Return `ℓ, μs, σs, logws, loghs, iter, conv`, where:

1. ℓ is the log likelihood,
2. μs, σs are the means and variances of the normal components
3. logws are log weights,
4. loghs are the log posterior,
5. iter is the number of iterations

When iter == maxiter, the algorithm did not converge.

Implementing the algorithm as described in
[[https://en.wikipedia.org/wiki/Mixture_model#Expectation_maximization_.28EM.29]].

The function uses the Expectation Maximization algorithm to update the
parameters of the Gaussian Mixture Model, namely 'μs, σs and the
weights' the function also gives back the loglikelihood of the Mixture
Model with the updated parameters.
"""
function normal_mixture_EM(xs, K; maxiter = 1000, tol = eps())
    μs, σs, logws, loghs, ℓ = normal_mixture_crude_init(K, xs)
    normal_mixture_EM!(μs, σs, logws, loghs, xs; maxiter = maxiter, tol = tol, ℓ = ℓ)
end

"""
    normal_mixture_EM!(μs, σs, logws, loghs, xs; [maxiter], [tol], [ℓ])

Same as `normal_mixture_EM`, but with parameters provided as buffers.

Starts with the maximization step (updating the parameters), expects
valid `loghs`, however, when ℓ is not provided and calculated, this is
enforced automatically.

Can be used to save allocation time (with preallocated buffers), or
start the algorithm from a known value (eg in MCMC, the previous
iteration).
"""
function normal_mixture_EM!(μs, σs, logws, loghs, xs;
                            maxiter = 1000, tol = eps(),
                            ℓ = normal_mixture_EM_posterior!(μs, σs, logws, loghs, xs))
    iter = 0
    while iter < maxiter
        normal_mixture_EM_parameters!(μs, σs, logws, loghs, xs)
        ℓ′ = normal_mixture_EM_posterior!(μs, σs, logws, loghs, xs)
        ℓ′ < ℓ && warn("decreasing likelihood, this should not happen")
        ℓ′-ℓ ≤ tol && break
        ℓ = ℓ′
        iter += 1
    end
    iter == maxiter &&
        warn("reached maximum number of iterations without convergence")
    p = sortperm(μs)
    ℓ, μs[p], σs[p], logws[p], loghs, iter
end

@testset "one-component mixture of normal EM" begin
    xs = rand(Normal(2.5, 0.8), 10000) # just one component, K = 1
    ℓ, μs, σs, logws, loghs, iter = normal_mixture_EM(xs, 1)
    @test mean(xs) ≈ μs[1]
    @test std(xs, corrected = false) ≈ σs[1]
    @test iter ≤ 5        # should get there quickly, no updates to hs
end

@testset "mixture of normal EM" begin
    # generating the real mixture density
    μs = [1.2, 3.7, 12.3]
    σs = [0.4, 1.1, 4.3]
    ws = [0.3, 0.1, 0.6]
    xs = rand(normal_mixture(μs, σs, ws), 100000)
    ℓ, μe, σe, we, _, iter = normal_mixture_EM(xs, 3)
    @test norm(μs - μe, 1) ≤ 0.1
    @test norm(σs - σe, 1) ≤ 0.1
    @test norm(ws - exp.(we)) ≤ 0.1
    @test iter < 500
end

@testset "mixture with outliers" begin
    xs = [12.932452188340273, 2.2016456675666727, 0.23196720150275324, 1.3691367352423272,
          -1.9233294652628463, 0.5309730139202926, -16.3914493973905, 0.5135149269904411,
          0.5004869515882554, -0.4249034374632298, -16.309097459297387, 1.8273014356996606,
          0.9068012606137947, 0.18502720824389784, 0.611191325556077, 0.33694679175260556,
          0.41909904351062194, 0.3183507750621336, 3.4788126363867646, 0.650440006783765,
          -0.2297950276864753, 0.037752598867667464, 0.6883596258902946, 0.016175910469082733,
          -0.43895880066117143, 1162.2229232994482, 1.8295627371522625, 4.234673977355862,
          0.3435988754507263, -0.22757689294492156, -0.10570769651003853, 1.388937436866165,
          0.6491483010788963, 0.629879050472239, 1.281837862100565, -0.17399472202932442,
          0.5718223728848091, -0.8792641804249895, 1.827421833992593, 0.38218451185136754,
          0.016510405992137323, 0.06351884450746353, 0.3643079903721287, 0.452080027779106,
          -1.090084082339093, 1.4366196407002945, 0.28103662388081546, -0.22474872099544618,
          -0.3177668023128347, 0.13551948074007314]
    ℓ, μs, σs, logws, _, iter = normal_mixture_EM(xs, 3)
    @test maximum(abs.(μs[1:2])) ≤ 2
    @test μs[3] ≥ 1000          # the outlier
    @test minimum(abs.(σs[1:2])) ≥ 0.5
    @test σs[3] ≤ 2*eps()       # the outlier
    @test iter ≤ 50
end
