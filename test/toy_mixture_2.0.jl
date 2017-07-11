using Distributions
using Parameters
using HamiltonianABC
using ContinuousTransformations
using Plots
import HamiltonianABC: logdensity, simulate!

# remove this when adding to tests
using Base.Test
cd(Pkg.dir("HamiltonianABC", "test"))
include("utilities.jl")
include("EM_GaussianMixtureModel.jl")
include("g_and_k_quantile.jl")

struct ToyQuantProblem
    "observed data"
    ys
    "prior for a (location)"
    prior_a
    "prior for b (scale)"
    prior_b
    "prior for b (scale)"
    prior_g
    "prior for b (scale)"
    prior_k
    "convergence tolerance for (log) likelihood estimation"
    likelihood_tol::Float64
    "number of mixtures"
    K::Integer
    "Normal(0,1) draws for simulation"
    ϵ::Vector{Float64}
end

simulate!(pp::ToyQuantProblem) = randn!(pp.ϵ)

function logdensity(pp::ToyQuantProblem, θ)
    a, b, g, k = θ
    @unpack ys, prior_a, prior_b, prior_g, prior_k, ϵ, likelihood_tol, K = pp

    logprior = logpdf(prior_a, a) + logpdf(prior_b, b) + logpdf(prior_g, g) + logpdf(prior_k, k)

    try
        gk = GandK(a, b, g, k)  # will catch invalid parameters FIXME use transformations
        xs = transform_standard_normal.(gk,ϵ)
        ℓ, μs, σs, ws, hs, iter = normal_mixture_EM(xs, K; tol = likelihood_tol)
        if iter > 500
            warn("$(iter) iterations")
        end

        hs = zeros(length(ys), K)
        loglikelihood = normal_mixture_EM_posterior!(μs, σs, ws, hs, ys)

        loglikelihood + logprior
    catch
        return -Inf
    end
end

θ = [0.3, 0.5, 2.0, 3.1]
ys = rand(GandK(θ...), 1000)
logdensity(pp, θ)
pp = ToyQuantProblem(ys,
                     Uniform(0, 1), Uniform(0, 1), Uniform(-5, 5), Uniform(0, 10),
                     √eps(), 3, randn(10000))
chain, a = mcmc(RWMH(diagm([0.02, 0.02, 0.02, 0.02])), pp, θ, 10000)



struct ToyQuantModel
    "observed data"
    ys
    "prior for a (location)"
    prior_a
    "prior for b (scale)"
    prior_b
    "prior for b (scale)"
    prior_g
    "prior for b (scale)"
    prior_k
    "convergence tolerance for (log) likelihood estimation"
    likelihood_tol::Float64
    "number of mixtures"
    K::Integer
    "Normal(0,1) draws for simulation"
    ϵ::Vector{Float64}
    "initial μs"
    μs
    "initial σs"
    σs
    "initial weights"
    ws
    "initial posterior"
    hs
    "number of maximum iterations"
    maxiter::Integer
end


"""
    bridge_trans(dist)

Take in a distribution, give back a Continuous transformation, with image on the support of the distribution.
This is a facilitator function.

"""

function bridge_trans(dist)
    dist_left, dist_right = dist.a, dist.b
    dist_tr = bridge(ℝ, dist_left .. dist_right)
end

##############################################################################
## testing the facilitator function
##############################################################################
# the inverse of the transformed number should equal the original number

@testset "bridge testing" begin

    tt = bridge_trans(Uniform(0, 5))
    trans = tt(5)
    @test inv(tt)(trans) ≈ 5.0
end


simulate!(pp::ToyQuantModel) = randn!(pp.ϵ)

function logdensity(pp::ToyQuantModel, θ)
    a, b, g, k = θ
    @unpack ys, prior_a, prior_b, prior_g, prior_k, ϵ, likelihood_tol, K, μs, σs, ws, hs, maxiter = pp

    # transforming b and k
    bb = bridge_trans(prior_b)
    kk = bridge_trans(prior_k)
    b, log_b = bb(b, LOGJAC)
    k, log_k = kk(k, LOGJAC)

    logprior = logpdf(prior_a, a) + logpdf(prior_b, b)  + logpdf(prior_g, g) + logpdf(prior_k, k)  +  log_b + log_k

    gk = GandK(a, b, g, k)
    xs = transform_standard_normal.(gk,ϵ)
    ℓ, μs, σs, ws, hs, iter = normal_mixture_EM(xs, K, maxiter = maxiter)
    #normal_mixture_EM!(μs, σs, ws, hs, xs;  maxiter = maxiter, tol = likelihood_tol) #

    if iter ≥ maxiter
        return -Inf
    end
    hs = zeros(length(ys), K)
    loglikelihood = normal_mixture_EM_posterior!(μs, σs, ws, hs, ys)

    loglikelihood + logprior

end

θ = [0.3, 0.5, 2.0, 3.1]
ys = rand(GandK(θ...), 10000)
#  initializing the parameters - does not work (yet)
ξ = [0.3, bb(0.5), 2.0, kk(3.1)]
χ = [0.3, inv(bb)(0.5), 2.0, inv(kk)(3.1)]
gk = GandK(ξ...)
xs = transform_standard_normal.(gk, randn(1000))
μs, σs, ws, hs = normal_mixture_EM(xs, 3)[2:5]

pp = ToyQuantModel(ys,
                     Uniform(0, 1), Uniform(0, 2), Uniform(0, 5), Uniform(0, 10),
                     √eps(), 3, randn(1000), μs, σs, ws, hs, 1000)

bb = bridge_trans(Uniform(0, 2))
kk = bridge_trans(Uniform(0, 10))


chain, a = mcmc(RWMH(diagm([0.01, 0.01, 0.01, 0.01])), pp, χ, 10000)
# saving the transformed values
r = length(chain)
a_sample = Vector(r)
b_sample = Vector(r)
g_sample = Vector(r)
k_sample = Vector(r)
for i in 1:r
    a_sample[i] = chain[i][1]
    b_sample[i] = bb(chain[i][2])
    g_sample[i] = chain[i][3]
    k_sample[i] = kk(chain[i][4])
end
plot(a_sample)
plot(b_sample)
plot(g_sample)
plot(k_sample)
mean(a_sample)
mean(b_sample)
mean(g_sample)
mean(k_sample)

#############################################################################
## testing the estimation
##############################################################################


@testset "toy mixture " begin
    @test mean(a_sample) ≈ θ[1] rtol = 0.15
    @test mean(b_sample) ≈ θ[2] rtol = 0.3
    @test mean(g_sample) ≈ θ[3] rtol = 0.5
    @test mean(k_sample) ≈ θ[4] rtol = 0.15
end
