using Distributions
using Parameters
using HamiltonianABC
using ContinuousTransformations
using Plots
using JLD
import HamiltonianABC: logdensity, simulate!

# remove this when adding to tests
using Base.Test
cd(Pkg.dir("HamiltonianABC", "test"))
include("utilities.jl")
include("EM_GaussianMixtureModel.jl")
include("g_and_k_quantile.jl")

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
    "a different hs for y"
    hs_y
    "number of maximum iterations"
    maxiter::Integer
end

"""
    ToyQuantModel(ys, prior_a, prior_b, prior_g, prior_k, K, M;
                  [maxiter], [likelihood_tol])

Convenience constructor with sensible default values.
"""
function ToyQuantModel(ys, prior_a, prior_b, prior_g, prior_k, K, M;
                       maxiter = 1000, likelihood_tol = √eps())
    ϵ = randn(M)
    μs, σs, ws, hs_y, _ = normal_mixture_crude_init(K, ys)
    hs = zeros(M, K)
    ToyQuantModel(ys, prior_a, prior_b, prior_g, prior_k,
                  likelihood_tol, ϵ, μs, σs, ws, hs, hs_y, maxiter)
end


"""
    bridge_trans(dist)

Take in a distribution, give back a Continuous transformation, with
image on the support of the distribution.  This is a facilitator
function.
"""
function bridge_trans(dist::Distribution{Univariate,Continuous})
    supp = support(dist)
    bridge(ℝ, minimum(supp) .. maximum(supp))
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

"""
Return a vector of transformations that map ℝ to valid parameters,
respectively.

Order of parameters is a, b, g, k.
"""
parameter_transformations(pp::ToyQuantModel) =
    bridge_trans.([pp.prior_a, pp.prior_b, pp.prior_g, pp.prior_k])

simulate!(pp::ToyQuantModel) = randn!(pp.ϵ)

function logdensity(pp::ToyQuantModel, θ)
    @unpack ys, ϵ, likelihood_tol, prior_a, prior_b, prior_g, prior_k, μs, σs, ws, hs, hs_y, maxiter, likelihood_tol = pp

    # transforming parameters
    trans = parameter_transformations(pp)
    value_and_logjac = [t(raw_θ, LOGJAC) for (t, raw_θ) in zip(trans, θ)]
    a, b, g, k = first.(value_and_logjac)

    #println("a=$a, b=$b, g=$g, k=$k")
    logprior = logpdf(prior_a, a) + logpdf(prior_b, b)  + logpdf(prior_g, g) +
        logpdf(prior_k, k) + sum(last, value_and_logjac)

    gk = GandK(a, b, g, k)
    xs = transform_standard_normal.(gk, ϵ)
    #println("x=$xs")
    #println("μs=$μs")
    #println("σs=$σs")
    #println("ws=$ws")
    #save("/tmp/lastparams.jld", Dict("xs" => xs, "μs" => μs, "σs" => σs,
                                     "ws" => ws, "hs" => hs))
    ℓ = normal_mixture_crude_init!(μs, σs, ws, hs, xs)
    ℓ, μs, σs, ws, hs, iter = normal_mixture_EM!(μs, σs, ws, hs, xs;
                                                 maxiter = maxiter,
                                                 tol = likelihood_tol,
                                                 ℓ = ℓ)
    loglikelihood = normal_mixture_EM_posterior!(μs, σs, ws, hs_y, ys)

    loglikelihood + logprior
end

a, b, g, k = 0.3, 0.5, 2.0, 3.1
ys = rand(GandK(a, b, g, k), 50)

pp = ToyQuantModel(ys, Uniform(0, 1), Uniform(0, 2), Uniform(0, 5), Uniform(0, 10),
                   3, 50)
θ₀ = [inv(t)(param)
      for (t,param) in zip(parameter_transformations(pp), [a, b, g, k])]
# this currently breaks
chain, a = mcmc(RWMH(diagm([0.01, 0.01, 0.01, 0.01])), pp, θ₀, 1000)

# analysis of what happens
lastparams = load("/tmp/lastparams.jld")
μs, σs, ws, hs, xs = getindex.(lastparams, ["μs","σs","ws","hs","xs"])

normal_mixture_EM(xs, 3)               # works fine
normal_mixture_EM!(μs, σs, ws, hs, xs) # fails

########## I stopped here ##########

# saving the transformed values
r = length(chain)
a_sample = Vector(r)
b_sample = Vector(r)
g_sample = Vector(r)
k_sample = Vector(r)
q, w, e, z = [parameter_transformations(pp)...]
for i in 1:r
    a_sample[i] = q(chain[i][1])
    b_sample[i] = w(chain[i][2])
    g_sample[i] = e(chain[i][3])
    k_sample[i] = z(chain[i][4])
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
    @test mean(a_sample) ≈ a rtol = 0.15
    @test mean(b_sample) ≈ b rtol = 0.3
    @test mean(g_sample) ≈ g rtol = 0.5
    @test mean(k_sample) ≈ k rtol = 0.15
end
