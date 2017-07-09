using Distributions
using Parameters
using HamiltonianABC
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
        xs = transform_standard_normal.(gk, ϵ)
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

pp = ToyQuantProblem(ys,
                     Uniform(0, 1), Uniform(0, 1), Uniform(-5, 5), Uniform(0, 10),
                     √eps(), 3, randn(1000))
chain, a = mcmc(RWMH(diagm([0.02, 0.02, 0.02, 0.02])), pp, θ, 10000)

