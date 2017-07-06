using Distributions
using StatsBase
using Parameters
using HamiltonianABC
import HamiltonianABC: logdensity, simulate!

struct ToyQuantProblem
    "observed data"
    y::Vector{Float64}
    "lower boundary of the uniform prior distribution"
    Al::Float64
    "upper boundary of the uniform prior distribution"
    Au::Float64
    "lower boundary of the uniform prior distribution"
    Bl::Float64
    "upper boundary of the uniform prior distribution"
    Bu::Float64
    "lower boundary of the uniform prior distribution"
    Gl::Float64
    "upper boundary of the uniform prior distribution"
    Gu::Float64
    "lower boundary of the uniform prior distribution"
    Kl::Float64
    "upper boundary of the uniform prior distribution"
    Ku::Float64
    "number of draws for simulated data"
    M::Integer
    "approx closeness"
    tol::Float64
    "number of mixtures"
    mix::Integer
    "Uniform(0,1) draws"
    ϵ::Vector{Float64}
end

function logdensity(pp::ToyQuantProblem, θ)

    @unpack y, Al, Au, Bl, Bu, Gl, Gu, Kl, Ku,  M, tol, mix, ϵ = pp

    logprior = logpdf(Uniform(Al, Au), θ[1]) + logpdf(Uniform(Bl, Bu), θ[2]) + logpdf(Uniform(Gl, Gu), θ[3]) + logpdf(Uniform(Kl, Ku), θ[4])

    z = quantile.(Normal(0, 1), ϵ)
    X = normal_gk_quant.(z, θ...)

    params_quant = normal_mixture_EMM(X, mix, tol)

    loglikelihood = normal_mixture_EM_posterior!(params_quant[2], params_quant[3], params_quant[4], params_quant[5], y)

    loglikelihood + logprior

end

simulate!(pp::ToyQuantProblem) = rand!(pp.ϵ)
Θ = [0.3,0.5,2.0,3.1]
pp = ToyQuantProblem(map(x -> gk_quant(x, Θ...), rand(Uniform(0.1, 0.9), 1000)), 0.0, 1.0, 0.0, 1.0, -5.0, 5.0, 0.0, 10.0, 10000, eps(), 3, rand(Uniform(0.1, 0.9), 1000))
chain, a = mcmc(RWMH(diagm([0.02, 0.02, 0.02, 0.02])), pp, Θ  , 20000)
