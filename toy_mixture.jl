using Distributions
using StatsBase
using Parameters
using HamiltonianABC
import HamiltonianABC: logdensity, simulate!

type ToyQuantProblem
    "observed data"
    y::Vector{Float64}
    "lower boundary of the uniform prior distribution"
    al::Float64
    "upper boundary of the uniform prior distribution"
    au::Float64
    "lower boundary of the uniform prior distribution"
    bl::Float64
    "upper boundary of the uniform prior distribution"
    bu::Float64
    "lower boundary of the uniform prior distribution"
    gl::Float64
    "upper boundary of the uniform prior distribution"
    gu::Float64
    "lower boundary of the uniform prior distribution"
    kl::Float64
    "upper boundary of the uniform prior distribution"
    ku::Float64
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

    @unpack y, al, au, bl, bu, gl, gu, kl, ku, M, tol, mix, ϵ = pp

    logprior = sum(logpdf(Uniform(al, au), Θ[1]) + logpdf(Uniform(bl, bu), Θ[2]) + logpdf(Uniform(gl, gu), Θ[3]) + logpdf(Uniform(kl, ku), Θ[4]))

    X = gk_quant.(ϵ, Θ...)
    params_quant = normal_mixture_EMM(X, mix)

    loglikelihood = normal_mixture_EM_posterior!(params_quant[2], params_quant[3], params_quant[4], params_quant[5], y)

    loglikelihood + logprior

end

simulate!(p::ToyQuantProblem) = rand!(p.ϵ)
Θ = [1.0,2.0,2.0,1.0]
cdf_samples(x) = get_cdf_quant(gk_quant(x,Θ...), Θ)

pp = ToyQuantProblem(map(x -> gk_quant(x, Θ...), rand(1000)),0.0,5.0,0.0,5.0,0.0,5.0,0.0,10.0,1000,eps(),3, rand(1000))
χ = [2.1, 2.4, 1.1, 2.7]
chain, a = mcmc(RWMH(diagm([0.02, 0.02, 0.02, 0.02])), pp, χ, 500)
mean(first.(chain))
plot(first.(chain))
