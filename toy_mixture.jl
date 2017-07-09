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
    Xs = normal_gk_quant.(z, θ...)
    # estimating parameters for the auxiliary model
    params_quant = normal_mixture_EM(Xs, mix)
    prob_matrix = unit_row_matrix(length(y), mix)
    # loglikelihood with the estimated parameters and the ´observed´ points
    log_likelihood = normal_mixture_EM_posterior!(params_quant[2], params_quant[3], params_quant[4], prob_matrix , y)

    log_likelihood + logprior

end


simulate!(pp::ToyQuantProblem) = rand!(pp.ϵ)

# true parameters
χ = [3.05,1.0,2.0,3.1]
# structure
pp = ToyQuantProblem(map(x -> gk_quant(x, χ...), rand(10000)), 0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 10.0, 10000, eps(), 3, rand(1000))

# estimation
chain, a = mcmc(RWMH(diagm([0.02, 0.02, 0.02, 0.02])), pp, [1.1, 0.9, 1.2, 1.4], 20000)
mean(chain) - χ
