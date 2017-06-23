######################################################################
# toy problem: samples from an exponential, estimated using a normal
######################################################################

using Distributions
import HamiltonianABC: logprior,
    simulate_ϵ, generate_x, estimate_ϕ, data_loglikelihood
using Cubature                  # for numerical integral
using Parameters
using StatsBase

"""
The true model is ``y ∼ Exponential(λ), IID``, where `λ` is the scale.

The auxiliary model is ``y ∼ N(μ, σ²)``, with statistics ϕ = (μ, σ).

We use a ``λ ∼ Uniform(A,B)`` prior that contains the true value.

The `ϵ` are Uniform(0,1) draws, mapped to exponential draws using the CDF.
"""
struct ToyExponentialProblem
    "observed data"
    y::Vector{Float64}
    "lower boundary of the uniform prior distribution"
    A::Float64
    "upper boundary of the uniform prior distribution"
    B::Float64
    "number of draws for simulated data"
    M::Int
end

"""
Helper function, verifies that `θ` has a single element, this is is
taken as `log(λ)`, transformed to `λ` and returned.
"""
get_λ(θ) = ((logλ,) = θ; exp(logλ))

function logprior(pp::ToyExponentialProblem, θ)
    # log Jacobian is ``log(|exp(logλ)|) = logλ``, hence the ``+ θ[1]``.
    logpdf(Uniform(pp.A, pp.B), get_λ(θ)) + θ[1]
end

simulate_ϵ(pp::ToyExponentialProblem) = rand(pp.M)

generate_x(pp::ToyExponentialProblem, θ, ϵ) = quantile.(Exponential(get_λ(θ)), ϵ)

estimate_ϕ(pp::ToyExponentialProblem, x) = mean_and_var(x; corrected = false)

data_loglikelihood(pp::ToyExponentialProblem, ϕ) = sum(logpdf.(Normal(ϕ...), pp.y))

function analytical_distribution(p::ToyExponentialProblem)
    @unpack y, A, B = p
    Truncated(InverseGamma(length(y),sum(y)), A, B)
end

analytical_posterior(p) = (dist = analytical_distribution(p); x->pdf(dist, x))
