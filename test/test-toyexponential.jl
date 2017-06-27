######################################################################
# Exponential with normal auxiliary model
######################################################################

"""
The true model is y ∼ Exponential(λ), IID, where λ is the scale.

The auxiliary model is y ∼ N(μ, σ²), with statistics ϕ = (μ, σ).

The prior is λ ∼ Uniform(A,B) prior.
"""
struct ToyExponentialModel
    "observed data"
    y::Vector{Float64}
    "lower boundary of the uniform prior distribution"
    A::Float64
    "upper boundary of the uniform prior distribution"
    B::Float64
    """
    Uniform(0,1) draws, mapped to exponential draws using the
    CDF. For the empty vector, the limiting statistics are used.
    """
    ϵ::Vector{Float64}
end

"""
Convenience constructor for ToyExponentialModel, true value `λ`, `N`
draws, `M` simulated values.
"""
function ToyExponentialModel(λ::Float64, N::Int, M::Int)
    ToyExponentialModel(rand(Exponential(λ), N), 0.0, 2*λ, rand(M))
end
 
function logdensity(p::ToyExponentialModel, θ)
    @unpack y, A, B, ϵ = p
    logλ, = θ
    λ = exp(logλ)
    # log Jacobian is ``log(|exp(logλ)|) = logλ``, hence the ``+ logλ``.
    logprior = logpdf(Uniform(A, B), λ) + logλ
    dist = Exponential(λ)
    if isempty(ϵ)
        μ = mean(dist)
        σ² = var(dist)
    else
        μ, σ² = mean_and_var(quantile.(dist, ϵ); corrected = false)
    end
    loglikelihood = sum(logpdf.(Normal(μ, √σ²), y))
    loglikelihood + logprior
end

simulate!(p::ToyExponentialModel) = rand!(p.ϵ)

"Analytical distribution for the problem."
function analytical_distribution(p::ToyExponentialModel)
    @unpack y, A, B = p
    Truncated(InverseGamma(length(y), sum(y)), A, B)
end

function limit_posterior(p::ToyExponentialModel)
    @unpack y, A, B = p
    N = length(y)
    ℓ(λ) = -N*log(λ) - sum((y./λ-1).^2)/2 # log posterior
    exp ∘ normalize_logdensity(ℓ, mean(y), A, B)
end
                   
@testset "toy exponential limit" begin
    λ = 2.0
    p = ToyExponentialModel(λ, 100, 0)
    posterior = limit_posterior(p)
    
    @test hquadrature(posterior, p.A, p.B)[1] ≈ 1
    
    chain, a = mcmc(RWMH(diagm([0.02])), p, [log(λ)], 10000)
    λs = exp.(first.(chain))

    @test mean(λs) ≈ mean(p.y) rtol = 0.15
    
    test_cdf(pdf2cdf(limit_posterior(p), p.A), λs)
end

@testset "toy exponential 10x replication" begin
    λ = 3.0
    p = ToyExponentialModel(λ, 100, 1000)
    
    chain, a = mcmc(RWMH(diagm([0.02])), p, [log(λ)], 10000)
    λs = exp.(first.(chain))

    @test mean(λs) ≈ mean(p.y) rtol = 0.15
    
    test_cdf(pdf2cdf(limit_posterior(p), p.A), λs; atol = 0.1)
end
