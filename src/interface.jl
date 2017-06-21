export
    simulate_ϵ,
    generate_x,
    logprior,
    estimate_ϕ,
    data_loglikelihood

"""
    simulate_ϵ([rng=GLOBAL_RNG,] problem)

Return random noise for the problem, in a compatible format. Noise is reused
when calculating likelihoods around a state. Importantly, noise is combined with
parameters to generate observations.
"""
function simulate_ϵ end

"""
    generate_x(problem, θ, ϵ)

Generate observations for `problem` using parameters `θ` and noise `ϵ`.
"""
function generate_x end

"""
    logprior(problem, θ)

Return the log prior density for `problem` at parameters `θ`.
"""
function logprior end

"""
    estimate_ϕ(problem, x)

Return the maximum likelihood estimate ``ϕ`` of the auxiliary model for
`problem` given observations `x`.
"""
function estimate_ϕ end

"""
    data_loglikelihood(problem, ϕ)

Return the log likelihood of the observed data of `problem` under the
auxiliary model with parameters `ϕ`.
"""
function data_loglikelihood end

"""
Helper function for calculating posterior values. Follows Gallant, A. R., &
McCulloch, R. E. (2009).
"""
function calculate_posterior(problem, θ, ϵ)
    x = generate_x(problem, θ, ϵ)
    ϕ = estimate_ϕ(problem, x)
    data_loglikelihood(problem, ϕ) + logprior(problem, θ)
end

