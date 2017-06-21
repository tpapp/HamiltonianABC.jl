using Distributions

export
    Proposal,
    propose,
    mcmc,
    RWMH

"""
A Metropolis-Hastings proposal, with the current state and the new
parameters.
"""
struct Proposal{Ts,Tθ,Ta}
    "The current state, contains the parameter too."
    state::Ts
    "Proposal."
    θ′::Tθ
    "Acceptance probability for proposal."
    a::Ta
end

"""
Random walk Metropolis-Hastings.

In this library, it is provided purely for testing and pedagogical
purposes. See [`NUTS`](@ref) for a more practical alternative.
"""
struct RWMH{S <: AbstractMatrix}
    "Covariance matrix for proposals."
    Σ::S
end

"""
Generate a Metropolis-Hastings proposal using the given `transition`
algorithm and `problem`, starting from `state`.

The methods `get_θ(state)` and `logposterior(problem, state[, θ])`
need to be supported by the arguments.
"""
function propose(transition::RWMH, problem, state)
    θ′ = rand(MvNormal(get_θ(state), transition.Σ))
    a = min(1, exp(logposterior(problem, state, θ′) - logposterior(problem, state)))
    Proposal(state, θ′, a)
end

"""
Runs a Markov Chain Monte Carlo algorithm by calling [`propose`](@ref)
repeatedly `N` times.

Return the vector of states, and the average acceptance rate.
"""
function mcmc{T}(transition, problem, state::T, N)
    chain = Vector{T}(N)
    as = Vector{Float64}(N)
    for i in 1:N
        p = propose(transition, problem, state)
        if rand() ≤ p.a
            state = simulate_state(problem, p.θ′)
        end
        as[i] = p.a
        chain[i] = state
    end
    chain, mean(as)
end
