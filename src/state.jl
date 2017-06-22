export
    SimulatedState,
    get_θ,
    simulate_state

"""
An implementation for simulated states that saves the (approximate) posterior
value.
"""
struct SimulatedState{Tθ, Tϵ, Tℓ <: Real}
    "Parameter."
    θ::Tθ
    "Shocks."
    ϵ::Tϵ
    "Saved value of the posterior."
    ℓ::Tℓ
end

"""
    get_θ(state)

Return the parameter vector for `state`.
"""
get_θ(state::SimulatedState) = state.θ

"""
    logposterior(problem, state[, θ])

Return the log posterior (log likelihood + log prior) for `problem` at `state`,
evaluated at parameters `θ`, up to an additive constant that is fixed for
`problem`.

When `θ` is omitted, parameters are taken from the state (which may cache this
information, leading to faster calculation).
"""
logposterior(problem, state::SimulatedState) = state.ℓ

function logposterior(problem, state::SimulatedState, θ)
    calculate_posterior(problem, θ, state.ϵ)
end

"""
    simulate_state(problem, θ[, ϵ])

Generate a state for `problem` with parameters `θ`. `ϵ` is used when
provided, otherwise it is simulated.
"""
function simulate_state(problem, θ, ϵ = simulate_ϵ(problem))
    SimulatedState(θ, ϵ, calculate_posterior(problem, θ, ϵ))
end
