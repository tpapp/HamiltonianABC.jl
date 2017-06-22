export
    DensityWrapper,
    StateWrapper

"""
A wrapper for sampling from a given log density.

Useful for testing MCMC.
"""
struct DensityWrapper{T}
    "Log density function."
    logdensity::T
end

"""
A wrapper for a state *without* simulated components (just the
parameter).

Useful for testing MCMC.
"""
struct StateWrapper{T}
    θ::T
end

get_θ(sw::StateWrapper) = sw.θ

logposterior(dw::DensityWrapper, state::StateWrapper) = dw.logdensity(get_θ(state))

logposterior(dw::DensityWrapper, state, θ′) = dw.logdensity(θ′)

simulate_state(dw::DensityWrapper, θ) = StateWrapper(θ)
