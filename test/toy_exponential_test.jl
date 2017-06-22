using Distributions
using HamiltonianABC

include("../examples/toy_exponential_problem.jl")

@testset "toy exponential" begin

    λ = 5.0
    pp = ToyExponentialProblem(rand(Exponential(λ), 100), 0.0, 10.0, 1000)
    state = simulate_state(pp, [log(λ)])
    chain, a = mcmc(RWMH(fill(0.05,1,1)), pp, state, 10000)
    
    λs = get_λ.(get_θ.(chain))

    @test mean(λs) ≈ mean(pp.y) atol = 0.5
    
end
