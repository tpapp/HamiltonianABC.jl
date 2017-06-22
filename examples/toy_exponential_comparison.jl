using Plots; plotlyjs()
using StatPlots
using HamiltonianABC

include(Pkg.dir("HamiltonianABC", "examples", "toy_exponential_problem.jl"))

pp = ToyExponentialProblem(rand(Exponential(5), 50), 0, 10, 500)

chain, a = mcmc(RWMH(fill(0.1,1,1)), pp, simulate_state(pp, [log(10)]), 10000)
λs = get_λ.(get_θ.(chain))

plot(density(λs), xlim = (pp.A,pp.B), label = "posterior from RWMH")
plot!(analytical_posterior(pp),  label = "true posterior")
vline!([mean(pp.y)], label = "ML")
vline!([mean(λs)], label = "posterior mean")
