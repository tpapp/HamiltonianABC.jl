using Plots; plotlyjs()
using StatPlots
using HamiltonianABC
using KernelDensity

include(Pkg.dir("HamiltonianABC", "examples", "toy_exponential_problem.jl"))

pp = ToyExponentialProblem(rand(Exponential(1), 100), 0, 10, 1000)

chain, a = mcmc(RWMH(fill(0.1,1,1)), pp, simulate_state(pp, [log(10)]), 10000)
λs = get_λ.(get_θ.(chain))
dens = kde(λs)

plot(x->pdf(dens, x), xlim = (0,2), label = "posterior from RWMH")
plot!(analytical_posterior(pp), label = "true posterior")
vline!([mean(pp.y)], label = "ML")
vline!([mean(λs)], label = "posterior mean")

(var(λs)/var(rand(analytical_distribution(pp), 100000); corrected = false))^2
