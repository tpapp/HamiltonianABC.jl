######################################################################
# Numerical experiment: estimate a normal with known variance, noise
# taken from a Sobol sequence
######################################################################
import HamiltonianABC: logprior,
    simulate_ϵ, generate_x, estimate_ϕ, data_loglikelihood
using Plots; plotlyjs()
using HamiltonianABC
using KernelDensity
using Parameters
using Sobol
using Distributions
import StatsFuns: norminvcdf

struct ToyNormalProblem
    "observed data"
    y::Vector{Float64}
    "draws for simulated data"
    ϵ::Vector{Float64}
end

function ToyNormalProblem(y, M::Int)
    ToyNormalProblem(y, [norminvcdf(x[1]) for x in Base.Iterators.take(SobolSeq(1), M)])
end

logprior(::ToyNormalProblem, θ) = 0.0

simulate_ϵ(problem::ToyNormalProblem) = problem.ϵ # reuse

generate_x(problem::ToyNormalProblem, θ, ϵ) = θ[1] + ϵ

estimate_ϕ(problem::ToyNormalProblem, x) = [mean(x)]

data_loglikelihood(problem::ToyNormalProblem, ϕ) = -sum(abs2, problem.y-ϕ[1])/2

function analytical_posterior(problem::ToyNormalProblem)
    @unpack y = problem
    dist = Normal(mean(y), 1/√length(y))
    μ -> pdf(dist, μ)
end

y = rand(100)
xs = linspace(-0.5,0.5,100)+mean(y)
p = ToyNormalProblem(y, 1)
plt = plot(xs, analytical_posterior(p).(xs), label = "true posterior",
           title = "Bayesian indirect inference w/ Sobol")
for m in 2:4
    M = 10^m
    p = ToyNormalProblem(y, M)
    chain, a = mcmc(RWMH(fill(0.1,1,1)), p, simulate_state(p, [0.0]), M)
    dens = kde(first.(get_θ.(chain)))
    plot!(plt, xs, pdf.(dens, xs), label = "M=$(M)")
end
display(plt)
