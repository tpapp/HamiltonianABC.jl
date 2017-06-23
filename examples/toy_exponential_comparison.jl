using Plots; plotlyjs()
using StatPlots
using HamiltonianABC
using KernelDensity

include(Pkg.dir("HamiltonianABC", "examples", "toy_exponential_problem.jl"))

λ₀ = 1.0
y = rand(Exponential(λ₀), 1000)

xs = linspace(λ₀*0.8,λ₀*1.2,100)
plt = plot(xs, analytical_posterior(ToyExponentialProblem(y, 0, 10, 1)).(xs),
           label = "true posterior")
for m in 2:4
    M = 10^m
    p = ToyExponentialProblem(y, 0, 10, M)
    chain, a = mcmc(RWMH(fill(0.1,1,1)), p, simulate_state(p, [log(λ₀)]), 10000)
    λs = get_λ.(get_θ.(chain))
    dens = kde(λs)
    plot!(plt, xs, pdf.(dens, xs), label = "M = $(M)")
end
display(plt)

vline!([mean(p.y)], label = "ML")
vline!([mean(λs)], label = "posterior mean")
(var(λs)/var(rand(analytical_distribution(p), 100000); corrected = false))^2



######################################################################
# comparison of MCMC and indirect posterior
######################################################################

M = 100
p = ToyExponentialProblem(y, 0, 10, M)
state = simulate_state(p, [log(λ₀)])
chain, a = mcmc(RWMH(fill(0.1,1,1)), p, state, 10000)
λs = get_λ.(get_θ.(chain))
dens = kde(λs)
f(x) = HamiltonianABC.logposterior(p, state, log(x))
c = f(λ₀)-log(pdf(dens, λ₀))
plt = plot(xs, x->exp(f(x)-c), label = "indirect posterior")
plot(xs, pdf.(dens, xs), label = "MCMC, M = $(M)")

f(l)-c = log(pdf(dens, l))

exp(f(λ₀)-c)

data_loglikelihood(p, mean_and_var(generate_x(p, [0.0], rand(100))))

y = rand(Exponential(1),100)
p = ToyExponentialProblem(y, 0, 10, 100)
xs = linspace(0.5,2,100)
plt = plot(xs, log.(analytical_posterior(p).(xs)), label = "true log posterior")
for m in 2:4
    M = 10^m
    p = ToyExponentialProblem(y, 0, 10, M)
    state = simulate_state(p, [log(1)])
    plot!(plt, xs, x->HamiltonianABC.logposterior(p, state, log(x)),
          label = "M=$(M)")
end
display(plt)
