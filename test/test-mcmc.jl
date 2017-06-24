######################################################################
# test counter and mcmc with a standard normal
######################################################################

"Normal(0,1) density. Keeps track of simulations. For unit tests."
mutable struct StandardNormalModel
    counter::Int
end

logdensity(::StandardNormalModel, θ) = -0.5*θ[1]^2

simulate!(p::StandardNormalModel) = p.counter += 1

@testset "standard normal" begin
    p = StandardNormalModel(0)
    chain, a = mcmc(RWMH(diagm([0.5])), p, [0.0], 1000)
    @test p.counter == 1000
    xs = first.(chain)
    @test mean(xs) ≈ 0 atol = 0.1
    @test std(xs) ≈ 1 atol = 0.2
end

######################################################################
# test mcmc with a transformed Γ
######################################################################

@testset "log transformed Γ" begin
    dist = Gamma(5.0, 1.0)
    # sample log(x), where x ∼ distribution, apply the Jacobian correction
    p = DensityWrapper(θ -> logpdf(dist, exp(θ[1])) + θ[1])

    chain, a = mcmc(RWMH(diagm([2.0])), p, [log(5.0)], 10000)
    xs = exp.(first.(chain))

    @test mean(xs) ≈ mean(dist) atol = 0.2
    for q in 0.1:0.1:0.9
        @test quantile(xs, q) ≈ quantile(dist, q) atol = 0.2
    end
end
