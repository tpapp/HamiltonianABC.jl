using Distributions
using HamiltonianABC
using Base.Test
import HamiltonianABC: get_θ

@testset "log transformed Γ" begin
    distribution = Gamma(5.0, 1.0)

    # sample log(x), where x ∼ distribution, apply the Jacobian correction
    chain, a = mcmc(RWMH(fill(2.0, 1, 1)),
                    DensityWrapper(θ -> logpdf(distribution, exp(θ[1])) + θ[1]),
                    StateWrapper([5.0]), 10000)

    xs = exp.(first.(get_θ.(chain)))
    
    @test mean(xs) ≈ mean(distribution) atol = 0.2
    for q in 0.1:0.1:0.9
        @test quantile(xs, q) ≈ quantile(distribution, q) atol = 0.2
    end
end
