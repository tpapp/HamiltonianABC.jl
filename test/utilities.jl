######################################################################
# utility functions
######################################################################

"""
A wrapper for sampling from a given log density.

Useful for testing MCMC.
"""
struct DensityWrapper{T}
    "Log density function."
    ℓ::T
end

logdensity(dw::DensityWrapper, θ) = dw.ℓ(θ)

simulate!(::DensityWrapper) = nothing

"""
    normalized_density(logdensity, x, xmin, xmax)

Return a density function from an unnormalize `logdensity`.

`x` should be a point where the density had a value that is not far
form the typical region (eg the mode or the mean will be fine). This
is used to correct the value *before* calculating the integral, for
better accuracy.

`xmin` and `xmax` are integration bounds.
"""
function normalized_density(logdensity, x, xmin, xmax)
    c = logdensity(x)
    C, _ = hquadrature(x -> exp(logdensity(x)-c), xmin, xmax)
    c += log(C)
    x -> exp(logdensity(x)-c)
end

"""
    test_cdf(density, xs, xmin[; ps, atol])

Compare the random values `xs` to the given `density` function, which
has support srting at `xmin`. Comparison is made at `ps`, with
absolute tolerance `atol` (in probability space). Useful for testing
distributions.
"""
function test_cdf(density, xs, xmin; ps = 0.1:0.1:0.9, atol = 0.05)
    for (p, x) in zip(ps, quantile(xs, ps))
        p′, _ = hquadrature(density, xmin, x)
        @test p ≈ p′ atol = atol
    end
end
