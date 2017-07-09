using Distributions

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
    normalize_logdensity(logdensity, x, xmin, xmax)

Return a logdensity function from an unnormalized `logdensity`.

`x` should be a point where the density had a value that is not far
form the typical region (eg the mode or the mean will be fine). This
is used to correct the value *before* calculating the integral, for
better accuracy.

`xmin` and `xmax` are integration bounds.
"""
function normalize_logdensity(logdensity, x, xmin, xmax)
    c = logdensity(x)
    C, _ = hquadrature(x -> exp(logdensity(x)-c), xmin, xmax)
    c += log(C)
    x -> logdensity(x)-c
end

pdf2cdf(pdf, xmin) =  x ->  hquadrature(pdf, xmin, x)[1]

"""
    test_cdf(density, xs[; ps, atol])

Compare the random values `xs` to the given `cdf` (cumulative density)
function. Comparison is made at `ps`, with absolute tolerance `atol`
(in probability space). Useful for testing distributions.
"""
function test_cdf(cdf, xs; ps = 0.1:0.1:0.9, atol = 0.05)
    for (p, x) in zip(ps, quantile(xs, ps))
        @test p ≈ cdf(x) atol = atol
    end
end

function test_cdf(dist::Distribution{Univariate, Continuous}, xs; args...)
    test_cdf(x ->  cdf(dist, x), xs; args...)
end
