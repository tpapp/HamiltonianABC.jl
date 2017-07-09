using Distributions
using Calculus
using Roots
using Cubature
using HamiltonianABC
"""
    gk_quant(x, a, b, g, k, c)

Calculate the g-and-k quantile function, which
can also be interpreted as a transformation of a standard random variate
x represents the quantile of interest, a, b, g, k and c are parameters.
a is the location parameter
b is the non-negative, scale parameter
c is a constant usually 0.8
g is a measure of skewness
k is a measure of kurtosis of the distribution 
"""

function gk_quant(x, a, b, g, k, c = 0.8)
    z = quantile(Normal(0, 1), x)
    a + b * (1 + c * (1 - exp(-g * z)) / (1 + exp(-g * z))) * ((1 + z ^ 2) ^ k) * z
end

"""
    normal_gk_quant(z, a, b, g, k, c = 0.8)

Take in the normal quantile 'z' and the parameters a, b, g and k.
"""

function normal_gk_quant(z, a, b, g, k, c = 0.8)
    a + b * (1 + c * (1 - exp(-g * z)) / (1 + exp(-g * z))) * ((1 + z ^ 2) ^ k) * z
end


"""
    get_cdf_quant(x, Θ)

Invert the g-and-k quantile function at ´x´ in order to get
to the cdf of the quantile distribution.
Θ contains the parameters for the g-and-k quantile function.
"""

function get_cdf_quant(x, Θ)
    fzero(y -> gk_quant(y, Θ...)-x, 1e-10, 1-1e-10)
end

"""
    get_pdf_quant(x)

Return the logpdf of the g-and-k quantile function at x
"""

function get_pdf_quant(quant_cdf, x)
    log(derivative.(quant_cdf, x))
end


# the quantiles should coincide with the Normal(0,1)

@testset "g-and-k quantile distribution" begin
    # parameters of the g-and-k quantile function
    χ = [0, 1, 0, 0, 1]
    # cdf function that only depends on the xs, but not on the parameters
    cdf_samples(x) = get_cdf_quant(normal_gk_quant(x,χ...), χ)
    # generating standard normally distributed samples
    xxs = rand(Normal(0, 1), 1000)
    # comparing the quantiles
    test_cdf(cdf_samples, xxs; ps = 0.1:0.1:0.9, atol = 0.05)
end


## testing for a t-distribution with 5 degrees of freedom

@testset "g-and-k quantile for t-distribution" begin
    # t-distribution with 5 degrees of freedom can be sampled
    # with the following parameters in the g-and-k quantile function
    χ = [0, 0.9, 0, 0.26]
    # generating a t-distribution with 5 degrees of freedom
    tx = rand(TDist(5), 1000)
    # cdf
    cdf_t_samples(x) = get_cdf_quant(normal_gk_quant(x,χ...), χ)
    # comparing the quantiles
    test_cdf(cdf_t_samples, tx; ps = 0.1:0.1:0.9, atol = 0.05)
end
