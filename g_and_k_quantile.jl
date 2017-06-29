using Distributions
using Calculus
using Roots
using Cubature
"""
    gk_quant(x, a, b, g, k, c)

Calculate the g-and-k quantile function, which
can also be interpreted as a transformation of a standard random variate
x represents the quantile of interest, a, b, g, k and c are parameters.
a is the location parameter
b is the non-negative, scale parameter
c is a constant usually 0.8
g is a measure of skewness
k is a measure of kurtosis of the distribution  (>-0.5)
"""

function gk_quant(x, a, b, g, k, c)
    z = quantile(Normal(0, 1), x)
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

function get_pdf_quant(x)
    log(derivative.(get_cdf_quant, x))
end
