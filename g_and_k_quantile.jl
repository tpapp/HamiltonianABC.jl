using Distributions
using Calculus
using Roots
using Cubature
"""
        gk_quant(x,a,b,g,k,c)

the gk_quant function calculates the g-and-k quantile function, which
can also be interpreted as a transformation of a standard random variate
x represents the quantile of interest, a,b,g,k and c are parameters
"""
function gk_quant(x,a,b,g,k,c)
    z=quantile(Normal(0,1),x)
 a+b*(1+c*(1-exp(-g*z))/(1+exp(-g*z)))*((1+z^2)^k)*z
end

"""
        get_cdf_quant(x)

get_cdf_quant inverts the g-and-k quantile function at ´x´ in order to get
to the cdf of the quantile distribution
"""
function get_cdf_quant(x)
    fzero(y->gk_quant(y,Θs...)-x,1e-10,1-1e-10)
end
"""
        get_pdf_quant(x)

get_pdf_quant returns the logpdf of the g-and-k quantile function at x
"""

function get_pdf_quant(x)
    log(derivative.(get_cdf_quant,x))
end
