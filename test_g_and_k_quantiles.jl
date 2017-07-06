######################################################################
## testing the g-and-k quantile function                            ##
######################################################################

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
    # true parameters
    χ = [0, 0.9, 0, 0.26]
    # generating a t-distribution with 5 degrees of freedom
    tx = rand(TDist(5), 1000)
    # cdf
    cdf_t_samples(x) = get_cdf_quant(normal_gk_quant(x,χ...), χ)
    # comparing the quantiles
    test_cdf(cdf_t_samples, tx; ps = 0.1:0.1:0.9, atol = 0.05)
end
