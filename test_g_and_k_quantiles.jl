######################################################################
## testing the g-and-k quantile function                            ##
######################################################################
'the quantiles should coincide with the Normal(0,1)'

@testset "g-and-k quantile distribution" begin
    'parameters of the g-and-k quantile function'
    χ = (0, 1, 0, 0, 1)
    'sampling from the quantil function'
    samples = map(x -> gk_quant.(x, χ...), (0.01 : 0.01 : 0.99))
    'approximating the cdf of the quantile function'
    cdf_samples = get_cdf_quant.(samples)
    'generating standard normally distribution samples'
    xxs = rand(Normal(0, 1), 1000)
    'comparing the quantiles'
    test_cdf(get_cdf_quant, xxs; ps = 0.1:0.1:0.9, atol = 0.2)

end
