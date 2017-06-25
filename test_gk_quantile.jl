######################################################################
## testing the g-and-k quantile function                            ##
######################################################################

@testset "g-and-k quantile distribution" begin

    Θs=(3,1,2,1,0.8)
    samples=map(x->gk_quant.(x,Θs...),(0.01:0.01:0.99))
    μ=mean(samples)
    post_=normalized_density(get_pdf_quant,μ,0,10)
    "checking the ´density´"
    @test hquadrature(post_,0,10)[1]≈1
    "checking the quantiles"
    test_cdf(get_pdf_quant, samples, -1)

end
