###############################################################################
## testing the building functions of EM_GMModel
###############################################################################


###############################################################################
## testing the normal_mixture_EM_parameters
###############################################################################
'If the input hs matrix contains ones in one column and zeros in the pthers, then
it should give back the actual distribution'

@testset "not mixture" begin
    'building hs with 1s in the first column and zeros in the others'
    rh = zeros(1000, 2)
    lh = ones(1000)
    hh = [lh rh]
    'drawing normally distributed numbers'
    ww = rand(Normal(1.2, 0.4), 1000)
    'estimating the parameters'
    parameter = normal_mixture_EM_parameters!([3.2, 1.7, 2.3], [0.9, 0.4, 2.3], [0.3, 0.2, 0.5], hh, ww)
    'density with estimated parameters'
    dens = Normal(parameter[1][1],parameter[2][1])

    test_cdf(dens, ww; ps = 0.1:0.1:0.9, atol = 0.2)
end

################################################################################
## testing the normal_mixture_EM_posterior
################################################################################
'Testingthe function with a *one* mixture normal. It should give back hs with 1s '

@testset "testing posterior probabilities" begin
    'starting with hs=zeros'
    hs = zeros(100)
    'the function updates the hs matrix'
    normal_mixture_EM_posterior!([1.2],[0.5],[1.0], hs, rand(Normal(1.0, 0.2), 100))
    'testing whether we got back a vector of ones or not'
    hs ≈ ones(length(zt))
end
################################################################################
## testing the normal_mixture_EMM function
################################################################################
'the function should converge to the real parameters of the true of normal model,
which is not a mixture'

@testset "one-component mixture of normal EM" begin
    'generating the real mixture density, with just one component m = 1'
    dd = normal_mixture((1.2,3.7,12.3), (0.4,1.1,4.3), (0.0,1.0,0.0))
    xx = rand(dd, 100000)
    'pretend that we have a 3-component mixture'
    estim_= normal_mixture_EMM(xx, 3)
    'building a mixture of normal distribution with the estimated parameters'
    zz = normal_mixture(estim_[2], estim_[3], estim_[4])
    'checking the quantiles'
    test_cdf(zz, xx; ps = 0.1:0.1:0.9, atol = 0.2)
end

################################################################################
## testing the normal_mixture_EMM function
################################################################################
'the function should converge to the real parameters of the true mixture of normal model'

@testset "mixture of normal EM" begin
    'generating the real mixture density, with just one component m = 1'
    dd = normal_mixture((1.2,3.7,12.3), (0.4,1.1,4.3), (0.3,0.1,0.6))
    xx = rand(dd, 100000)
    'estimating the parameters'
    estim_= normal_mixture_EMM(xx, 3)
    'building a mixture of normal distribution with the estimated parameters'
    zz = normal_mixture(estim_[2], estim_[3], estim_[4])
    'checking the quantiles'
    test_cdf(zz, xx; ps = 0.1:0.1:0.9, atol = 0.2)
end
