###############################################################################
## testing the building functions of EM_GMModel
###############################################################################


###############################################################################
## testing the normal_mixture_EM_parameters                                  ##
###############################################################################

# If the input hs matrix contains ones in one column and zeros in the pthers, then
it should give back the actual distribution'

@testset "not mixture" begin
    # building hs with 1s in the first column and zeros in the others
    rh = zeros(1000, 2)
    lh = ones(1000)
    hh = [lh rh]
    # drawing normally distributed numbers
    ww = rand(Normal(1.2, 0.4), 1000)
    # estimating the parameters
    parameter = normal_mixture_EM_parameters!([3.2, 1.7, 2.3], [0.9, 0.4, 2.3], [0.3, 0.2, 0.5], hh, ww)
    # density with estimated parameters
    mean(ww) ≈ parameter[1][1]
    std(ww, corrected = false) ≈ parameter[2][1]
end

################################################################################
## testing the normal_mixture_EM_posterior                                    ##
################################################################################

# Testing the function with a *one* mixture normal. It should give back hs with 1s.

@testset "testing posterior probabilities" begin
    # starting with hs = zeros
    hs = zeros(100)
    # the function updates the hs matrix
    normal_mixture_EM_posterior!([1.2],[0.5],[1.0], hs, rand(Normal(1.0, 0.2), 100))
    # testing whether we got back a vector of ones or not
    hs ≈ ones(length(hs))
end
################################################################################
## testing the normal_mixture_EMM function                                    ##
################################################################################
# The function should converge to the real parameters of the true of normal model,which is not a mixture

@testset "one-component mixture of normal EM" begin
    # generating the real mixture density, with just one component m = 1
    dd = rand(Normal(2.5,0.8),10000)
    # approximating the parameters
    estim_= normal_mixture_EMM(dd, 1)
    # checking the closeness of the estimated parameters to the moments
    mean(dd) ≈ estim_[2][1]
    std(dd, corrected = false) ≈ estim_[3][1]
end

################################################################################
## testing the normal_mixture_EMM function                                    ##
################################################################################

# the function should converge to the real parameters of the true mixture of normal model

@testset "mixture of normal EM" begin
    # generating the real mixture density
    μs = [1.2,3.7,12.3]
    σs = [0.4,1.1,4.3]
    weights = [0.3,0.1,0.6]
    dd = normal_mixture(μs, σs, weights)
    xx = rand(dd, 100000)
    # estimating the parameters
    estim_= normal_mixture_EMM(xx, 3)
    # checking whether the estimated parameters and weights are close to the true values or not
    norm(μs-estim_[2]) ≤ 0.1
    norm(σs-estim_[3]) ≤ 0.1
    norm(weights-estim_[4]) ≤ 0.1
end
