###############################################################################
## testing the building functions of EM_GMModel
###############################################################################


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
