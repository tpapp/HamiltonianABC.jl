using Distributions
using Plots
Plots.gr()
using Parameters
export
    help_mix_dens,
    em_gmmix,
    plotting_mixture
"""
help_mix_dens is a facilitator function which takes in the
parameters of the mixtire normals and the observed or estimated points
if logl=true, then it gives back the loglikelihood,
if false, then it gives back a vector of draws from the mixture of normals and the pdf
"""

function help_mix_dens(result,x,logl=true,N=1000)
    "result contains the mus, sigmas and the weights"
    n=length(result)/3
    d=Array{Float64}(0,1)
    for i in 1:n
        d=cat(1,d,[(result[i], result[i+n])])
    end
    "mixture of normal model"
    dd=MixtureModel(Normal,[d...] ,[result[2*n+1:end]...])
    logl==true ? sum(logpdf(dd,x)) : (rand(dd,N),pdf(dd,x))
end
"""
the em_gmmix approximates the means, st. deviations and the weigths
of the observed or approximated points (=x)
it iterates until the difference of the loglikelihood of the model with
two consecutively calculated vector of parameters is larger or equal than "tol"
num_mix is the number of normal densities we want to use (finite)
"""
function em_gmmix(x,tol=eps(),num_mix=3)
    "means"
    mus=rand(Uniform(2,43),num_mix)
    "st. deviations"
    sigmas=rand(Uniform(2,43),num_mix)
    "weigths"
    ws=fill(1/num_mix,num_mix)
    T=length(x)
    s=length(mus)
    hs=Matrix(T,s)
    loglike_=Matrix(2,1)
    loglike_[1]=0
    loglike_[2]=help_mix_dens((mus...,sigmas...,ws...),x)
    qq=1
    "iteration until the difference is small enough"
    while(abs(loglike_[qq+1]-loglike_[qq]) >= √tol)
        for k in 1:s
            for t in 1:T
                hs[t,k]=ws[k]*pdf(Normal(mus[k],sigmas[k]),x[t])/help_mix_dens((mus...,sigmas...,ws...),x[t],false,1)[2]
            end
        end
        "updating"
        for ll in 1:s
            ws[ll]=(1/T)*sum(hs[:,ll])
            mus[ll]=sum(hs[:,ll].*x)/sum(hs[:,ll])
            sigmas[ll]=√(sum(hs[:,ll].*(x.-mus[ll]).^2)/sum(hs[:,ll]))
        end
        "new loglikelihood calculated with the facilitator function"
        loglike_new=help_mix_dens((mus...,sigmas...,ws...),x)
        loglike_=cat(1,loglike_,[loglike_new])
        qq+=1
    end
    mus...,sigmas...,ws...
end
"""
the plotting_mixture function helps us check whether the approximated
mixture of normals is "close enough" or not, graphically
"""
function plotting_mixture(x_grid,tru_,estim_)
    "plotting the true density"
    plot(x_grid,tru_[2])
    "plotting the approximated density"
    plot!(x_grid,estim_[2])
end
