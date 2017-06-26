using Distributions
using Plots
Plots.gr()
using Parameters
using Roots
"""
        unit_row_matrix(n,m)

facilitator function, generates an nxm matrix, whose rows sum up to 1
"""

function unit_row_matrix(n,m)
    MM=rand(n,m)
    for i in 1:n
        b=sum(MM[i,:])
        for j in 1:m
        MM[i,j]=MM[i,j]/b
        end
    end
    MM
end
"""
    normal_mixture(μs,σs,weights)

the function takes in the means (μs), standard deviations (σs) and the weigths
to generate a Gaussian Mixture Model for later usage
"""

function normal_mixture(μs,σs,weights)
    d=Array{Float64}(0,1)
    for i in 1:length(μs)
        d=cat(1,d,[(μs[i], σs[i])])
    end
    "mixture of normal model"
    dd=MixtureModel(Normal,[d...] ,[weights...])
    dd
end

"""
    normal_mixture_EM_parameters!(μs,σs,weights,hs,x)

the function updates the parameters μs,σs and weigths of the Gaussian Mixture Model
using the Expectation Maximization algorithm
"""
function normal_mixture_EM_parameters!(μs,σs,weights,hs,x)
    T=length(x)
    for ll in 1:length(μs)
        weights[ll]=(1/T)*sum(hs[:,ll])
        μs[ll]=sum(hs[:,ll].*x)/sum(hs[:,ll])
        σs[ll]=√(sum(hs[:,ll].*(x.-μs[ll]).^2)/sum(hs[:,ll]))
    end
end
"""
    normal_mixture_EM_posterior!(μs,σs,weights,hs,x)

the function updates the hs matrix given the Parameters
μs,σs and weights of the Gaussian Mixture Model
using the Expectation Maximization algorithm
the hs matrix contains the posterior probabilities
what we get back is the logpdf of the Mixture Model
"""


function normal_mixture_EM_posterior!(μs,σs,weights,hs,x)
    dd=normal_mixture(μs,σs,weights)
    for k in 1:length(μs)
        for t in 1:length(x)
            hs[t,k]=weights[k]*pdf(Normal(μs[k],σs[k]),x[t])/pdf(dd,x[t])
        end
    end
    sum(logpdf(dd,x))#,hs
end
"""
    normal_mixture_EMM(x,max_step=1000,tol=eps())

the function takes in the observed data points 'x', the maximum number of iteration steps
and the tolerance
the function uses the Expectation Maximization algorithm  to update the parameters of
the Gaussian Mixture Model, namely 'μs,σs and the weights'
the function also gives back the loglikelihood of the Mixture Model with
the updated parameters
"""

function normal_mixture_EMM(x,max_step=1000,tol=eps())
    ℓ=0
    step=1
    n=length(x)
    # allocate - initialize the parameters
    hs=unit_row_matrix(n,3)
    μs=[4.1,2.1,13.13]
    σs=[1.2,0.7,3.1]
    weights=[0.3,0.6,0.1]
    while step ≤ max_step
        normal_mixture_EM_parameters!(μs,σs,weights,hs,x)
        ℓ′=normal_mixture_EM_posterior!(μs,σs,weights,hs,x)
        Δ, ℓ = abs(ℓ′-ℓ),ℓ′
        if Δ ≤ tol
            break
        end
        step +=1
    end
    ℓ,μs,σs,weights,step
end

## testing
## sometimes it takes around 900 iterations or 1001 and it is not
# that precise in those cases
dd=normal_mixture((1.2,3.7,12.3),(0.4,1.1,4.3),(0.3,0.35,0.35))
xx=rand(dd,10000)
estim_=normal_mixture_EMM(xx)
