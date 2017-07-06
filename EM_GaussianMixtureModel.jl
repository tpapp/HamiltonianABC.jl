using Distributions
using Plots
Plots.gr()
using Parameters
using Roots
using HamiltonianABC

"""
I am following the notations of Wikipedia while implementing the Expectation-Maximization algortihm of Mixture of Normals
link: https://en.wikipedia.org/wiki/Mixture_model#Expectation_maximization_.28EM.29
"""


"""
    unit_row_matrix(n,m)

Facilitator function, generates an nxm matrix, whose rows sum up to 1.
"""

function unit_row_matrix(n,m)
    MM = rand(n, m)
    for i in 1:n
        b = sum(MM[i, :])
        for j in 1:m
            MM[i, j] = MM[i, j] / b
        end
    end
    MM
end
"""
    normal_mixture(μs, σs, weights)

Generate a mixture of normals with means 'μs', st. deviations 'σs'
with probabilities 'weights'
"""

function normal_mixture(μs, σs, weights)
    # mixture of normals model
    p = sortperm([μs...])
    μs, σs, weights = μs[p], σs[p], weights[p]
    dd = MixtureModel(collect(map((m, v) -> Normal(m, v), μs, σs)), [weights...])
end


"""
    normal_mixture_EM_parameters!(μs, σs, weights, hs, x)

Update the parameters 'μs','σs' and 'weigths' of the Gaussian Mixture Model
using the Expectation Maximization algorithm
"""

function normal_mixture_EM_parameters!(μs, σs, weights, hs, x)
    # updating the parameters based on the probability matrix hs
    for ll in 1:length(μs)
        sum_col = sum(hs[:, ll])
        weights[ll] = mean(hs, 1)[ll]
        μs[ll] = sum(hs[:, ll].* x) / sum_col
        σs[ll] = √(sum(hs[:, ll].* (x.-μs[ll]).^ 2) / sum_col)
    end
    μs, σs, weights
end



"""
    normal_mixture_EM_posterior!(μs,σs,weights,hs,x)

Update the 'hs' matrix given the parameters 'μs','σs' and 'weigths' of the Gaussian Mixture Model.

Using the Expectation Maximization algorithm the hs matrix contains the posterior probabilities.
Return the logpdf of the Mixture Model.
"""

function normal_mixture_EM_posterior!(μs, σs, weights, hs, x)
    for k in 1:length(μs)
        dist = Normal(μs[k],σs[k])
        for t in 1:length(x)
            hs[t, k] = weights[k] * pdf(dist, x[t])
        end
    end
    mix_likelihood = sum(log.(sum(hs, 2)))
    broadcast!(/, hs, hs, sum(hs, 2))
    return(mix_likelihood)
end


"""
    normal_mixture_EMM(x, m, max_step=1000, tol=eps())

Take in the observed data 'x' and , return the approximated parameters.
Take in the observed data points 'x', the number of the normal mixture 'm', the maximum number of iteration steps, return the approximated parameters.

The function uses the Expectation Maximization algorithm  to update the parameters of
the Gaussian Mixture Model, namely 'μs, σs and the weights'
the function also gives back the loglikelihood of the Mixture Model with
the updated parameters.
"""

function normal_mixture_EMM(x, m, tol = eps(), max_step = 1000 )
    ℓ = NaN
    step = 1
    n = length(x)
    # initialize the parameters
    hs = unit_row_matrix(n, m)
    μs = fill(mean(x), m)
    σs = fill(std(x, corrected = false), m)
    weights = fill(1 / m, m)
    while step ≤ max_step
        normal_mixture_EM_parameters!(μs, σs, weights, hs, x)
        ℓ′ = normal_mixture_EM_posterior!(μs, σs, weights, hs, x)
        Δ, ℓ = abs(ℓ′ - ℓ), ℓ′
        if Δ ≤ tol
            break
        end
        step += 1
    end
    p = sortperm([μs...])
    μs, σs, weights = μs[p], σs[p], weights[p]
    ℓ, μs, σs, weights, hs, step
end
