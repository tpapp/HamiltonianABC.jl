using Distributions
using Optim
using StatsFuns
using Plots; pyplot()
using Parameters


"""
the structure Params contains the needed initial parameters for the first toymodel
α is the first parameter of the Gamma distribution
β is the second parameter of the Gamma distribution
niter is an integer, indicating how many replications we want
N is the length of the observed data
"""

@with_kw immutable Params
  α ::Float64
  β::Float64
  niter::Integer
  N::Integer
end

"""
the structure ToyModelPoisson contains the initial λ_prior for the Poisson model
λ_prior ∼Gamma(Params.α,Params.β)
"""
@with_kw immutable ToyModelPoisson
  λ_prior::Float64
end

"""
the simulate_poi function simulates points using the λ_prior
the length of the simulation depends on the length of the observed data (N)
and the number of replications we want (niter)
"""

function simulate_poi(model::ToyModelPoisson,parameter::Params)
  @unpack λ_prior = model
  @unpack α,β,niter,N = parameter
  X_repl= rand(Poisson(λ_prior),N*niter)
  X_repl
end

"""
the function MLE_poi gives back the MLE of the mean and variance of the
simulated x vector
"""
function MLE_poi(x,params_initial)
  function log_likeli_hood(params_initial)
      μ =params_initial[1]
      σ_2=exp(params_initial[2])
      distrib=Normal(μ,√(σ_2))
      value=logpdf(distrib,x)
      loglike=sum(value)
      -loglike
  end
  optimum=optimize(log_likeli_hood,params_initial).minimizer
  [optimum[1],exp(optimum[2])]
end

"""
the function RWMH runs a random-walk Metropolis-Hastings sampler in order to draw λs
the initial value is the drew λ_prior and the proposal distribution is centered around
the previously sampled λ with variance matrix Σ
Y is the observed DataTypex is the simulated data
M is the number of iterations we want to run
"""

function RWMH(model::ToyModelPoisson,parameter::Params,Y, Σ, M,params_init=[0.5,3.2])
  @unpack λ_prior = model
  @unpack α,β,niter,N = parameter
  λ=ones(M)
  ϕ=ones(M,2)
  λ[1]=λ_prior
  ϕ[1,:]=MLE_poi(simulate_poi(model,parameter),params_init)
  for i in 2:M
    λ_prop = rand(Normal(λ[i-1],Σ))
    model=ToyModelPoisson(λ_prop)
    X=simulate_poi(model,parameter)
    MLE=MLE_poi(X,params_init)
    ## step 8a:
    ratio=(mean(pdf(Normal(MLE[1],MLE[2]),Y))*pdf(Gamma(α,β),λ_prop))/(mean(pdf(Normal(ϕ[i-1,1],ϕ[i-1,2]),Y))*pdf(Gamma(α,β),λ[i-1]))
    ## step 8b:
    r=min(1,ratio)
    if rand()<r
      λ[i]=λ_prop
      ϕ[i,:]=MLE
    else
      λ[i]=λ[i-1]
      ϕ[i,:]=  ϕ[i-1,:]
    end
  end
  λ
end


## testing the functions
y=rand(Poisson(30),100)
parr=Params(α=30,β=1,niter=10,N=100)
prior_λ= ToyModelPoisson(λ_prior=rand(Gamma(parr.α,parr.β)))
results=RWMH(prior_λ,parr,y,1,100000)
plot(results)
histogram(results)
