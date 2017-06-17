using Distributions
using Optim
using StatsFuns
using Plots; pyplot()
using Parameters
using Convex

"""
Quant_Params contains the needed parameters for the priors
a pair a_l,a_u is the lower and upper parameter for the uniformly distriuted prior
"""

@with_kw immutable Quant_Params
  a_l::Float64
  a_u::Float64
  b_l::Float64
  b_u::Float64
  g_l::Float64
  g_u::Float64
  k_l::Float64
  k_u::Float64

end

"""
the structure ToyModelQuantile contains the model parameters
a is the location parameter
b is the non-negative, scale parameter
c is a constant usually 0.8
g is a measure of skewness
k is a measure of kurtosis of the distribution  (>-0.5)
niter is the number of replications we w_quant
N is the length of the observed data
"""
@with_kw immutable ToyModelQuantile
  a::Float64
  b::Float64
  c::Float64
  g::Float64
  k::Float64
  niter::Integer
  N::Integer
end

"""
the quant function calculates the quantile function, which
can also be interpreted as a transformation of a standard random variate
zp represents the quantile function of the standard normal Distribution
"""
function quant(zp,a,b,c,g,k)
  a+b*(1+c*(1-exp(-g*zp))/(1+exp(-g*zp)))*((1+zp^2)^k)*zp
end

"""
simulate_quant generates draws from the quantile function
it uses the model parameters given in ToyModelQuantile
and the parameters from Quant_Params
"""

function simulate_quant(model::ToyModelQuantile)
  @unpack a,b,c,g,k,niter, N =model
  X=Vector(N*niter)
  for i in 1:(N*niter)
    which_quant=(1/(niter*N+1))*i
    X[i]=quant(quantile(Normal(),which_quant),a,b,c,g,k)
  end
  X
end
"""
the MLE_quant function yields the 8 parameters of the auxiliary model, namely
the first three are the three means, the second three are the three variances
and the last two are the weights of the mixture of normal distributions
the input X are the points simulated with simulate_quant function
"""
function MLE_quant(X,theta)
  function mixture_of_normals(theta)
    mu1 = theta[1]
    mu2 = theta[2]
    mu3 = theta[3]
    v1 = theta[4]
    v2 = theta[5]
    v3 = theta[6]
    p1 = theta[7]
    p2 = theta[8]
    distrib1 = Normal(mu1,√(v1))
    distrib2 =Normal(mu2,√(v2))
    distrib3 = Normal(mu3,√(v3))
    value1 = logpdf(distrib1,X)
    value2 = logpdf(distrib2,X)
    value3 = logpdf(distrib3,X)
    log_mix_like = sum(p1*value1 + p2*value2 +(1-p1-p2)*value3)
    -log_mix_like
  end

  optimum=optimize(mixture_of_normals,theta)
  [optimum[1],optimum[2],optimum[3],exp(optimum[4]),exp(optimum[5]),exp(optimum[6]),optimum[7],optimum[8]]

end
"""
the function RWMH_quant runs a random-walk Metropolis-Hastings sampler to draw a,b,g and k for the quantile function
"""



function RWMH_quant(model::ToyModelQuantile,parameter::Quant_Params,Y, Σ, M,params_init=[1.2,0.6,3.4,2.1,0.6,1.4,0.4,0.2])
  @unpack a,b,c,g,k,niter,N = model
  @unpack  a_l,a_u,b_l,b_u,g_l,g_u,k_l,k_u,= parameter
  Θ=ones(M,4)
  ϕ=ones(M,8)
  Θ[1,:]=[a,b,g,k]
  ϕ[1,:]=MLE_quant(simulate_quant(model,parameter),params_init)
  for i in 2:M
    a_prop,b_prop,g_prop,k_prop=rand(MvNormal(Θ[i-1,:],Σ))
    model=ToyModelQuantile(a_prop,b_prop,0.8,g_prop,k_prop,niter,N)
    X=simulate_quant(model,parameter)
    MLE=MLE_poi(X,params_init)
    ## step 8a:
    ratio=(mean(MLE[7]*pdf(Normal(MLE[1],MLE[4]),Y)+MLE[8]*pdf(Normal(MLE[2],MLE[5]),Y)+(1-MLE[7]-MLE[8])*pdf(Normal(MLE[3],MLE[6]),Y))*pdf(Uniform(a_l,a_u),a_prop)*pdf(Uniform(b_l,b_u),b_prop)*pdf(Uniform(g_l,g_u),g_prop)*pdf(Uniform(k_l,k_u),k_prop))/(mean(ϕ[i-1,7]*pdf(Normal(ϕ[i-1,1],ϕ[i-1,4]),Y)+ϕ[i-1,8]*pdf(Normal(ϕ[i-1,2],ϕ[i-1,5]),Y)+(1-ϕ[i-1,7]-ϕ[i-1,8])*pdf(Normal(ϕ[i-1,3],ϕ[i-1,6]),Y))*pdf(Uniform(a_l,a_u),Θ[i-1,1])*pdf(Uniform(b_l,b_u),Θ[i-1,2])*pdf(Uniform(g_l,g_u),Θ[i-1,3])*pdf(Uniform(k_l,k_u),Θ[i-1,4]))
    ## step 8b:
    r=min(1,ratio)
    if rand()<r
      Θ[i,:]=[a_prop,b_prop,g_prop,k_prop]
      ϕ[i,:]=MLE
    else
      Θ[i,:]=Θ[i-1,:]
      ϕ[i,:]=  ϕ[i-1,:]
    end
  end
  Θ
end
### testing

paramm=Quant_Params(-5,5,0,5,-5,5,0,10)
paramm
model_quant=ToyModelQuantile(rand(Uniform(paramm.a_l,paramm.a_u)),rand(Uniform(paramm.b_l,paramm.b_u)),0.8,rand(Uniform(paramm.g_l,paramm.g_u)),rand(Uniform(paramm.k_l,paramm.k_u)),1.0,100.0)
xx=simulate_quant(model_quant)
plot(xx)
model_true=ToyModelQuantile(3,1,0.8,2,5,1,100)
Y=simulate_quant(model_true)
plot(Y)
plot!(xx)
## domain error
MLE_quant(xx,[1.2,0.6,3.4,2.1,0.6,1.4,0.4,0.2])
RWMH_quant(model_quant,paramm,Y,[1,1,1,1],10000)
