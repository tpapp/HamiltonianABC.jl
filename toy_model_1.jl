### Toy Model 1
#Pkg.add("Optim")
#Pkg.add("StatsFuns")
#Pkg.add("Plots")
#Pkg.add("GR")
#Pkg.add("KernelDensity")
#Pkg.add("Pyplot")
using Distributions
using Optim
using StatsFuns
using Plots; pyplot()
using KernelDensity
n=1  ## number of replications, will be a parameter in the function
N=100   ## length of data, as in the article
α=30
β=1
y=rand(Poisson(30),N)  ### not sure whether here β=1 is the scale or the rate

## prior lambda
λ_prior=rand(Gamma(α,β))
λ_prior
## generating the "data" from p(.|λ_prior)
x_star = rand(Poisson(λ_prior),N)

Y=y
X=x_star
## function for the log-likelihood of a normal
function log_likeli_hood(params)
    μ =params[1]
    σ_2=exp(params[2])
    ϵ=Y-X*μ
    distrib=Normal(0,√(σ_2))
    value=logpdf(distrib,ϵ)
    loglike=sum(value)
    -loglike
end
## testing
log_likeli_hood([1,2])

## optimization
params_init=[5.4,10.2]
optimum=optimize(log_likeli_hood,params_init)
MLE=optimum.minimizer
MLE[2]=exp(MLE[2])
MLE ## step 3
MLE[1]
#####################################################################
## iteration ##
T=10000
λ=ones(T)
λ[1]=λ_prior
ϕ=ones(T,2)
ϕ[1,:]=MLE
## random walk metropolis-hastings -> cancels out in the ratio of proposals
for i in 2:T
  λ_prop = rand(Normal(λ[i-1],1))  #step 5
  X= rand(Poisson(λ_prop),N)  ## step 6
  optimum=optimize(log_likeli_hood,params_init)
  MLE=optimum.minimizer
  MLE[2]=exp(MLE[2])  ## step 7
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
#MLE
#λ
#ϕ
plot(λ)
histogram(λ)
mean(λ)

### what happens, when there are more than one replication
repl=10
y_repl=repeat(Y,outer=[repl])
Y=y_repl
x_repl= rand(Poisson(λ_prior),N*repl)
X=x_repl

params_init=[5.4,10.2]
optimum=optimize(log_likeli_hood,params_init)
MLE=optimum.minimizer
MLE[2]=exp(MLE[2])
MLE ## step 3
MLE[1]
#####################################################################
## iteration ##
T=10000
λ=ones(T)
λ[1]=λ_prior
ϕ=ones(T,2)
ϕ[1,:]=MLE
## random walk metropolis-hastings -> cancels out in the ratio of proposals
for i in 2:T
  λ_prop = rand(Normal(λ[i-1],1))  #step 5
  X= rand(Poisson(λ_prop),N*repl)  ## step 6
  optimum=optimize(log_likeli_hood,params_init)
  MLE=optimum.minimizer
  MLE[2]=exp(MLE[2])  ## step 7
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
plot(λ)
histogram(λ)
density(λ)  ## does not work
mean(λ)   ## this is almast perfect
