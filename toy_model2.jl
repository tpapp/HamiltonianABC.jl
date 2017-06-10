using Distributions
using Optim
using StatsFuns
using Plots; pyplot()
using KernelDensity

## MLE of mixture of normals
function mixture_of_normals(theta)
    mu1 = theta[1]
    mu2 = theta[2]
    mu3 = theta[3]
    v1 = theta[4]
    v2 = theta[5]
    v3 = theta[6]
    p1 = theta[7]
    p2 = theta[8]
    ϵ1=Y-X*mu1
    ϵ2=Y-X*mu2
    ϵ3=Y-X*mu3
    distrib1 = Normal(mu1,√(v1))
    distrib2 =Normal(mu2,√(v2))
    distrib3 = Normal(mu3,√(v3))
    value1 = logpdf(distrib1,ϵ1)
    value2 = logpdf(distrib2,ϵ2)
    value3 = logpdf(distrib3,ϵ3)
    log_mix_like = sum(p1*value1 + p2*value2 +(1-p1-p2)*value3)
    -log_mix_like
end
## testing
mixture_of_normals((1.2,2.2,3.2,4.2,5.2,6.2,0.3,0.4))


## quantile function
function quant(zp,a,b,c,g,k)
  a+b*(1+c*(1-exp(-g*zp))/(1+exp(-g*zp)))*((1+zp^2)^k)*zp
end

## testing the quantile function of the standard normal distribution
quantile(Normal(),0.25)

## building up the 2nd toy model
a_true=3
b_true=1
c_true=0.8
g_true=2
k_true=5
n_quant=10001


### quantile testing, what i should get
w_quant=ones(n_quant-1)
plot_quant=ones(n_quant-1)
for i in 1:(n_quant-1)
    w_quant[i]=(1/n_quant)*i
    plot_quant[i]= quantile(Normal(),w_quant[i])
end

plot(plot_quant)  #looks like the probit we wanted

## generating the observed data
obs_data=ones(n_quant-1)
for i in 1:(n_quant-1)
  obs_data[i]=quant(plot_quant[i],a_true,b_true,c_true,g_true,k_true)
end
#obs_data
## maybe?
plot(obs_data[1500:8500])
## honestly, it looks weird

## priors
## k and b are positive
a_prior=rand(Uniform(-5,5))
b_prior=rand(Uniform(0,5))
g_prior=rand(Uniform(-5,5))
k_prior=rand(Uniform(0,10))

## iteration ##
Y=obs_data
X=ones(n_quant-1)
for i in 1:(n_quant-1)
X[i]=quant(plot_quant[i],a_prior,b_prior,c_true,g_prior,k_prior)
end
plot(X)
params_initial=[1.2,0.6,3.4,2.1,0.6,1.4,0.4,0.2]
lower=[-Inf,-Inf,-Inf,0,0,0,0,0]
upper=[Inf,Inf,Inf,Inf,Inf,Inf,1,1]
optimum_mix=optimize(DifferentiableFunction(mixture_of_normals),params_initial,lower,upper, Fminbox(), optimizer = GradientDescent)
## DOMAIN ERROR - cannot run it for some reason
MLE=optimum.minimizer
MLE[4]=exp(MLE[4])
MLE[5]=exp(MLE[5])
MLE[6]=exp(MLE[6])

T=100000
Θ=ones(T,4)
Θ[1,:]=[a_prior,b_prior,g_prior,k_prior]
ϕ=ones(T,8)
ϕ[1,:]=MLE
## random walk metropolis-hastings -> cancels out in the ratio of proposals
for i in 2:T
  #step 5  - proposed new parameters
  a_prop,b_prop,g_prop,k_prop=rand(MvNormal(Θ[i-1,:],[1,1,1,1]))

  for i in 1:(n_quant-1)
  X[i]=quant(plot_quant[i],a_prop,b_prop,c_true,g_prop,k_prop)
  end
  optimum=optimize(mixture_of_normals,params_initial,lower,upper)
  MLE=optimum.minimizer
  MLE[4]=exp(MLE[4])
  MLE[5]=exp(MLE[5])
  MLE[6]=exp(MLE[6])    ## step 7.
  ## step 8a:
  ratio=(mean(MLE[7]*pdf(Normal(MLE[1],MLE[4]),Y)+MLE[8]*pdf(Normal(MLE[2],MLE[5]),Y)+(1-MLE[7]-MLE[8])*pdf(Normal(MLE[3],MLE[6]),Y))*pdf(Uniform(-5,5),a_prop)*pdf(Uniform(0,5),b_prop)*pdf(Uniform(-5,5),g_prop)*pdf(Uniform(0,10),k_prop))/((mean(ϕ[i-1,7]*pdf(Normal(ϕ[i-1,1],ϕ[i-1,4]),Y)+ϕ[i-1,8]*pdf(Normal(ϕ[i-1,2],ϕ[i-1,5]),Y)+(1-ϕ[i-1,7]-ϕ[i-1,8])*pdf(Normal(ϕ[i-1,3],ϕ[i-1,6]),Y))*pdf(Uniform(-5,5),Θ[i-1,1])*pdf(Uniform(0,5),Θ[i-1,2])*pdf(Uniform(-5,5),Θ[i-1,3])*pdf(Uniform(0,10),Θ[i-1,4]))
  ## step 8b:
  r=min(1,ratio)
  if rand()<r
    Θ[i,:]=[a_prop,b_prop,g_prop,k_prop]
    ϕ[i,:]=MLE
  else
    Θ[i]=Θ[i-1]
    ϕ[i,:]=  ϕ[i-1,:]
  end
end

### have no idea why it does not work
