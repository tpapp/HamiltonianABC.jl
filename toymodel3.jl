using Distributions
using Optim
using StatsFuns
using Plots; pyplot()
using Parameters


"""
The ToyModelParasite contains the initial values of the the following variables
l_a is the initial larvae injection for the ath host
M_t is the number of mature parasites
L_t is the number of larvae
I_t is the discrete version of the host's immunity
t_curr is the current time, usually it is set to 0
t_a is the autopsy time
"""

@with_kw immutable ToyModelParasite
  l_a::Integer
  M_t::Integer
  L_t::Integer
  I_t::Integer
  t_curr::Float64
  t_a::Float64
end
"""
The Params_para contains the needed parameters for the prior distributions for several parameters of the model
"""
@with_kw immutable Params_para
  ν_l::Float64
  ν_u::Float64
  μ_I_l::Float64
  μ_I_u::Float64
  μ_L_l::Float64
  μ_L_u::Float64
  β_l::Float64
  β_u::Float64
  γ::Float64
  μ_M::Float64
end
"""
the function simulate_parasite generates the triple M(t),L(t), I(t)
it uses the the ToyModelParasite and the parameters from Params_para
it simulates the full model for the ath host
"""

function simulate_parasite(model::ToyModelParasite,parameter::Params_para)
  @unpack l_a,M_t,L_t,I_t,t_curr,t_a = model
  @unpack γ,μ_M = parameter
  prob=Matrix(0,5)
  dataa=Matrix(0,3)
  check=Matrix(0,2)
  while t_curr < t_a
    t_next = rand(Exponential(γ*L_t+(μ_L+β*I_t)*L_t+μ_M*M_t+ν*L_t+μ_I*I_t))
    t_curr +=t_next
    if t_curr > t_a
      break
    end
    pm1=γ*L_t
    pt_1= (μ_L+β*I_t)*L_t
    pm_1=μ_M*M_t
    pi1=ν*L_t
    pi_1=μ_I*I_t
    sum_weigth=pm1+pt_1+pm_1+pi1+pi_1
    prob=cat(1,prob,[pm1 pt_1 pm_1 pi1 pi_1]/sum_weigth)
    pp=rand(Multinomial(1,[pm1,pt_1,pm_1,pi1,pi_1]/sum_weigth))
    j =  find(pp .== 1)
    if j ==[1]
      M_t +=1
      L_t +=-1
    elseif j ==[2]
      L_t +=-1
    elseif j ==[3]
      M_t += -1
    elseif j ==[4]
      I_t += 1
    elseif j ==[5]
      I_t += -1
    else
      println("j should be between 1 and 5")
    end
    dataa=cat(1,dataa,[M_t L_t I_t])
    check=cat(1,check,[t_next j])
  end
  dataa
end
