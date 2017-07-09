using Distributions
import Distributions: quantile
using Parameters
using ArgCheck
import Base: rand


"g-and-k distribution."
struct GandK{T}
    "location"
    a::T
    "scale"
    b::T
    "controls skewness"
    g::T
    "controls kurtosis"
    k::T
    "a correction constant, usually 0.8"
    c::T
    function GandK{T}(a::T, b::T, g::T, k::T, c::T) where T
        @argcheck b > 0 "Scale has to be positive."
        @argcheck k ≥ 0 "Kurtosis has to be nonnegative."
        new(a, b, g, k, c)
    end
end

function GandK(a, b, g, k)
    args = promote(a, b, g, k, 0.8)
    GandK{typeof(args[1])}(args...)
end

"""
    transform_standard_normal(gk, z)

Transform a standard normal variate `z` to one with a g-and-k
distribution `gk`.
"""
function transform_standard_normal(gk::GandK, z)
    @unpack a, b, g, k, c = gk
    e = exp(-g * z)
    a + b * (1 + c*(1-e)/(1+e)) * ((1+z^2)^k) * z
end

quantile(gk::GandK, q) = transform_standard_normal(gk, quantile(Normal(0, 1), q))

@testset "g-and-k sanity checks" begin
    @test_throws ArgumentError GandK(0, 0, 0, 0)
    @test_throws ArgumentError GandK(0, -1, 0, 0)
    @test_throws ArgumentError GandK(0, 1, 0, -1)
end

@testset "g-and-k quantile distribution" begin
    gk = GandK(0, 1, 0, 0)
    xs = transform_standard_normal.(gk, rand(Normal(0, 1), 1000))
    test_cdf(x->cdf(Normal(0, 1), x), xs)
end

rand(gk::GandK, dims...) = transform_standard_normal.(gk, randn(dims...))

@testset "g-and-k rand" begin
    gk = GandK(0, 1, 0, 0)
    test_cdf(x->cdf(Normal(0, 1), x), rand(gk, 1000))
end
