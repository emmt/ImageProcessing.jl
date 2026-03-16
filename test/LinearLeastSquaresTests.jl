module LinearLeastSquaresTests

using Test
using Neutrals
using ImageProcessing
using ImageProcessing.LinearLeastSquares
using .LinearLeastSquares: lazy_convert
using Base: @propagate_inbounds

const Indexable{T} = Union{AbstractVector{T},Tuple{Vararg{T}}}

function mdot(x, y)
    @assert length(x) == length(y)
    return mapreduce(*, +, x, y)
end

@propagate_inbounds _mcall(fns::Indexable{<:Function}, i::Int, x) = fns[i](x)
@propagate_inbounds _mcall(fns::Indexable{<:Function}, i::Int, x::Tuple) = fns[i](x...)

function mcall(fns::NTuple{N,Function}, x) where {N}
    return ntuple(i -> _mcall(fns, i, x), Val(N))
end

function mcall(fns::AbstractVector{<:Function}, x)
    return [@inbounds(_mcall(fns, i, x)) for i in eachindex(fns)]
end

@testset "Linear least squares" begin
    # Utilities.
    @test -2.5f0 === @inferred lazy_convert(Float32, -5//2)
    @test  4.0f0 === @inferred lazy_convert(Float32,  4)
    @test  5.0f0 === @inferred lazy_convert(Float32,  5.0)
    @test   ZERO === @inferred lazy_convert(Float32, ZERO)
    @test    ONE === @inferred lazy_convert(Float32,  ONE)
    @test   -ONE === @inferred lazy_convert(Float32, -ONE)
    c = @inferred lazy_convert(Float32)
    @test c isa Function
    @test -2.5f0 === @inferred c(-5//2)
    @test  4.0f0 === @inferred c(4)
    @test  5.0f0 === @inferred c(5.0)
    @test   ZERO === @inferred c(ZERO)
    @test    ONE === @inferred c( ONE)
    @test   -ONE === @inferred c(-ONE)

    # Fit a straight line.
    fns = (x -> 1.0, x -> x)
    c0 = (-2.25, 0.50)
    n = length(fns)
    X = -1.0:0.125:2.0
    eqs = @inferred StaticNormalEquations{n,Float64}()
    @test eqs isa StaticNormalEquations{n,Float64}
    @test @inferred(eltype(eqs)) === Float64
    A = @inferred lhs_matrix(eqs; readonly=false)
    b = @inferred rhs_vector(eqs; readonly=false)
    @test size(A) == (n, n)
    @test size(b) == (n,)
    @test all(iszero, A)
    @test all(iszero, b)
    @test applicable(update!, eqs)
    @test_throws Exception update!(eqs)
    for x in X
        fx = mcall(fns, x)
        y = mdot(c0, fx)
        eqs = @inferred update(eqs, y, fx)
    end
    c = @inferred solve(eqs)
    @test collect(c) ≈ collect(c0)
    # idem with weights
    eqs1 = @inferred zero(eqs)
    @test eqs1 === @inferred StaticNormalEquations{n,eltype(eqs)}()
    @test eqs1 === @inferred StaticNormalEquations{n,eltype(eqs),(n*(n+1))>>1}()
    @test_throws Exception StaticNormalEquations{n,Int16}() # T must be floating-point
    @test_throws Exception StaticNormalEquations{n,Float32,1}() # invalid L
    # weights must be finite and non-negative
    @test_throws ArgumentError update(eqs1, -1.0, 1.0, (2.0, 3.0))
    @test_throws ArgumentError update(eqs1, Inf, 1.0, (2.0, 3.0))
    @test_throws ArgumentError update(eqs1, NaN, 1.0, (2.0, 3.0))
    # zero weight change nothing
    eqs2 = @inferred update(eqs1, 0.0, 1.0, (2.0, 3.0))
    @test eqs2 === eqs1
    eqs2 = eqs1
    w = 1.0
    for x in X
        fx = mcall(fns, x)
        y = mdot(c0, fx)
        w *= 1.1
        eqs1 = @inferred update(eqs1, w, y, fx)
        eqs2 = @inferred update(eqs2, y, fx...; weight=w)
    end
    @test collect(eqs1.A) ≈ collect(eqs2.A)
    @test collect(eqs1.b) ≈ collect(eqs2.b)
    c1 = @inferred solve(eqs1)
    c2 = @inferred solve(eqs2)
    @test collect(c1) ≈ collect(c0)
    @test collect(c2) ≈ collect(c0)
    # Same fit but with list of functions.
    eqs = @inferred zero(StaticNormalEquations{length(fns),Float64})
    @test_throws DimensionMismatch update(eqs, 1.0, (fns..., identity), 0.0) # too many functions
    @test_throws DimensionMismatch update(eqs, 1.0, fns[2:end], 0.0) # too few functions
    for x in X
        y = mdot(c0, mcall(fns, x))
        eqs = @inferred update(eqs, y, fns, x)
    end
    c = @inferred solve(eqs)
    @test collect(c) ≈ collect(c0)

    # Fit a straight line plus sine.
    fns = (x -> true, identity, sin) # `true` is to force a conversion
    c0 = (-7.25, -2.50, 3.75)
    n = length(fns)
    X = -1.0:0.125:2.0
    eqs = @inferred StaticNormalEquations{n,Float32}()
    @test eqs isa StaticNormalEquations{n,Float32}
    @test @inferred(eltype(eqs)) === Float32
    A = @inferred lhs_matrix(eqs; readonly=false)
    b = @inferred rhs_vector(eqs; readonly=false)
    @test size(A) == (n, n)
    @test size(b) == (n,)
    @test all(iszero, A)
    @test all(iszero, b)
    for x in X
        fx = mcall(fns, x)
        y = mdot(c0, fx)
        eqs = @inferred update(eqs, y, fx)
    end
    c = @inferred solve(eqs)
    @test collect(c) ≈ collect(c0)
    # Same fit but with a tuple of functions.
    eqs = @inferred zero(StaticNormalEquations{length(fns),Float32})
    for x in X
        y = mdot(c0, mcall(fns, x))
        eqs = @inferred update(eqs, y, fns, x)
    end
    c = @inferred solve(eqs)
    @test collect(c) ≈ collect(c0)
end

end # module
