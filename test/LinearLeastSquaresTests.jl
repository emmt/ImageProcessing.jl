module LinearLeastSquaresTests
using Test, ImageProcessing
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
    # Fit a straight line.
    fns = (x -> 1.0, x -> x)
    c0 = (-2.25, 0.50)
    n = length(fns)
    X = -1.0:0.125:2.0
    eqs = @inferred NormalEquations{n,Float64}()
    @test eqs isa NormalEquations{n,Float64}
    @test @inferred(eltype(eqs)) === Float64
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
    # idem with weights
    eqs1 = @inferred zero(eqs)
    @test eqs1 === @inferred NormalEquations{n,eltype(eqs)}()
    @test eqs1 === @inferred NormalEquations{n,eltype(eqs),(n*(n+1))>>1}()
    @test_throws Exception NormalEquations{n,Int16}() # T must be floating-point
    @test_throws Exception NormalEquations{n,Float32,1}() # invalid L
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

    # Fit a straight line plus sine.
    fns = (x -> true, identity, sin) # `true` is to force a conversion
    c0 = (-7.25, -2.50, 3.75)
    n = length(fns)
    X = -1.0:0.125:2.0
    eqs = @inferred NormalEquations{n,Float32}()
    @test eqs isa NormalEquations{n,Float32}
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
end

end # module
