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

    # Fit a straight line plus sine.
    fns = (x -> 1.0, identity, sin)
    c0 = (-7.25, -2.50, 3.75)
    n = length(fns)
    X = -1.0:0.125:2.0
    eqs = @inferred NormalEquations{n,Float64}()
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
