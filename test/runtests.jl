using ImageProcessing
using TypeUtils
using Test

@testset "ImageProcessing.jl" begin
    @testset "Points" begin
        @testset "Miscellaneaous" begin
            @test_throws Exception Point(sin, "foo", 3)
        end
        @testset "Point$coords" for coords in ((1,2,3), (-1f0,2f0), (-2e0, 0, 4f0, 0x5))
            N = length(coords)
            T = promote_type(map(typeof, coords)...)
            A = @inferred Point(coords)
            @test A isa Point{N,T}
            # Iterator API.
            @test Base.IteratorSize(A) === Base.HasLength()
            @test length(Point(coords)) == N
            @test Base.IteratorEltype(A) === Base.HasEltype()
            @test eltype(Point(coords)) == T
            @test collect(A) == collect(A.coords)
            # Other constructors.
            @test A === @inferred Point(coords...)
            @test A === @inferred Point(Tuple(A))
            @test A === @inferred Point(Tuple(A)...)
            @test A === @inferred Point{N}(Tuple(coords))
            @test A === @inferred Point{N}(Tuple(coords)...)
            @test A === @inferred Point{N,T}(Tuple(coords))
            @test A === @inferred Point{N,T}(Tuple(coords)...)
            # Point as a tuple.
            @test A.coords === Tuple(A)
            @test A.coords === values(A)
            @test keys(A) === Base.OneTo(N)
            @test eachindex(A) === keys(A)
            @test firstindex(A) === first(keys(A))
            @test lastindex(A) === last(keys(A))
            if N > 0
                @test A[1] === first(A.coords)
                @test A[end] === last(A.coords)
            end
            @test A[2:end-1] === A.coords[2:end-1]
            if N == 2
                x, y = A
                @test A.coords === (x, y)
            elseif N == 3
                x, y, z = A
                @test A.coords === (x, y, z)
            elseif N == 4
                w, x, y, z = A
                @test A.coords === (w, x, y, z)
            end
            # Conversions.
            S = T === Float64 ? Float32 : Float64
            B = Point(map(as(S), A.coords))
            @test Point(A) === A
            @test Point{N}(A) === A
            @test Point{N,T}(A) === A
            @test Point{N,S}(A) === B
            @test convert(Point, A) === A
            @test convert(Point{N}, A) === A
            @test convert(Point{N,T}, A) === A
            @test convert(Point{N,S}, A) === B
            @test convert_eltype(T, A) === A
            @test convert_eltype(S, A) === B
            # Unary plus and minus.
            @test A === +A
            @test map(-, A.coords) === Tuple(-A)
            # Multiplication and division by a scalar.
            @test 2*A === Point(map(x->2*x, A.coords))
            @test A*2 === Point(map(x->x*2, A.coords))
            @test 2\A === Point(map(x->2\x, A.coords))
            @test A/2 === Point(map(x->x/2, A.coords))
            # Addition and subtraction of points.
            B = 2*A
            @test A + B === Point(A.coords .+ B.coords)
            @test B + A === Point(B.coords .+ A.coords)
            @test A - B === Point(A.coords .- B.coords)
            @test B - A === Point(B.coords .- A.coords)
            # Comparisons.
            @test A == A
            @test A != -A
            @test A != B
            @test (A < B) === (A.coords < B.coords)
            # `zero` yields neutral element for addition.
            @test A + zero(A) === A
            @test A - zero(A) === A
            @test zero(A) + A === A
            @test zero(A) - A == -A
            # `one` yields multiplicative identity.
            @test one(A)*A === A
            @test A*one(A) === A
            @test (-one(A))*A === -A
            @test A*(-one(A)) === -A
            @test one(A)\A == A
            @test A/one(A) == A
        end
        @testset "Points and Cartesian indices" begin
            I = CartesianIndex(-1,2,3)
            A = Point{3,Int16}(-1,2,3)
            @test Point(I) === Point(Tuple(I))
            @test Point{3}(I) === Point(Tuple(I))
            @test Point{3,Float32}(I) === Point(map(Float32, Tuple(I)))
            @test CartesianIndex(A) === CartesianIndex(A.coords)
        end
    end
end
