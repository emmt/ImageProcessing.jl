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

    @testset "Boxes" begin
        # 0-dimensional box.
        B = @inferred IndexBox()
        @test B == @inferred IndexBox{0}()
        @test size(B) == ()
        @test length(B) == 1
        # 1-dimensional box.
        for rngs in ((0x2:0x3,), (Base.OneTo(7), -1:4,))
            N = length(rngs)
            start = map(first, rngs)
            stop = map(last, rngs)
            B = @inferred IndexBox(rngs)
            @test B.indices == rngs
            @test B === @inferred IndexBox(rngs...)
            @test B === @inferred IndexBox{N}(rngs)
            @test B === @inferred IndexBox{N}(rngs...)
            @test B === @inferred IndexBox(CartesianIndex(start), CartesianIndex(stop))
            @test B === @inferred IndexBox{N}(CartesianIndex(start), CartesianIndex(stop))
            @test B === @inferred IndexBox(Point(start), Point(stop))
            @test B === @inferred IndexBox{N}(Point(start), Point(stop))
            @test ndims(B) === length(rngs)
            @test ndims(typeof(B)) === length(rngs)
            @test size(B) === map(length, rngs)
            @test length(B) == prod(map(length, rngs))
            @test isempty(B) == (length(B) == 0)
            R = CartesianIndices(rngs)
            @test B.indices == R.indices
            @test R == @inferred CartesianIndices(B)
            @test B == @inferred IndexBox(B)
            @test B === @inferred convert(IndexBox, B)
            @test B === @inferred convert(IndexBox{N}, B)
            @test B === @inferred convert(IndexBox, R)
            @test B === @inferred convert(IndexBox{N}, R)
            @test R == @inferred convert(CartesianIndices, B)
            @test R == @inferred convert(CartesianIndices{N}, B)

            @test first(B) === CartesianIndex(start)
            @test last(B) === CartesianIndex(stop)
            @test first(CartesianIndex, B) === CartesianIndex(start)
            @test last(CartesianIndex, B) === CartesianIndex(stop)
            @test first(CartesianIndex{N}, B) === CartesianIndex(start)
            @test last(CartesianIndex{N}, B) === CartesianIndex(stop)
            @test_throws Exception first(CartesianIndex{N+1}, B)
            @test_throws Exception last(CartesianIndex{N+1}, B)
            @test first(Point, B) === Point{N,Int}(start)
            @test last(Point, B) === Point{N,Int}(stop)
            @test first(Point{N}, B) === Point{N,Int}(start)
            @test last(Point{N}, B) === Point{N,Int}(stop)
            @test first(Point{N,Int}, B) === Point{N,Int}(start)
            @test last(Point{N,Int}, B) === Point{N,Int}(stop)
            @test first(Point{N,Float32}, B) === Point{N,Float32}(start)
            @test last(Point{N,Float32}, B) === Point{N,Float32}(stop)
            @test_throws Exception first(Point{N+1}, B)
            @test_throws Exception last(Point{N+1}, B)
        end
    end
end
nothing
