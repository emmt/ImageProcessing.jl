using ImageProcessing
using TypeUtils
using Test

using ImageProcessing: front, tail

const Returns = ImageProcessing.Returns

@testset "ImageProcessing.jl" begin
    @testset "Utilities" begin
        @test_throws ArgumentError front(())
        @test_throws ArgumentError tail(())
        @test front((1,)) === ()
        @test tail((1,)) === ()
        @test front((1,2)) === (1,)
        @test tail((1,2)) === (2,)
        @test front((1,pi,:b)) === (1,pi)
        @test tail((1,pi,:b)) === (pi,:b)
    end
    @testset "Points" begin
        @testset "Miscellaneaous" begin
            @test_throws Exception Point(sin, "foo", 3)
        end
        @testset "Point$coords" for coords in ((3//4,), (1,2,3), (-1f0,2f0), (-2e0, 0, 4f0, 0x5))
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
            @test A[:] === Tuple(A)
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
            @test 2*A === Point(map(x -> 2*x, A.coords))
            @test A*2 === Point(map(x -> x*2, A.coords))
            @test 2\A === Point(map(x -> 2\x, A.coords))
            @test A/2 === Point(map(x -> x/2, A.coords))
            # Addition and subtraction of points.
            B = 2*A
            @test A + A === 2A
            @test A + B === Point(A.coords .+ B.coords)
            @test B + A === Point(B.coords .+ A.coords)
            @test A - B === Point(A.coords .- B.coords)
            @test B - A === Point(B.coords .- A.coords)
            # Comparisons.
            @test A == A
            @test A != -A
            @test A != B
            @test (A < B) === (reverse(A.coords) < reverse(B.coords))
            @test (A <= B) === (reverse(A.coords) <= reverse(B.coords))
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
            # `oneunit` yields Point(1,1,...).
            @test Tuple(oneunit(A)) === ntuple(Returns(oneunit(eltype(A))), length(A))
        end
        @testset "Points and Cartesian indices" begin
            I = CartesianIndex(-1,2,3)
            A = Point{3,Int16}(-1,2,3)
            @test Point(I) === Point(Tuple(I))
            @test Point{3}(I) === Point(Tuple(I))
            @test Point{3,Float32}(I) === Point(map(Float32, Tuple(I)))
            @test CartesianIndex(A) === CartesianIndex(A.coords)
        end
        @testset "Comparions for points" begin
            # `isless` shall compare in reverse lexicographic order.
            @test (Point((1,2)) < Point((1,2))) == false
            @test (Point((1,2)) < Point((2,1))) == false
            @test (Point((2,1)) < Point((1,2))) == true
            @test (Point((2,1)) < Point((2,1))) == false
        end
    end

    @testset "Boxes" begin
        # 0-dimensional box.
        @test_throws ErrorException BoundingBox()
        @test_throws ErrorException BoundingBox{0}()
        B = @inferred BoundingBox{0,Int16}()
        @test B.start == B.stop == Point{0,Int16}()
        @test B.intervals == ()
        @test !isempty(B)
        @test ndims(B) == 0
        @test eltype(B) == Point{0,Int16}
        # 1-dimensional box.
        for rngs in ((0x2:0x3,), (Base.OneTo(7), Int16(-1):Int16(4),))
            N = length(rngs)
            T = promote_type(map(eltype, rngs)...)
            start = Point(map(first, rngs))
            stop = Point(map(last, rngs))
            B = @inferred BoundingBox(rngs)
            @test eltype(B) === Point{N,T}
            @test ndims(B) === N
            @test ndims(B) === ndims(typeof(B))
            @test ndims(B) === ndims(B.start)
            @test ndims(B) === ndims(B.stop)
            @test ndims(B) === length(B.intervals)
            @test isempty(B) == !(start <= stop)
            @test B.start == start
            @test B.stop == stop
            @test B.intervals === map(Interval, start.coords, stop.coords)
            @test first(B) === B.start
            @test last(B) === B.stop
            @test endpoints(B) === (B.start, B.stop)
            @test B.intervals == map(Interval, rngs)
            @test B === @inferred BoundingBox(rngs...)
            @test B === @inferred BoundingBox{N}(rngs)
            @test B === @inferred BoundingBox{N}(rngs...)
            @test B === @inferred BoundingBox{N,T}(rngs)
            @test B === @inferred BoundingBox{N,T}(rngs...)
            @test B === @inferred BoundingBox(start, stop)
            @test B === @inferred BoundingBox{N}(start, stop)
            @test B === @inferred BoundingBox{N,T}(start, stop)
            @test B === @inferred (start : stop)
            @test B === @inferred BoundingBox(B)
            @test B === @inferred BoundingBox{N}(B)
            @test B === @inferred BoundingBox{N,T}(B)
            C = @inferred BoundingBox{N,Int8}(start, stop)
            @test B !== C
            @test B  == C
            @test B === @inferred convert(BoundingBox, B)
            @test B === @inferred convert(BoundingBox{N}, B)
            @test B === @inferred convert(BoundingBox{N,T}, B)
            @test C === @inferred convert(BoundingBox{N,Int8}, B)
            C = @inferred BoundingBox{N,Int}(B)
            @test B == C
            @test C isa BoundingBox{N,Int}
            @test C === @inferred BoundingBox(       CartesianIndex(start), CartesianIndex(stop))
            @test C === @inferred BoundingBox{N}(    CartesianIndex(start), CartesianIndex(stop))
            @test C === @inferred BoundingBox{N,Int}(CartesianIndex(start), CartesianIndex(stop))
            R = CartesianIndices(rngs)
            @test endpoints(B) === (first(B), last(B))
            @test endpoints(R) === (first(R), last(R))
            @test endpoints(B) == map(Point,          endpoints(R))
            @test endpoints(R) == map(CartesianIndex, endpoints(B))
            @test   R ⊆ B  # the discrete set is a subset of the continuous set
            @test !(B ⊆ R) # but not the contrary
            @test R == @inferred (B ∩ CartesianIndices)
            @test R == @inferred (B ∩ CartesianIndices{N})
            @test R == @inferred (CartesianIndices ∩ B)
            @test R == @inferred (CartesianIndices{N} ∩ B)
            # Conversion of a `BoundingBox` to `CartesianIndices` must be done by ∩, not
            # by `convert` nor by the `CartesianIndices` constructor.
            @test_throws Exception convert(CartesianIndices, B)
            @test_throws Exception CartesianIndices(B)
        end
    end
end
nothing
