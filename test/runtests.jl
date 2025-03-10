using ImageProcessing
using TypeUtils
using Test

using ImageProcessing: front, tail, mapreduce

const Returns = ImageProcessing.Returns

build(::Type{CartesianIndices}, start::CartesianIndex{N}, stop::CartesianIndex{N}) where {N} =
    CartesianIndices(map(UnitRange{Int}, Tuple(start), Tuple(stop)))

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
        # NOTE For exact comparison (with `===`), we use coordinates that have exact
        #      representation in floating-point (integers or rationals with a denominator
        #      equal to a power of 2).
        @testset "Point$coords" for coords in ((), (3//4,), (1,2,3), (-1f0,2f0), (-2e0, 0, 4f0, 0x5))
            N = length(coords)
            T = N > 0 ? promote_type(map(typeof, coords)...) : Float32
            A = N > 0 ? @inferred(Point(coords)) : @inferred(Point{N,T}(coords))
            @test A isa Point{N,T}
            # Iterator API.
            @test Base.IteratorSize(A) === Base.HasLength()
            @test length(A) == N
            @test length(typeof(A)) == N
            @test Base.IteratorEltype(A) === Base.HasEltype()
            @test eltype(A) == T
            @test eltype(typeof(A)) == T
            @test collect(A) == collect(A.coords)
            # Other constructors.
            if N > 0
                @test A === @inferred Point(coords...)
                @test A === @inferred Point(Tuple(A))
                @test A === @inferred Point(Tuple(A)...)
                @test A === @inferred Point{N}(Tuple(coords))
                @test A === @inferred Point{N}(Tuple(coords)...)
                @test_throws Exception Point{N,true}(coords...)
            end
            @test A === @inferred Point{N,T}(Tuple(coords))
            @test A === @inferred Point{N,T}(Tuple(coords)...)
            @test_throws Exception Point{length(coords)+1,T}(coords...)
            @test_throws Exception Point{Int8(length(coords)),T}(coords...)
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
            N > 0 || continue # FIXME make other tests run for 0-length points?
            # Conversions.
            S = T === Float64 ? Float32 : Float64
            B = @inferred Point(map(as(S), A.coords))
            @test A === @inferred Point(A)
            @test A === @inferred Point{N}(A)
            @test A === @inferred Point{N,T}(A)
            @test B === @inferred Point{N,S}(A)
            @test A === @inferred convert(Point, A)
            @test A === @inferred convert(Point{N}, A)
            @test A === @inferred convert(Point{N,T}, A)
            @test B === @inferred convert(Point{N,S}, A)
            @test A === @inferred convert_eltype(T, A)
            @test B === @inferred convert_eltype(S, A)
            @test B === S.(A)
            @test B === @inferred map(S, A)
            @test A.coords === @inferred convert(Tuple, A)
            @test B.coords === @inferred convert(Tuple{Vararg{S}}, A)
            @test B.coords === @inferred convert(NTuple{N,S}, A)
            # Unary plus and minus.
            @test A === +A
            @test Tuple(@inferred(-A)) === @inferred map(-, A.coords)
            @test A === @inferred map(identity, A)
            @test A === @inferred map(+, A)
            @test @inferred(-A) === @inferred map(-, A)
            # Multiplication and division by a scalar. NOTE Following above remark: only
            # divide by powers of two.
            @test 2*A === Point(map(x -> 2*x, A.coords))
            @test 2*A ===       map(x -> 2*x, A)
            @test A*2 === Point(map(x -> x*2, A.coords))
            @test A*2 ===       map(x -> x*2, A)
            @test 2\A === Point(map(x -> 2\x, A.coords))
            @test 2\A ===       map(x -> 2\x, A)
            @test A/2 === Point(map(x -> x/2, A.coords))
            @test A/2 ===       map(x -> x/2, A)
            # Addition and subtraction of points.
            B = @inferred 3*A
            C = @inferred A + A
            @test C === @inferred(2A)
            @test C === @inferred(Point(map(+, A.coords, A.coords)))
            @test C === @inferred(Point(A.coords .+ A.coords))
            @test C === @inferred(map(+, A, A))
            @test C === @inferred(A .+ A)
            C = @inferred A + B
            @test C === @inferred(B + A)
            @test C === @inferred(Point(map(+, A.coords, B.coords)))
            @test C === @inferred(Point(A.coords .+ B.coords))
            @test C === @inferred(map(+, A, B))
            @test C === @inferred(A .+ B)
            @test C === @inferred(B .+ A)
            C = @inferred A - B
            @test -C ==  @inferred(B - A) # `===` cannot hold because of the signed zero
            @test  C === @inferred(Point(map(-, A.coords, B.coords)))
            @test  C === @inferred(Point(A.coords .- B.coords))
            @test  C === @inferred(map(-, A, B))
            @test  C === @inferred(A .- B)
            C = Point(ntuple(i -> isodd(i) ? -A[i] : 5A[i], length(A)))
            D = @inferred(A + C + B)
            @test D === @inferred(map(+, A, C, B))
            @test D === @inferred((A + C) + B)
            @test D === @inferred(A + (C + B))
            # Comparisons.
            @test A == A
            @test A != -A
            @test A != B
            @test (A < B) === (reverse(A.coords) < reverse(B.coords))
            @test (A <= B) === (reverse(A.coords) <= reverse(B.coords))
            # `zero` yields neutral element for addition.
            @test  A === @inferred(A + zero(A))
            @test  A === @inferred(A - zero(A))
            @test  A === @inferred(zero(A) + A)
            @test -A ==  @inferred(zero(A) - A) # `===` cannot hold because of the signed zero
            # `one` yields multiplicative identity.
            @test  A === @inferred(one(A)*A)
            @test  A === @inferred(A*one(A))
            @test -A === @inferred((-one(A))*A)
            @test -A === @inferred(A*(-one(A)))
            @test one(A)\A == A
            @test A/one(A) == A
            # `oneunit` yields Point(1,1,...).
            @test Tuple(oneunit(A)) === ntuple(Returns(oneunit(eltype(A))), length(A))
            # `map` and broadcasted operations on points.
            @test float(Point(1,2)) === float.(Point(1,2)) === Point(float(1), float(2))
            @test Point(1,2) + Point(-4,7) === Point(1,2) .+ Point(-4,7) === Point(-3,9)
            @test Point(1,2) - Point(-4,7) === Point(1,2) .- Point(-4,7) === Point(5,-5)
            @test Point(1,2)*3 === 3*Point(1,2) === 3 .* Point(1,2) === Point(1,2) .* 3 === Point(3,6)
            @test Point(6,9)/3 === 3\Point(6,9) === Point(6,9) ./ 3 === 3 .\ Point(6,9) === Point(2.0,3.0)
            @test clamp.(Point(-3,1,7), Point(0,1,2), Point(4,3,2)) === Point(0,1,2)
            @test Point(-1,5) .+ 4.0 === 4.0 .+ Point(-1,5) === Point(3.0,9.0)
            @test Point(-1,5) .- 4.0 === Point(-5.0,1.0)
            @test 4.0 .- Point(-1,5) === Point(5.0,-1.0)
        end
        @testset "Points and Cartesian indices" begin
            I = CartesianIndex(-1,2,3)
            N = length(I)
            A = @inferred Point{N,Int16}(Tuple(I))
            @test Point(I) === Point(Tuple(I))
            @test Point{N}(I) === Point(Tuple(I))
            @test Point{N,Float32}(I) === Point(map(Float32, Tuple(I)))
            @test CartesianIndex(A) === CartesianIndex(A.coords)
            @test_throws Exception CartesianIndex(Point(1.0, 2.0))
            A = @inferred Point(reverse(Tuple(I)))
            B = @inferred A + I
            @test B isa Point
            @test B === Point(map(+, Tuple(A), Tuple(I)))
            @test B === @inferred I + A
            B = @inferred A - I
            @test B isa Point
            @test B === Point(map(-, Tuple(A), Tuple(I)))
            B = @inferred I - A
            @test B isa Point
            @test B === Point(map(-, Tuple(I), Tuple(A)))
            B = @inferred float(A) - I
            @test B isa Point
            @test eltype(B) <: AbstractFloat
            @test B === @inferred float(A - I)
        end
        @testset "Array indexing by points" begin
            A = copy(reshape(1:12, 3,4))
            @test A[Point(0x1,0x4)] == A[1,4]
            @test A[Point(2,3)] == A[2,3]
            A[Point(0x1,0x4)] = -5
            @test A[1,4] == -5
            A[Point(2,3)] = -10
            @test A[2,3] == -10
        end
        @testset "Comparisons for points" begin
            # `isless` shall compare in reverse lexicographic order.
            @test (Point((1,2)) < Point((1,2))) == false
            @test (Point((1,2)) < Point((2,1))) == false
            @test (Point((2,1)) < Point((1,2))) == true
            @test (Point((2,1)) < Point((2,1))) == false
        end
    end

    @testset "End-points, intervals, etc." begin
        # Ckeck `endpoints` method. In particular, result for an empty set shall be empty.
        #
        # Get end-points of ranges.
        @test endpoints(Base.OneTo(7)) === (1, 7)
        @test endpoints(2:7) === (2, 7)
        @test endpoints(2:1:7) === (2, 7)
        @test endpoints(7:-1:2) === (2, 7)
        r = 1.1:2.3:23.9 # nopn-empty range with a positive step
        @test endpoints(r) === (first(r), last(r))
        @test isempty(r) == isempty(Interval(endpoints(r)...))
        r = 1.1:2.3:-13.9 # empty range with a positive step
        @test endpoints(r) === (first(r), last(r))
        @test isempty(r) == isempty(Interval(endpoints(r)...))
        r = 1.1:-2.3:-13.9 # non-empty range with a negative step
        @test endpoints(r) === (last(r), first(r))
        @test isempty(r) == isempty(Interval(endpoints(r)...))
        r = -1.1:-2.3:13.9 # empty range with a negative step
        @test endpoints(r) === (last(r), first(r))
        @test isempty(r) == isempty(Interval(endpoints(r)...))
        #
        # Get end-points of intervals.
        @test isempty(Interval(endpoints(2:-1:7)...))
        @test endpoints(Interval(0.1, 11.7)) === (0.1, 11.7)
        @test isempty(Interval(endpoints(Interval(1, 0))...))
        #
        # Get end-points of Cartesian indices.
        R = CartesianIndices((-2:8, 3:5, Base.OneTo(6))) # only unit steps
        @test endpoints(R) === (CartesianIndex(-2,3,1),  CartesianIndex(8,5,6))
        @test R == build(CartesianIndices, endpoints(R)...) # only works for unt steps
        if VERSION ≥ v"1.6"
            # Step-range not allowed in Julia prior to 1.6.
            R = CartesianIndices((0:1:8, 3:5, Base.OneTo(6))) # only unit steps
            @test endpoints(R) === (CartesianIndex(0,3,1),  CartesianIndex(8,5,6))
            @test R == build(CartesianIndices, endpoints(R)...) # only works for unt steps
            R = CartesianIndices((0:2:8, 3:5, Base.OneTo(6))) # some non-unit step
            @test endpoints(R) === (CartesianIndex(0,3,1),  CartesianIndex(8,5,6))
            R = CartesianIndices((10:-2:0, 4:3:16)) # with some negative step
            @test endpoints(R) === (CartesianIndex(0,4),  CartesianIndex(10, 16))
        end
        #
        # Get end-points of collections of values.
        I = @inferred Interval(endpoints(Float32[])...)
        @test I isa Interval{Float32}
        @test isempty(I)
        A = [1.2, 7.9, 0.2, 11.4, -3.5]
        @test endpoints(A) === extrema(A)
        B = Tuple(A)
        @test endpoints(A) === extrema(A)
        #
        # Get end-points of strings.
        I = @inferred Interval(endpoints("")...)
        @test I isa Interval{Char}
        @test isempty(I)
        I = @inferred Interval(endpoints("a")...)
        @test I isa Interval{Char}
        @test !isempty(I)
        @test endpoints(I) === ('a', 'a')
        I = @inferred Interval(endpoints("abcyxu")...)
        @test I isa Interval{Char}
        @test !isempty(I)
        @test endpoints(I) === ('a', 'y')
        #
        # Get end-points of bounding-boxes.
        start, stop = Point(2,5,10), Point(3,5,10)
        B = @inferred BoundingBox(start, stop)
        @test endpoints(B) === (start, stop)

        # Intervals.
        #
        # Check that empty interval remains empty after addition, subtraction, etc.
        I = @inferred Interval(2, 1) # empty interval
        @test I isa Interval{Int}
        @test isempty(I)
        J = @inferred I + Inf
        @test J isa Interval{typeof(1 + Inf)}
        @test isempty(J)
        @test !isempty(Interval(Inf, Inf))
        @test isempty(I - Inf)
        @test isempty(Inf - I)
        @test isempty(Inf*I)
        @test isempty(-I)
        @test isempty(I - I)
        @test isempty(3*I)
        @test isempty(I/0)
        @test isempty(I/2)

        I = @inferred Interval(-1, 3)
        @test I isa Interval{Int}
        @test !isempty(I)
        @test !isempty(I + Inf)
        @test endpoints(float(I)) === map(float, endpoints(I))
        J = @inferred I + pi
        @test J === @inferred pi + I
        @test endpoints(J) === map(x -> x + pi, endpoints(I))
        J = @inferred I - 7.1
        @test endpoints(J) === map(x -> x - 7.1, endpoints(I))
        J = @inferred sqrt(2) - I
        @test endpoints(J) === map(x -> sqrt(2) - x, reverse(endpoints(I)))
        @test pi*I === I*pi
        @test endpoints(pi*I) === map(x -> pi*x, endpoints(I))
        @test endpoints((-pi)*I) === map(x -> -pi*x, reverse(endpoints(I)))
        J = @inferred Interval(-2.1, 7.3)
        @test I + J === J + I
        @test endpoints(I + J) === (I.start + J.start, I.stop + J.stop)
        @test endpoints(I - J) === (I.start - J.stop, I.stop - J.start)
        @test endpoints(J - I) === (J.start - I.stop, J.stop - I.start)
        @test_broken I - I === zero(I)
        @test I + I === 2I
        @test I + zero(I) ===  I
        @test I - zero(I) ===  I
        @test zero(I) + I ===  I
        @test zero(I) - I === -I
        @test I*one(I) === I
        @test one(I)*I === I
        @test one(I)\I === float(I)
        @test I/one(I) === float(I)
    end

    @testset "Boxes" begin
        # 0-dimensional box.
        @test_throws ErrorException BoundingBox()
        @test_throws ErrorException BoundingBox{0}()
        B = @inferred BoundingBox{0,Int16}()
        @test B.start == B.stop == Point{0,Int16}()
        @test B.intervals == ()
        @test !isempty(B)
        @test eltype(B) == Point{0,Int16}
        # 1-dimensional box.
        for rngs in ((0x2:0x3,), (Base.OneTo(7), Int16(-1):Int16(4),))
            N = length(rngs)
            T = promote_type(map(eltype, rngs)...)
            start = Point(map(first, rngs))
            stop = Point(map(last, rngs))
            B = @inferred BoundingBox(rngs)
            @test eltype(B) === Point{N,T}
            @test isempty(B) == !mapreduce(<=, &, start, stop)
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
            if VERSION < v"1.3"
                # Inference is broken here for old Julia versions.
                @test R == (B ∩ CartesianIndices)
                @test R == (B ∩ CartesianIndices{N})
                @test R == (CartesianIndices ∩ B)
                @test R == (CartesianIndices{N} ∩ B)
            else
                @test R == @inferred (B ∩ CartesianIndices)
                @test R == @inferred (B ∩ CartesianIndices{N})
                @test R == @inferred (CartesianIndices ∩ B)
                @test R == @inferred (CartesianIndices{N} ∩ B)
            end
            # Conversion of a `BoundingBox` to `CartesianIndices` must be done by ∩, not
            # by `convert` nor by the `CartesianIndices` constructor.
            @test_throws Exception convert(CartesianIndices, B)
            @test_throws Exception CartesianIndices(B)
        end
    end
end
nothing
