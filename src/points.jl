"""
    A = Point{N,T}(coords...)
    A = Point{N}(coords...)
    A = Point(coords...)

build a `N`-dimensional point of coordinates `coords...` stored as values of type `T`.
Unspecified parameters `N` and/or `T` are inferred from `coords...`.

Coordinates may also be given by a `N`-tuple or by a Cartesian index. Conversely, a point
`A` can be converted into a `N`-tuple or into a Cartesian index by calling the
corresponding constructor: `Tuple(A)` or `CartesianIndex(A)`.

A point `A` may be indexed to retrieve its individual coordinates: the syntax `A[i]` is
equivalent to `A.coords[i]` with `i` an integer index or an index range. For example,
`A[2:3]` yields a tuple of the 2nd to 3rd coordinates of `A` while `A[:]` yields the
`N`-tuple of coordinates of `A`. The latter may also be retrieved by `A.coords`,
`values(A)`, or `Tuple(A)`. The type and number of coordinates are given by `T =
eltype(A)` and `N = length(A)`.

A point may be used as an iterator over its coordinates. For example, `x,y,z = A` can be
written to extract the coordinates of a 3-dimensional point `A`.

A number of arithmetic operations are implemented for points and are applied elementwise
to the point coordinates: negation, addition, subtraction, and multiplication or division
by a scalar number.

The following math functions can be applied to points:

- `abs(A)`, `norm(A)`, `norm(A, 2)`, and `hypot(A)` yield the Euclidean norm of the vector
  of coordinates of point `A`, while `norm(A, p)` yields the `p`-norm of the vector of
  coordinates of point `A`;

- `min(A, B)`, `max(A, B)`, and `minmax(A, B)` work for points `A` and `B` as for
  Cartesian indices;

- `dot(A, B)` yields the scalar product of the vectors of coordinates of points `A` and
  `B`;

- `atan(A)` yields the polar angle of 2-dimensional point `A`;

- `cross(A, B)` and `A × B` yield the cross product of the vectors of coordinates of
  2-dimensional points `A` and `B`.

""" Point

# Outer constructors and error catchers (needed to avoid stack overflow).
Point(coords...) = Point(coords)
Point(coords::NTuple{N,T}) where {N,T} = Point{N,T}(coords)
@inline Point(coords::NTuple{N,Any}) where {N} =
    Point{N,to_same_concrete_type(map(typeof, coords)...)}(coords)

Point{N}(coords...) where {N} = Point{N}(coords)
Point{N}(coords::NTuple{N,Any}) where {N} = Point(coords)
Point{N}(coords::Tuple) where {N} =
    error("invalid $(length(coords))-tuple of coordinates for $N-dimensional point")

Point{N,T}(coords...) where {N,T} = Point{N,T}(coords)
Point{N,T}(coords::Tuple) where {N,T} =
    error("invalid $(length(coords))-tuple of coordinates for $N-dimensional point")

# 0-dimensional points are supported but type parameter `T` must be specified.
Point{0,T}() where {T} = Point{0,T}(())
Point(coords::Tuple{}) = error("missing type parameter `T` for 0-dimensional point")

# A point with integer coordinates can be automatically converted (i.e. by `convert`) into
# a Cartesian index and conversely.
Base.CartesianIndex(   p::Point{N}) where {N} = CartesianIndex(NTuple{N,Int}(Tuple(p)))
Base.CartesianIndex{N}(p::Point{N}) where {N} = CartesianIndex(p)

Base.convert(::Type{CartesianIndex},    p::Point{N,<:Integer}) where {N} = CartesianIndex(p)
Base.convert(::Type{CartesianIndex{N}}, p::Point{N,<:Integer}) where {N} = CartesianIndex(p)

Point(     i::CartesianIndex{N}) where {N}   = Point{N}(  Tuple(i))
Point{N}(  i::CartesianIndex{N}) where {N}   = Point{N}(  Tuple(i))
Point{N,T}(i::CartesianIndex{N}) where {N,T} = Point{N,T}(Tuple(i))

# Implement part of the API of `N`-tuples and iterators. NOTE: For `getindex`, bound
# checking cannot be avoided for tuples. For `Point`, Base methods `eltype` and `length`
# follows the same semantics as `CartesianIndex`.
Base.eltype(::Type{<:Point{N,T}}) where {N,T} = T
Base.length(::Type{<:Point{N,T}}) where {N,T} = N
@inline Base.getindex(A::Point, i) = getindex(Tuple(A), i)
Base.IteratorSize(::Type{<:Point}) = Base.HasLength()
Base.IteratorEltype(::Type{<:Point}) = Base.HasEltype()
@inline Base.iterate(iter::Point, i::Int = 1) =
    1 ≤ i ≤ length(iter) ? (iter[i], i + 1) : nothing

Base.firstindex(A::Point) = 1
Base.lastindex(A::Point) = length(A)
Base.eachindex(A::Point) = Base.OneTo(length(A))
Base.eachindex(::IndexLinear, A::Point) = Base.OneTo(length(A))
Base.keys(A::Point) = eachindex(A)
Base.values(A::Point) = Tuple(A)

# `one(x)` yields a multiplicative identity for x.
# `zero(x)` is the neutral element for the addition.
# `oneunit(x)` follows the same semantics as for Cartesian indices.
Base.one(::Type{Point{N,T}}) where {N,T} = one(T)
for f in (:zero, :oneunit)
    @eval Base.$f(::Type{Point{N,T}}) where {N,T} =
        Point{N,T}(ntuple(Returns($f(T)), Val(N)))
end

# These methods are "traits" and only depend on the type.
for f in (:zero, :one, :oneunit, :length, :eltype)
    @eval Base.$f(::Point{N,T}) where {N,T} = $f(Point{N,T})
end

# Conversion of points to tuples. The `Point` constructors already implement conversion
# from tuples.
Base.Tuple(x::Point) = x.coords
Base.NTuple(x::Point) = Tuple(x)
Base.NTuple{N}(x::Point{N}) where {N} = Tuple(x)
Base.NTuple{N,T}(x::Point{N}) where {N,T} = NTuple{N,T}(Tuple(x))
Base.convert(::Type{T}, x::Point) where {T<:Tuple} = convert(T, Tuple(x))

# Unary plus and minus.
Base.:(+)(a::Point) = a
Base.:(-)(a::Point) = Point(map(-, Tuple(a)))

# Multiplication by a scalar.
Base.:(*)(a::Number, b::Point) = Point(map(Base.Fix1(*, a), Tuple(b)))
Base.:(*)(a::Point, b::Number) = b*a

# Division by a scalar.
Base.:(/)(a::Point, b::Number) = Point(map(Base.Fix2(/, b), Tuple(a)))
Base.:(\)(a::Number, b::Point) = b/a

# Binary operations between point-like objects.
for (A, B) in ((:Point, :Point), (:Point, :CartesianIndex), (:CartesianIndex, :Point))
    # Addition.
    @eval Base.:(+)(a::$A{N}, b::$B{N}) where {N} = Point(map(+, Tuple(a), Tuple(b)))

    # Subtraction.
    @eval Base.:(-)(a::$A{N}, b::$B{N}) where {N} = Point(map(-, Tuple(a), Tuple(b)))

    # Equality and ordering.
    @eval @inline Base.isequal(a::$A, b::$B) = isequal(Tuple(a), Tuple(b))
    @eval @inline Base.isless(a::$A{N}, b::$B{N}) where {N} =
        isless(reverse(Tuple(a)), reverse(Tuple(b)))

    # Comparison operators.
    for op in (:(==), :(!=), :(<), :(<=), :(>), :(>=))
        @eval begin
            Base.$op(a::$A, b::$B) = compare_coordinates(a, $op, b)
        end
    end
end

# `min()`, `max()`, and `minmax()` for points work as for Cartesian indices.
@inline Base.min(a::Point{N}, b::Point{N}) where {N} = Point(map(min, Tuple(a), Tuple(b)))
@inline Base.max(a::Point{N}, b::Point{N}) where {N} = Point(map(max, Tuple(a), Tuple(b)))
@inline function Base.minmax(a::Point{N}, b::Point{N}) where {N}
    t = map(minmax, Tuple(a), Tuple(b))
    return Point(map(first, t)), Point(map(last, t))
end

Base.show(io::IO,                      x::Point) = show(io, MIME"text/plain"(), x)
Base.show(io::IO, m::MIME"text/plain", x::Point) = _show(io, m, x)
Base.show(io::IO, m::MIME,             x::Point) = _show(io, m, x)
function _show(io::IO, m::MIME, x::Point{N}) where {N}
    show(io, m, typeof(x))
    write(io, "(")
    @inbounds for i in 1:N
        i > 1 && write(io, ", ")
        show(io, m, x[i])
    end
    write(io, ")")
end

struct Round{T,R<:RoundingMode}
    Round(::Type{T}, r::R) where {T,R<:RoundingMode} = new{T,R}()
end
(::Round{T,R})(x) where {T,R} = round(T, x, R())

# Rounding coordinates to a tuple.
Base.round(::Type{Tuple}, x::Point{N,T}, r::RoundingMode = RoundNearest) where {N,T} =
    round(NTuple{N,T}, x, r)
Base.round(::Type{NTuple}, x::Point{N,T}, r::RoundingMode = RoundNearest) where {N,T} =
    round(NTuple{N,T}, x, r)
Base.round(::Type{NTuple{N}}, x::Point{N,T}, r::RoundingMode = RoundNearest) where {N,T} =
    round(Point{N,T}, x, r)
Base.round(::Type{NTuple{N,T}}, x::Point{N}, r::RoundingMode = RoundNearest) where {N,T} =
    map(Round(T,r), Tuple(x))
Base.round(::Type{NTuple{N,T}}, x::Point{N,T}, r::RoundingMode = RoundNearest) where {T<:Integer,N} =
    Tuple(x)

# Rounding coordinates to a point.
Base.round(x::Point{N,T}, r::RoundingMode = RoundNearest) where {N,T} = round(Point{N,T}, x, r)
Base.round(x::Point{N,<:Integer}, r::RoundingMode = RoundNearest) where {N} = x
Base.round(::Type{Point}, x::Point{N,T}, r::RoundingMode = RoundNearest) where {N,T} =
    round(Point{N,T}, x, r)
Base.round(::Type{Point{N}}, x::Point{N,T}, r::RoundingMode = RoundNearest) where {N,T} =
    round(Point{N,T}, x, r)
Base.round(::Type{Point{N,T}}, x::Point{N}, r::RoundingMode = RoundNearest) where {N,T} =
    Point(round(NTuple{N,T}, x, r))

# Rounding coordinates to a Cartesian index.
Base.round(::Type{CartesianIndex}, x::Point{N}, r::RoundingMode = RoundNearest) where {N} =
    CartesianIndex(round(NTuple{N,Int}, x, r))
Base.round(::Type{CartesianIndex{N}}, x::Point{N}, r::RoundingMode = RoundNearest) where {N} =
    round(CartesianIndex, x, r)

for (f, r) in ((:ceil, :RoundUp), (:floor, :RoundDown))
    @eval begin
        Base.$f(x::Point) = round(x, $r)

        Base.$f(::Type{Point},      x::Point{N,T}) where {N,T} = round(Point{N,T}, x, $r)
        Base.$f(::Type{Point{N}},   x::Point{N,T}) where {N,T} = round(Point{N,T}, x, $r)
        Base.$f(::Type{Point{N,T}}, x::Point{N}  ) where {N,T} = round(Point{N,T}, x, $r)

        Base.$f(::Type{Tuple},       x::Point{N,T}) where {N,T} = round(NTuple{N,T}, x, $r)
        Base.$f(::Type{NTuple},      x::Point{N,T}) where {N,T} = round(NTuple{N,T}, x, $r)
        Base.$f(::Type{NTuple{N}},   x::Point{N,T}) where {N,T} = round(NTuple{N,T}, x, $r)
        Base.$f(::Type{NTuple{N,T}}, x::Point{N}  ) where {N,T} = round(NTuple{N,T}, x, $r)

        Base.$f(::Type{CartesianIndex},    x::Point{N}) where {N} = round(CartesianIndex, x, $r)
        Base.$f(::Type{CartesianIndex{N}}, x::Point{N}) where {N} = round(CartesianIndex, x, $r)
    end
end

# To nearest Cartesian index.
nearest(::Type{CartesianIndex{N}}, x::CartesianIndex{N}) where {N} = x
nearest(::Type{CartesianIndex{N}}, x::NTuple{N}        ) where {N} = nearest(CartesianIndex, x)
nearest(::Type{CartesianIndex{N}}, x::Point{N}         ) where {N} = nearest(CartesianIndex, x)
nearest(::Type{CartesianIndex},    x::CartesianIndex   ) = x
nearest(::Type{CartesianIndex},    x::Point            ) = nearest(CartesianIndex, Tuple(x))
nearest(::Type{CartesianIndex},    x::NTuple{N,Integer}) where {N} = CartesianIndex(x)
nearest(::Type{CartesianIndex},    x::NTuple{N,Real}   ) where {N} =
    CartesianIndex(map(nearest(Int), x))

# To nearest point.
nearest(::Type{Point},      x::Point) = x
nearest(::Type{Point},      x::Union{CartesianIndex,NTuple}) = Point(x)
nearest(::Type{Point{N}},   x::Point{N}) where {N} = x
nearest(::Type{Point{N}},   x::Union{CartesianIndex{N},NTuple{N}}) where {N} = Point(x)
nearest(::Type{Point{N,T}}, x::Point{N,T}       ) where {N,T} = x
nearest(::Type{Point{N,T}}, x::Point{N}         ) where {N,T} = Point{N,T}(map(nearest(T), Tuple(x)))
nearest(::Type{Point{N,T}}, x::NTuple{N,T}      ) where {N,T} = Point{N,T}(x)
nearest(::Type{Point{N,T}}, x::NTuple{N}        ) where {N,T} = Point{N,T}(map(nearest(T), x))
nearest(::Type{Point{N,T}}, x::CartesianIndex{N}) where {N,T} = Point{N,T}(x)

# To nearest N-tuple.
nearest(::Type{Tuple},         x::Point            ) = Tuple(x)
nearest(::Type{Tuple},         x::CartesianIndex   ) = Tuple(x)
nearest(::Type{Tuple},         x::Tuple            ) = x

nearest(::Type{NTuple},        x::Point            ) = Tuple(x)
nearest(::Type{NTuple},        x::CartesianIndex   ) = Tuple(x)
nearest(::Type{NTuple},        x::NTuple           ) = x

nearest(::Type{NTuple{N}},     x::Point{N}         ) where {N} = Tuple(x)
nearest(::Type{NTuple{N}},     x::CartesianIndex{N}) where {N} = Tuple(x)
nearest(::Type{NTuple{N}},     x::NTuple{N}        ) where {N} = x

nearest(::Type{NTuple{N,T}},   x::Point{N,T}       ) where {N,T} = Tuple(x)
nearest(::Type{NTuple{N,T}},   x::Point{N}         ) where {N,T} = nearest(NTuple{N,T}, Tuple(x))
nearest(::Type{NTuple{N,Int}}, x::CartesianIndex{N}) where {N}   = Tuple(x)
nearest(::Type{NTuple{N,T}},   x::CartesianIndex{N}) where {N,T} = nearest(NTuple{N,T}, Tuple(x))
nearest(::Type{NTuple{N,T}},   x::NTuple{N,T}      ) where {N,T} = x
nearest(::Type{NTuple{N,T}},   x::NTuple{N}        ) where {N,T} = map(nearest(T), x)

# Extend `EasyRanges` package.
EasyRanges.normalize(x::Point{N,<:Integer}) where {N} = CartesianIndex(x)

# Some math functions.
# NOTE `Base.hypot(Tuple(x::Point)...)` is a bit faster than
#      `LinearAlgebra.norm2(Tuple(x::Point))`.
LinearAlgebra.norm(a::Point) = hypot(a)
LinearAlgebra.norm(a::Point, p::Real) = LinearAlgebra.norm(Tuple(a), p)
Base.hypot(a::Point) = hypot(Tuple(a)...)
Base.abs(a::Point) = hypot(a)
Base.abs2(a::Point) = mapreduce(abs2, +, Tuple(a))
Base.Math.atan(a::Point{2}) = atan(a[1], a[2])
LinearAlgebra.dot(a::Point{N}, b::Point{N}) where {N} = mapreduce(*, +, Tuple(a), Tuple(b))
LinearAlgebra.cross(a::Point{2}, b::Point{2}) = a[1]*b[2] - a[2]*b[1]
