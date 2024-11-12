"""
    A = Point{N,T}(coords...)
    A = Point{N}(coords...)
    A = Point(coords...)

build a `N`-dimensional point of coordinates `coords...` stored as values of type `T`.
Unspecified parameters `N` or `T` are inferred from `coords...`.

Coordinates may also be given by a `N`-tuple or by a Cartesian index. Conversely, a point
`A` can be converted into a `N`-tuple or into a Cartesian index by calling the
corresponding constructor: `Tuple(A)` or `CartesianIndex(A)`.

For a point `A`, the `N`-tuple of coordinates of `A` is retrieved by `A.coords`,
`values(A)`, or `Tuple(A)`. The type and number of coordinates are given by `T =
eltype(A)` and `N = length(A)`.

A point `A` may be indexed to retrieve its individual coordinates: the syntax `A[i]` is
equivalent to `A.coords[i]` with `i` an integer index or an index range. For example,
`A[2:3]` yields a tuple of the 2nd to 3rd coordinates of `A`.

A point may be used as an iterator over its coordinates. For example, `x,y,z = A` can be
written to extract the coordinates of a 3-dimensional point `A`.

A number of operations are implemented for points: negation, addition, subtraction, and
multiplication or division by a scalar number.

""" Point

# Outer constructors and error catchers (needed to avoid stack overflow).
Point(coords...) = Point(coords)
Point(coords::NTuple{N,T}) where {N,T} = Point{N,T}(coords)
function Point(coords::NTuple{N,Any}) where {N}
    T = promote_type(map(typeof, coords)...)
    isconcretetype(T) || error(
        "point coordinates cannot be converted to a common concrete type")
    return Point{N,T}(coords)
end

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

# Conversions.
Point(x::Point) = x
Point{N}(x::Point{N}) where {N} = x
Point{N,T}(x::Point{N,T}) where {N,T} = x
Point{N,T}(x::Point{N}) where {N,T} = Point{N,T}(Tuple(x))
TypeUtils.convert_eltype(::Type{T}, x::Point{N,T}) where {N,T} = x
TypeUtils.convert_eltype(::Type{T}, x::Point{N}) where {N,T} = Point{N,T}(x)

# Base `convert` method falls back to calling the constructor.
Base.convert(::Type{T}, x::T) where {T<:Point} = x
Base.convert(::Type{T}, x) where {T<:Point} = T(x)

# A point with integer coordinates can be converted into a Cartesian index and conversely.
Base.CartesianIndex(x::Point{N,<:Integer}) where {N} = CartesianIndex(Tuple(x))
Base.CartesianIndex{N}(x::Point{N,<:Integer}) where {N} = CartesianIndex(x)
Base.convert(::Type{CartesianIndex}, x::Point{N,<:Integer}) where {N} = CartesianIndex(x)
Base.convert(::Type{CartesianIndex{N}}, x::Point{N,<:Integer}) where {N} = CartesianIndex(x)

Point(index::CartesianIndex{N}) where {N} = Point{N,Int}(Tuple(index))
Point{N}(index::CartesianIndex{N}) where {N} = Point{N}(Tuple(index))
Point{N,T}(index::CartesianIndex{N}) where {N,T} = Point{N,T}(Tuple(index))

# Implement part of the API of `N`-tuples and iterators. NOTE: For `getindex`, bound
# checking cannot be avoided for tuples.
Base.eltype(A::Point) = eltype(typeof(A))
Base.eltype(::Type{Point{N,T}}) where {T,N} = T
Base.length(A::Point{N,T}) where {T,N} = N
@inline Base.getindex(A::Point, i) = getindex(Tuple(A), i)
Base.IteratorSize(::Type{<:Point}) = Base.HasLength()
Base.IteratorEltype(::Type{<:Point}) = Base.HasEltype()
@inline Base.iterate(@nospecialize(iter::Point), i::Int = 1) =
    1 ≤ i ≤ length(iter) ? (iter[i], i + 1) : nothing

Base.firstindex(A::Point) = 1
Base.lastindex(A::Point) = length(A)
Base.eachindex(A::Point) = Base.OneTo(length(A))
Base.keys(A::Point) = eachindex(A)

# Rounding to nearest coordinates.
Base.round(::Type{Point{N,T}}, A::Point{N,T}, r::RoundingMode = RoundNearest) where {T,N} = A
Base.round(::Type{Point{N,T}}, A::Point{N}, r::RoundingMode = RoundNearest) where {T,N} =
    Point{N,T}(round(NTuple{N,T}, A, r))
Base.round(::Type{CartesianIndex{N}}, A::Point{N}, r::RoundingMode = RoundNearest) where {N} =
    round(CartesianIndex, A, r)
Base.round(::Type{CartesianIndex}, A::Point{N}, r::RoundingMode = RoundNearest) where {N} =
    CartesianIndex(round(NTuple{N,Int}, A, r))
Base.round(::Type{NTuple{N,T}}, A::Point{N,T}, r::RoundingMode = RoundNearest) where {T,N} = A.coords
Base.round(::Type{NTuple{N,T}}, A::Point{N}, r::RoundingMode = RoundNearest) where {T,N} =
    map(x -> round(T, x, r), A.coords)

# Real to nearest integer.
nearest(::Type{T}, x::AbstractFloat) where {T<:Integer} = round(T, x)
nearest(::Type{T}, x::Real         ) where {T<:Integer} = round(T, float(x))

# N-dimensional coordinates to Cartesian index.
nearest(::Type{CartesianIndex}, A::CartesianIndex) = A
nearest(::Type{CartesianIndex}, A::Point) = nearest(CartesianIndex, A.coords)
nearest(::Type{CartesianIndex}, A::NTuple{N,Integer}) where {N} = CartesianIndex(A)
nearest(::Type{CartesianIndex}, A::NTuple{N,Real}) where {N} =
    CartesianIndex(map(nearest(Int), A))

# Get rid of the `N` parameter in `CartesianIndex{N}`.
nearest(::Type{CartesianIndex{N}}, A::CartesianIndex{N}) where {N} = A
nearest(::Type{CartesianIndex{N}}, A::NTuple{N}        ) where {N} = nearest(CartesianIndex, A)
nearest(::Type{CartesianIndex{N}}, A::Point{N}         ) where {N} = nearest(CartesianIndex, A)
Base.values(A::Point) = Tuple(A)

# Conversion of points to tuples. The `Point` constructors aloready implement conversion
# from tuples.
Base.Tuple(x::Point) = x.coords
Base.NTuple(x::Point) = Tuple(x)
Base.NTuple{N}(x::Point{N}) where {N} = Tuple(x)
Base.NTuple{N,T}(x::Point{N}) where {N,T} = NTuple{N,T}(Tuple(x))
Base.convert(::Type{T}, x::Point) where {T<:Tuple} = convert(T, Tuple(x))


# To nearest point.
nearest(::Type{Point{N,T}}, A::Point{N,T}       ) where {N,T} = A
nearest(::Type{Point{N,T}}, A::Point{N}         ) where {N,T} = Point{N,T}(map(nearest(T), A.coords))
nearest(::Type{Point{N,T}}, A::NTuple{N,T}      ) where {N,T} = Point{N,T}(A)
nearest(::Type{Point{N,T}}, A::NTuple{N}        ) where {N,T} = Point{N,T}(map(nearest(T), A))
nearest(::Type{Point{N,T}}, A::CartesianIndex{N}) where {N,T} = Point{N,T}(A)

# To nearest N-tuple.
nearest(::Type{NTuple{N,T}},   A::Point{N,T}       ) where {N,T} = Tuple(A)
nearest(::Type{NTuple{N,T}},   A::Point{N}         ) where {N,T} = map(nearest(T), Tuple(A))
nearest(::Type{NTuple{N,T}},   A::NTuple{N,T}      ) where {N,T} = A
nearest(::Type{NTuple{N,T}},   A::NTuple{N}        ) where {N,T} = map(nearest(T), A)
nearest(::Type{NTuple{N,T}},   A::CartesianIndex{N}) where {N,T} = nearest(NTuple{N,T}, Tuple(A))
nearest(::Type{NTuple{N,Int}}, A::CartesianIndex{N}) where {N}   = Tuple(A)

# Equality.
Base.:(==)(A::Point, B::Point) = false
Base.:(==)(A::Point{N}, B::Point{N}) where {N} = A.coords == B.coords

# Ordering.
Base.isless(A::Point, B::Point) = false
Base.isless(A::Point{N}, B::Point{N}) where {N} = A.coords < B.coords

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
for (LType, RType) in ((:Point, :Point), (:Point, :CartesianIndex), (:CartesianIndex, :Point))
    @eval begin
        # Addition.
        Base.:(+)(A::$(LType){N}, B::$(RType){N}) where {N} = Point(map(+, Tuple(A), Tuple(B)))

        # Subtraction.
        Base.:(-)(A::$(LType){N}, B::$(RType){N}) where {N} = Point(map(-, Tuple(A), Tuple(B)))

        # Equality.
        Base.:(==)(A::$(LType), B::$(RType)) = false
        Base.:(==)(A::$(LType){N}, B::$(RType){N}) where {N} = Tuple(A) == Tuple(B)

        # Ordering.
        Base.isless(A::$(LType), B::$(RType)) = false
        Base.isless(A::$(LType){N}, B::$(RType){N}) where {N} = Tuple(A) < Tuple(B)
    end
end

# zero(x) is the neutral element for the addition.
Base.zero(x::Point) = zero(typeof(x))
Base.zero(::Type{Point{N,T}}) where {N,T} = Point{N,T}(ntuple(Returns(zero(T)), Val(N)))

# one(x) yields a multiplicative identity for x.
Base.one(x::Point) = one(typeof(x))
Base.one(::Type{Point{N,T}}) where {N,T} = one(T)
