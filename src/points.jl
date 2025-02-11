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
Base.length(::Type{<:Point{N,T}}) where {N,T} = N
Base.getindex(p::Point, ::Colon) = Tuple(p)
Base.getindex(p::Point, i) = getindex(Tuple(p), i)
Base.IteratorSize(::Type{<:Point}) = Base.HasLength()
Base.IteratorEltype(::Type{<:Point}) = Base.HasEltype()
@inline Base.iterate(iter::Point, i::Int = 1) =
    1 ≤ i ≤ length(iter) ? (iter[i], i + 1) : nothing

Base.firstindex(p::Point) = 1
Base.lastindex(p::Point) = length(p)
Base.eachindex(p::Point) = Base.OneTo(length(p))
Base.eachindex(::IndexLinear, p::Point) = Base.OneTo(length(p))
Base.keys(p::Point) = eachindex(p)
Base.values(p::Point) = Tuple(p)

# Conversion of points to tuples. The `Point` constructors already implement conversion
# from tuples.
Base.Tuple(p::Point) = getfield(p, :coords)
Base.NTuple(p::Point) = Tuple(p)
Base.NTuple{N}(p::Point{N}) where {N} = Tuple(p)
Base.NTuple{N,T}(p::Point{N}) where {N,T} = NTuple{N,T}(Tuple(p))
Base.convert(::Type{T}, p::Point) where {T<:Tuple} = convert(T, Tuple(p))

Base.show(io::IO,                      p::Point) = show(io, MIME"text/plain"(), p)
Base.show(io::IO, m::MIME"text/plain", p::Point) = _show(io, m, p)
Base.show(io::IO, m::MIME,             p::Point) = _show(io, m, p)
function _show(io::IO, m::MIME, p::Point{N}) where {N}
    show(io, m, typeof(p))
    write(io, "(")
    @inbounds for i in 1:N
        i > 1 && write(io, ", ")
        show(io, m, p[i])
    end
    write(io, ")")
end
