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

# Outer constructors force a common concrete type and error catchers (to avoid stack
# overflow).
#
# NOTE My understanding is that a tuple can only be an `NTuple{N,T} where {N,T}` for
#      concrete type `T` otherwise it can only be a `NTuple{N,Any} where {N}`. This is
#      used to dispatch on these 2 possibilities so as to avoid `promote_type` if
#      possible.
Point(coords...) = Point(coords)
Point(coords::NTuple{N,T}) where {N,T} = Point{N,T}(coords)
@inline function Point(coords::NTuple{N,Any}) where {N}
    T = promote_type(map(typeof, coords)...)
    isconcretetype(T) || throw(ArgumentError(
        "coordinates cannot be converted to a common concrete type, you may explictly specify the type parameter `T` in constructor `Point{N,T}`"))
    return Point{N,T}(coords)
end

Point{N}(coords...) where {N} = Point{N}(coords)
Point{N}(coords::NTuple{N,Any}) where {N} = Point(coords)
Point{N}(coords::Tuple) where {N} = bad_point(N, Any, coords)

Point{N,T}(coords...) where {N,T} = Point{N,T}(coords)
Point{N,T}(coords::Tuple) where {N,T} = bad_point(N, T, coords)

# Cascading error catchers.
@noinline bad_point(N, T, coords) = throw(ArgumentError(
    "type parameter `T` in `Point{N,T}` must be a data type"))
@noinline bad_point(N, T::Type, coords) = throw(ArgumentError(
    "type parameter `N` in `Point{N,T}` must be an `Int`, got `$(typeof(N))`"))
@noinline function bad_point(N::Int, T::Type, coords)
    coords isa Tuple && length(coords) != N && throw(ArgumentError(
        "invalid $(length(coords))-tuple of coordinates for $N-dimensional point"))
    throw(ArgumentError(
        "invalid coordinates of type `typeof(coords)` for `Point{$N,$T}`"))
end

# 0-dimensional points are supported but type parameter `T` must be specified.
Point{0,T}() where {T} = Point{0,T}(())
Point(coords::Tuple{}) = error("missing type parameter `T` for 0-dimensional point")

# Conversion of Cartesian indices and abstract points into points.
for type in (:AbstractPoint, :CartesianIndex)
    @eval begin
        Point(     x::$type{N}) where {N}   = Point{N}(  Tuple(x))
        Point{N}(  x::$type{N}) where {N}   = Point{N}(  Tuple(x))
        Point{N,T}(x::$type{N}) where {N,T} = Point{N,T}(Tuple(x))
        Base.convert(::Type{Point},      x::$type{N}) where {N}   = Point(     Tuple(x))
        Base.convert(::Type{Point{N}},   x::$type{N}) where {N}   = Point{N}(  Tuple(x))
        Base.convert(::Type{Point{N,T}}, x::$type{N}) where {N,T} = Point{N,T}(Tuple(x))
    end
end

# Accessor.
Base.values(p::AbstractPoint) = getfield(p, :coords)

# Optimized conversion of points to tuples. The `Point` constructors already implement
# conversion from tuples.
Base.Tuple(p::Point) = values(p)

#-----------------------------------------------------------------------------------------
# Implement abstract vector API for abstract points.
Base.length(p::AbstractPoint{N,T}) where {N,T} = N
Base.size(p::AbstractPoint) = (length(p),)
Base.axes(p::AbstractPoint) = (keys(p),)
Base.IndexStyle(::Type{<:AbstractPoint}) = IndexLinear()
Base.firstindex(p::AbstractPoint) = 1
Base.lastindex(p::AbstractPoint) = length(p)
Base.keys(p::AbstractPoint) = Base.OneTo(length(p))
Base.getindex(p::AbstractPoint, ::Colon) = values(p)
@propagate_inbounds Base.getindex(p::AbstractPoint, i) = getindex(values(p), i)
Base.Tuple(p::AbstractPoint{N,T}) where {N,T} = NTuple{N,T}(values(p))
Base.NTuple(p::AbstractPoint) = Tuple(p)
Base.NTuple{N}(p::AbstractPoint{N}) where {N} = Tuple(p)
Base.NTuple{N,T}(p::AbstractPoint{N,T}) where {N,T} = Tuple(p)
Base.NTuple{N,T}(p::AbstractPoint{N}) where {N,T} = NTuple{N,T}(Tuple(p))
Base.convert(::Type{T}, p::AbstractPoint) where {T<:Tuple} = convert(T, Tuple(p))

# Implement part of the API of `N`-tuples and iterators. NOTE: For `getindex`, bound
# checking cannot be avoided for tuples. For `Point`, Base methods `eltype` and `length`
# follows the same semantics as `CartesianIndex`.
Base.IteratorSize(::Type{<:AbstractPoint}) = Base.HasLength()
Base.IteratorEltype(::Type{<:AbstractPoint}) = Base.HasEltype()
@propagate_inbounds Base.iterate(iter::AbstractPoint, i::Int = firstindex(iter)) =
    # NOTE Bounds are always checked for tuples, so only check upper bound.
    i ≤ lastindex(iter) ? (iter[i], i + 1) : nothing

# An abstract point with integer coordinates can be automatically converted (i.e. by
# `convert`) into a Cartesian index.
for type in (:(Base.CartesianIndex), :(Base.CartesianIndex{N}))
    @eval begin
        $type(p::AbstractPoint{N,<:Integer}) where {N} = CartesianIndex(Tuple(p))
        Base.convert(::Type{$type}, p::AbstractPoint{N,<:Integer}) where {N} =
            CartesianIndex(p)
    end
end

Base.show(io::IO,                      p::AbstractPoint) = show(io, MIME"text/plain"(), p)
Base.show(io::IO, m::MIME"text/plain", p::AbstractPoint) = _show(io, m, p)
Base.show(io::IO, m::MIME,             p::AbstractPoint) = _show(io, m, p)
function _show(io::IO, m::MIME, p::AbstractPoint{N}) where {N}
    show(io, m, typeof(p))
    write(io, "(")
    for i in 1:N
        i > 1 && write(io, ", ")
        show(io, m, p[i])
    end
    write(io, ")")
end
