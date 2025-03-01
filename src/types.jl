"""
    ImageProcessing.unspecified
    ImageProcessing.Unspecified()

singleton object to represent an unspecified optional argument or keyword.

""" unspecified
struct Unspecified end
const unspecified = Unspecified()
@public Unspecified unspecified
@doc unspecified Unspecified

"""
    AtLeastOne{T}

is an alias for n-tuples of at least one element of type `T`.

"""
const AtLeastOne{T} = Tuple{T,Vararg{T}}

struct Interval{T}
    start::T
    stop::T
    Interval{T}(start, stop) where {T} = new{T}(start, stop)
end

"""
    IntervalLike

is an alias to the union of types of objects `x` that can be directly converted into an
interval by calling the `Interval` constructor as `Interval(x)`. In other words, if `x isa
IntervalLike` holds then `Interval(x)` yields an interval.

See also [`Interval`](@ref), [`PointLike`](@ref), and [`BoundingBoxLike`](@ref).

"""
const IntervalLike = Union{Interval,AbstractRange}

"""
    AbstractPoint{N,T} <: AbstractVector{N}

is the abstract type of `N`-dimensional *points* having coordinates of type `T`. A point
`p` is also an (generally immutable) abstract vector whose `i`-th coordinate is given by
`p[i]`. For a point `p`, the number of coordinates given by `length(p)` is a *trait*: it
is inferable from the type.

For concrete point types, say `PointType`, apart from constructors given the coordinates,
at least the method `Base.values(p::PointType)` must be specialized to yield an indexable
(e.g. vector or tuple) list of the coordinates of `p`. In the API for abstract points,
points are assumed to have 1-based indices. If this is not the case, the base methods
`firstindex`, `lastindex`, and `keys` must also be specialized. For example:

```julia
Base.values(p::PointType) = p.coords
Base.firstindex(p::PointType) = ...
Base.lastindex(p::PointType) = length(p) - 1 + firstindex(p)
Base.keys(p::PointType) = firstindex(p):lastindex(p)
```

The default API is:

```julia
Base.IndexStyle(::Type{<:AbstractPoint}) = IndexLinear()
Base.length(p::AbstractVector{N,T}) where {N,T} = N
Base.firstindex(p::AbstractPoint) = 1
Base.lastindex(p::AbstractPoint) = length(p)
Base.keys(p::AbstractPoint) = Base.OneTo(length(p))
@propagate_inbounds Base.getindex(p::AbstractPoint, i) = getindex(values(p), i)
```

You may also want to specialize `Base.Tuple(p::PointType)` which yields a tuple of the
coordinates of the point `p` and whose default implementation is:

```julia
Base.Tuple(p::AbstractPoint{N,T}) where {N,T} = NTuple{N,T}(values(p))
```

This method is used in most methods implementing operations on the coordinates of abstract
points.

See also [`Point`](@ref), and [`PointLike`](@ref).

"""
abstract type AbstractPoint{N,T} <: AbstractVector{T} end

"""
    PointLike{N}

is an alias to the union of types of objects `x` that can be directly converted into a
`N`-dimensional point by calling the `Point` constructor as `Point(x)` or `Point{N}(x)`.
In other words, if `x isa PointLike{N}` holds then `Point(x)` yields a `N`-dimensional
point.

See also [`Point`](@ref), [`IntervalLike`](@ref), [`BoundingBoxLike`](@ref).

"""
const PointLike{N} = Union{AbstractPoint{N},CartesianIndex{N}}
# NOTE (1) The element type cannot be part of the signature because of Cartesian indices.
#      (2) `NTuple{N}` won't work because there is no guaranties that conversion to
#          a point is possible.

# Union of types to specify an `N`-dimensional Cartesian index.
const CartesianIndexLike{N} = Union{CartesianIndex{N},NTuple{N,Integer},
                                    AbstractPoint{N,<:Integer}}

# Union of types to specify a position in an `N`-dimensional array with possibly
# fractional coordinates.
const ArrayNode{N,T<:Real} = Union{CartesianIndex{N},NTuple{N,T},AbstractPoint{N,<:T}}

struct Point{N,T} <: AbstractPoint{N,T}
    coords::NTuple{N,T}

    # The following inner constructor relies on the `convert` base method to convert the
    # coordinates if needed. Since argument is an `N`-tuple, parameter `N` is guaranteed
    # to be an `Int`, no needs to check.
    Point{N,T}(coords::NTuple{N,Any}) where {N,T} = new{N,T}(coords)
end

struct BoundingBox{N,T}
    start::Point{N,T}
    stop::Point{N,T}
    BoundingBox{N,T}(start, stop) where {N,T} = new{N,T}(start, stop)
end

"""
    BoundingBoxLike{N}

is an alias to the union of types of objects `x` that can be directly converted into a
`N`-dimensional bounding-box by calling the `BoundingBox` constructor as `BoundingBox(x)`
or `BoundingBox{N}(x)`. In other words, if `x isa BoundingBoxLike{N}` holds then
`BoundingBox(x)` yields a `N`-dimensional bounding-box.

See also [`BoundingBox`](@ref), [`IntervalLike`](@ref), and [`PointLike`](@ref).

"""
const BoundingBoxLike{N} = Union{BoundingBox{N},CartesianIndices{N},NTuple{N,IntervalLike}}

# Since Julia 1.6 CartesianIndices may have non-unit steps.
const ContiguousCartesianIndices{N} =
    CartesianIndices{N,<:NTuple{N,AbstractUnitRange{<:Integer}}}

struct OnlineSum{T,N,A<:AbstractArray{<:Any,N},B<:AbstractArray{<:Any,N}}
    den::A # array to store the integrated denominator
    num::B # array to store the integrated numerator
    bad::T # value of bad pixels in the (weighted) mean
    org::NTuple{N,Int}
end
