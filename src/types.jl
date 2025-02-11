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

struct Point{N,T}
    coords::NTuple{N,T}

    # The following inner constructor relies on the `convert` base method to convert the
    # coordinates if needed.
    Point{N,T}(coords::NTuple{N,Any}) where {N,T} = new{N,T}(coords)
end

"""
    PointLike{N}

is an alias to the union of types of objects `x` that can be directly converted into a
`N`-dimensional point by calling the `Point` constructor as `Point(x)` or `Point{N}(x)`.
In other words, if `x isa PointLike{N}` holds then `Point(x)` yields a `N`-dimensional
point.

See also [`Point`](@ref), [`IntervalLike`](@ref), [`BoundingBoxLike`](@ref).

"""
const PointLike{N} = Union{Point{N},CartesianIndex{N}}
# NOTE (1) The element type cannot be part of the signature because of Cartesian indices.
#      (2) `NTuple{N}` won't work because there is no guaranties that conversion to
#          a point is possoble.

# Union of types to specify an `N`-dimensional Cartesian index.
const CartesianIndexLike{N} = Union{CartesianIndex{N},NTuple{N,Integer},Point{N,Integer}}

# Union of types to specify a position in an `N`-dimensional array with possibly
# fractional coordinates.
const ArrayNode{N,T<:Real} = Union{CartesianIndex{N},NTuple{N,T},Point{N,<:T}}

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
