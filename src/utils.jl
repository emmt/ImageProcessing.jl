"""
    zerofill!(A) -> A

fills entries in `A` by zeros and returns `A`.

"""
zerofill!(A::AbstractArray) = fill!(A, zero(eltype(A)))

"""
    hard_thresholder(val, lvl)

yields `val` if `val > lvl` holds and `zero(val)` otherwise.

See also [`soft_thresholder`](@ref).

"""
hard_thresholder(val::T, lvl::T) where {T} =
    ifelse(val > lvl, val, zero(T)) # NOTE: `ifelse` is to avoid branching

"""
    soft_thresholder(val, lvl)

yields the nonnegative part of `val - lvl`.

See also [`nonnegative_part`](@ref), [`hard_thresholder`](@ref).

"""
soft_thresholder(val, lvl) = nonnegative_part(val - lvl)

"""
    nonnegative_part(x)

yields `x` if `x > zero(x)` holds and `zero(x)` otherwise.

See also [`soft_thresholder`](@ref), [`ImageProcessing.fast_max`](@ref).

"""
nonnegative_part(x) = fast_max(x, zero(x))

"""
    ImageProcessing.fast_max(a, b)

yields `a` if `a > b` holds and `b` otherwise. Arguments must have the same type.

This function is intended for fast computations (e.g. vectorized loops). If one of `a` or
`b` is a NaN, `b` is returned.

See also [`ImageProcessing.fast_min`](@ref).

"""
fast_max(a::T, b::T) where {T} = a > b ? a : b
@public fast_max

"""
    ImageProcessing.fast_min(a, b)

yields `a` if `a < b` holds and `b` otherwise. Arguments must have the same type.

This function is intended for fast computations (e.g. vectorized loops). If one of `a` or
`b` is a NaN, `b` is returned.

See also [`ImageProcessing.fast_max`](@ref).

"""
fast_min(a::T, b::T) where {T} = a < b ? a : b
@public fast_min

"""
    nearest(T::Type, x) -> y::T

yields `x` rounded to the nearest value or instance of type `T`.

"""
nearest(::Type{T}, x::T) where {T} = x
nearest(::Type{T}, x) where {T} = as(T, x) # by default, simply convert...

# Real to nearest integer.
nearest(::Type{T}, x::AbstractFloat) where {T<:Integer} = round(T, x)
nearest(::Type{T}, x::Real         ) where {T<:Integer} = round(T, float(x))

"""
    nearest(T::Type) -> f

yields a callable object `f`, such that `f(x)` yields `nearest(T, x)`.

"""
nearest(::Type{T}) where {T} = Nearest{T}()

struct Nearest{T} <: Function; end
(::Nearest{T})(x) where {T} = nearest(T, x)

"""
    ImageProcessing.quick_all_ones(A) -> bool

yields whether it can be quickly inferred that all values of array `A` are equal to one.
The test is meant to be fast, the result is based on the type of `A` and, at most, on a
single value of `A`.

See also: [`ImageProcessing.default_weights`](@ref).

"""
quick_all_ones(A::AbstractArray) = false
quick_all_ones(A::AbstractUniformArray) = isone(StructuredArrays.value(A))
@public quick_all_ones

"""
    ImageProcessing.default_weights(A)

yields a fast uniform array of ones of same axes as array `A`.

See also: [`ImageProcessing.quick_all_ones`](@ref).

"""
default_weights(A::AbstractArray) = FastUniformArray(one(eltype(A)), axes(A))
@public default_weights

"""
    endpoints(S) -> a, b

yields the *end-points* `a` and `b` of `S` such that `a ≤ x ≤ b` holds for all `x ∈ S`. If

Contrary to `extrema(S)` which throws for empty sets or collections, `endpoints(S)`
attempts to return `a` and `b` if possible. Hence, if `a ≤ b` does not hold, it may be
because `S` is empty or contains unbounded values such as `NaN`s.

"""
endpoints(I::Interval) = first(I), last(I)
endpoints(B::BoundingBox) = first(B), last(B)
endpoints(R::AbstractUnitRange) = first(R), last(R)
function endpoints(R::AbstractRange)
    a, b, s = first(R), last(R), step(R)
    return s < zero(s) ? (b, a) : (a, b)
end
function endpoints(B::CartesianIndices)
    tup = map(endpoints, ranges(B))
    return CartesianIndex(map(first, tup)), CartesianIndex(map(last, tup))
end
endpoints(A::AbstractArray) = isempty(A) ? endpoints_empty(eltype(A)) : extrema(A)

# Fallback method assuming argument is an iterator.
endpoints(iter) = _endpoints(Base.IteratorEltype(iter), Base.IteratorSize(iter), iter)
_endpoints(::Base.HasEltype, ::Base.IteratorSize, iter) =
    # Iterator length unknown but element type known: returns the endpoints of an empty
    # interval if calling `extrema` fails (whatever the reason).
    try; extrema(iter); catch; endpoints_empty(eltype(iter)); end
_endpoints(::Base.HasEltype, ::Base.HasLength, iter) =
    # Iterator length and element type known: returns the endpoints of an empty interval
    # if length is zero, oterwise calls `extrema`.
    length(iter) > 0 ? extrema(iter) : endpoints_empty(eltype(iter))
_endpoints(::Base.IteratorEltype, ::Base.IteratorSize, iter) =
    # Iterator length and element type both unknown: fallback to calling `extrema`.
    extrema(iter)

# Yields the endpoints of an empty set of values of type `T`.
endpoints_empty(::Type{T}) where {T} = (oneunit(T), zero(T))

"""
    ImageProcessing.has_integer_coordinates(x)

yields whether `x` has integer coordinates. Note that floating-point coordinates are
considered as integer if they have no fractional part, e.g. `4.0` is integer.

"""
has_integer_coordinates(x::CartesianIndex) = true
has_integer_coordinates(x::Point{N,<:Integer}) where {N} = true
has_integer_coordinates(x::Point) = all(isinteger, Tuple(x))
@public has_integer_coordinates

"""
    ImageProcessing.compare_coordinates(a, op, b)

compares `a` and `b` with comparison operator `op` and as if `a` and `b` be Cartesian
coordinates. `a` and `b` can be tuples, Cartesian indices, or points.

For example, if `compare_coordinates(a, <, b)`, returns whether `a` is less than `b`
considering their elements in reverse order (from the last to the first ones). For tuples
of Cartesian indices, this corresponds to the ordering of array elements in column-major
order (as regular Julia arrays).

!!! warning
    This method favors unrolling and avoids branching, it may not be suitable for tuples
    with many elements or for elements with complex types for which comparisons takes many
    operations.

""" compare_coordinates
@public compare_coordinates

const ComparisonOperator = Union{typeof(==),typeof(!=),
                                 typeof(<),typeof(<=),
                                 typeof(>),typeof(>=)}

@inline function compare_coordinates(a::Union{Tuple,CartesianIndex,Point},
                                     op::ComparisonOperator,
                                     b::Union{Tuple,CartesianIndex,Point})
    return compare_coordinates(to_tuple(a), op, to_tuple(b))
end

# Error catcher.
compare_coordinates(a::Tuple, op::ComparisonOperator, b::Tuple) =
    throw(ArgumentError(string("`compare_coordinates(a, ", op, ", b)` for `length(a) = ",
                               length(a), " and `length(b) = ", length(b), "`")))

# Rewrite comparison.
compare_coordinates(a::Tuple, ::typeof(> ), b::Tuple) =  compare_coordinates(b, <, a)
compare_coordinates(a::Tuple, ::typeof(>=), b::Tuple) =  compare_coordinates(b, <=, a)
compare_coordinates(a::Tuple, ::typeof(!=), b::Tuple) = !compare_coordinates(a, ==, b)

# Equality.
compare_coordinates(a::Tuple, ::typeof(==), b::Tuple) = a == b

# `<` and `<=` for empty tuples.
compare_coordinates(a::Tuple{}, ::typeof(< ), b::Tuple{}) = false
compare_coordinates(a::Tuple{}, ::typeof(<=), b::Tuple{}) = true

# `<` and `<=` for 1-tuples.
for op in (:(<), :(<=))
    @eval @inline compare_coordinates(a::Tuple{Any}, ::typeof($op), b::Tuple{Any}) =
        @inbounds $op(a[1], b[1])
end

# Otherwise, `<` and `<=` are applied in reverse order.
@inline function compare_coordinates(a::NTuple{N,Any}, op::Union{typeof(<),typeof(<=)},
                                     b::NTuple{N,Any}) where {N}
    return op(reverse(a), reverse(b))
end

to_tuple(x::Tuple) = x
to_tuple(x::Point) = Tuple(x)
to_tuple(x::CartesianIndex) = Tuple(x)

"""
    ImageProcessing.front(t::Tuple)

yields the front part of tuple `t`, that is all elements of `t` but the last one keeping
their order.

Also see `Base.front` and [`ImageProcessing.tail`](@ref).

""" front
@public front
@inline front(t::Tuple) = @inbounds t[1:length(t)-1]
@inline front(t::Tuple{Any}) = ()
front(t::Tuple{}) = throw(ArgumentError("cannot call `front` on an empty tuple"))

"""
    ImageProcessing.tail(t::Tuple)

yields the tail part of tuple `t`, that is all elements of `t` but the first one keeping
their order.

Also see `Base.tail` and [`ImageProcessing.front`](@ref).

""" tail
@public tail
@inline tail(t::Tuple) = _tail(t...)
tail(t::Tuple{}) = throw(ArgumentError("cannot call `tail` on an empty tuple"))
@inline _tail(_, x...) = x
