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
    new_array(T::Type, args...) -> A
    new_array(T::Type, (args...,)) -> A

yield a new array with undefined elements of type `T` and shape specified by `args...`.
The shape consists in any number of array dimensions (integers) and/or axes
(integer-valued unit ranges). The shape may also be specified as a tuple. If all shape
parameters are integers or instances of `Base.OneTo`, an ordinary array of type `Array{T}`
is returned; otherwise, an offset array (wrapped on top of an ordinary array) is returned.

"""
new_array(::Type{T}, args::ArrayShapeArg...) where {T} = new_array(T, args)
new_array(::Type{T}, dims::ArraySizeLike) where {T} = Array{T}(undef, to_size(dims))
new_array(::Type{T}, shape::ArrayShape) where {T} =
    OffsetArray(Array{T}(undef, to_size(shape)), to_axes(shape))

"""
    ImageProcessing.to_dim(x) -> dim::Int

yields an array dimension (an `Int`), for `x` an array dimension (an integer) or
an array axis (an integer-valued unit range).

See also [`ImageProcessing.to_size`](@ref), [`ImageProcessing.to_axis`](@ref),
[`ImageProcessing.to_axes`](@ref), and [`ImageProcessing.new_array`](@ref).

"""
to_dim(x::Int) = x
to_dim(x::Integer) = as(Int, x)
to_dim(x::AbstractUnitRange{<:Integer}) = as(Int, length(x))
@public to_dim

"""
    ImageProcessing.to_axis(x) -> rng::AbstractUnitRange{Int}

yields an array axis, for `x` an array dimension (an integer) or an array axis (an
integer-valued unit range).

See also [`ImageProcessing.to_axes`](@ref), [`ImageProcessing.to_dim`](@ref),
[`ImageProcessing.to_size`](@ref), and [`ImageProcessing.new_array`](@ref).

"""
to_axis(x::Integer) = Base.OneTo{Int}(x)
to_axis(x::AbstractUnitRange{Int}) = x
to_axis(x::AbstractUnitRange{<:Integer}) = as(AbstractUnitRange{Int}, x)
@public to_axis

"""
    ImageProcessing.to_size(x) -> dims::Dims{N}

yields an `N`-dimensional array size corresponding to the array shape `x` specified as an
`N`-tuple of array dimensions (integers) and/or array axes (integer-valued unit ranges).

See also [`ImageProcessing.to_dim`](@ref), [`ImageProcessing.to_axes`](@ref),
[`ImageProcessing.to_axis`](@ref), and [`ImageProcessing.new_array`](@ref).

"""
to_size(x::ArrayShapeArg) = (to_dim(x),)
to_size(x::ArrayShape) = map(to_dim, x)
to_size(x::Dims) = x
@public to_size

"""
    ImageProcessing.to_axes(x) -> rngs::NTuple{N,AbstractUnitRange{Int}}

yields an `N`-dimensional array axes corresponding to the array shape `x` specified as an
`N`-tuple of array dimensions (integers) and/or array axes (integer-valued unit ranges).

See also [`ImageProcessing.to_axis`](@ref), [`ImageProcessing.to_size`](@ref),
[`ImageProcessing.to_dim`](@ref), and [`ImageProcessing.new_array`](@ref).

"""
to_axes(x::ArrayShapeArg) = (to_axis(x),)
to_axes(x::ArrayShape) = map(to_axis, x)
to_axes(x::Tuple{Vararg{AbstractUnitRange{Int}}}) = x
@public to_axes

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
