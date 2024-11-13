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

See also [`soft_thresholder`](@ref).

"""
nonnegative_part(x) = fast_max(x, zero(x))

"""
    ImageProcessing.fast_max(a, b)

yields `a` if `a > b` holds and `b` otherwise. Arguments must have the same type.

This function is intended for fast computations (e.g. vectorized loops). If one of `a` or
`b` is a NaN, `b` is returned.

"""
fast_max(a::T, b::T) where {T} = a > b ? a : b

"""
    ImageProcessing.fast_min(a, b)

yields `a` if `a < b` holds and `b` otherwise. Arguments must have the same type.

This function is intended for fast computations (e.g. vectorized loops). If one of `a` or
`b` is a NaN, `b` is returned.

"""
fast_min(a::T, b::T) where {T} = a < b ? a : b

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

"""
quick_all_ones(A::AbstractArray) = false
quick_all_ones(A::AbstractUniformArray) = isone(StructuredArrays.value(A))

"""
    ImageProcessing.default_weights(A)

yields a fast uniform array of ones of same axes as array `A`.

"""
default_weights(A::AbstractArray) = FastUniformArray(one(eltype(A)), axes(A))
