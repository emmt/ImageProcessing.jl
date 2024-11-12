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
