"""

Module `LinearLeastSquares` (LLS) provides methods for solving, possibly weighted, linear
least square problems.

In the (weighted) least squares method, the `k`-th measured data `yₖ` is assumed to be given
by:

```
yₖ ≈ f(xₖ, c)
```

where `≈` accounts for measurement noise and model approximations, `f(xₖ,c)` is the value
predicted by the model for the (known) *independent variable* `xₖ` and (unknown) *model
parameters* `c`.

The objective of the (weighted) least squares method is to estimate the unknown model
parameters by:

```
ĉ = argmin_c Σₖ wₖ⋅(yₖ - f(xₖ, c))²
```

for given nonnegative weights `wₖ`.

For a linear model, the model writes:

```
f(xₖ, c) = c₁⋅f₁(xₖ) + c₂⋅f₂(xₖ) + ...
```

and the (weighted) least squares problem has a closed-form solution given by solving the
*normal equations*.

# Example

Suppose we want to fit the model `c₁ + c₂*x + c₃*sin(x)` to some data in array `Y` with
corresponding independent variables in array `X`. This can be done by:

```julia
eqs = NormalEquations{3,Float64}() # instantiate normal equations for 3 unknowns
for (x, y) in zip(X, Y)
    eqs = update(eqs, y, (1.0, x, sin(x)))
end
c = solve(eqs)
```

It is also possible to directly specify a list of model functions in `update`.

```julia
fns = (x -> 1.0, identity, sin)
eqs = NormalEquations{length(fns),Float64}()
for (x, y) in zip(X, Y)
    eqs = update(eqs, y, fns, x)
end
c = solve(eqs)
```

"""
module LinearLeastSquares

export
    AbstractNormalEquations,
    StaticNormalEquations,
    MutableNormalEquations,
    NormalEquations,
    rhs_vector,
    lhs_matrix,
    solve,
    solve!,
    update,
    update!

using ..ImageProcessing
using StaticArrays
using LinearAlgebra
using Neutrals
using TypeUtils
using Base: @propagate_inbounds

TypeUtils.@public(lazy_convert)

#------------------------------------------------------------------------------------- API -

"""
    AbstractNormalEquations

Abstract super-type of objects storing the coefficients of the so-called *normal equations*
which are a system of linear equations of the form `A*x = b` where `A` and `b` are
respectively is the *left-hand side* (LHS) matrix and *right-hand side* (RHS) vector of the
normal equations.

"""
abstract type AbstractNormalEquations end

"""
   StaticNormalEquations{N}

Abstract super-type of static (i.e. immutable) objects storing the coefficients of the
*normal equations*. For an object `eqs` of this kind, the coefficients of the normal
equations are typically updated by:

```julia
eqs = update(eqs, args...; kwds..)
```

Parameter `N` is the number of unknowns.

"""
abstract type StaticNormalEquations{N} <: AbstractNormalEquations end

"""
   MutableNormalEquations

Abstract super-type of mutable objects storing the coefficients of the *normal equations*.
For an object `eqs` of this kind, the coefficients of the normal equations are typically
updated by:

```julia
update!(eqs, args...; kwds..)
```

"""
abstract type MutableNormalEquations <: AbstractNormalEquations end

"""
    eqs = update(eqs::StaticAbstractEquations, args...; kwds...)

Return the updated normal equations in `eqs`. The result is another immutable object than
`eqs`.

"""
function update(eqs::StaticNormalEquations, args...; kwds...) end

"""
    update!(eqs::MutableAbstractEquations, args...; kwds...) -> eqs

Update the normal equations in `eqs` in-place and return `eqs`.

"""
function update!(eqs::MutableNormalEquations, args...; kwds...) end

"""
    lhs_matrix(eqs::AbstractNormalEquations; readonly=false) -> A

Return the *left-hand side* (LHS) matrix of the normal equations `eqs`.

Keyword `readonly` is to specify whether a read-only result is acceptable in which case `A`
may be shared by `eqs` and the caller must not modify the content of `A`.

# See also

[`AbstractNormalEquations`](@ref) and [`rhs_vector`](@ref).

"""
function lhs_matrix(eqs::AbstractNormalEquations; readonly::Bool=false) end

"""
    rhs_vector(eqs::AbstractNormalEquations; readonly=false) -> b

Return the *right-hand side* (RHS) vector of the normal equations `eqs`.


Keyword `readonly` is to specify whether a read-only result is acceptable in which case `b`
may be shared by `eqs` and the caller must not modify the content of `b`.

# See also

[`AbstractNormalEquations`](@ref) and [`lhs_matrix`](@ref).

"""
function rhs_vector(eqs::AbstractNormalEquations;  readonly::Bool=false) end

"""
    solve(eqs::AbstractNormalEquations; kwds...) -> c

Return the solution of the normal equations `eqs`. The content of `eqs` is left unchanged.

# See also

[`solve!`](@ref).

"""
function solve(eqs::AbstractNormalEquations)
    A = lhs_matrix(eqs)
    b = rhs_vector(eqs)
    # FIXME Use Cholesky decomposition for N not too small?
    return A\b
end

"""
    solve!(eqs::AbstractNormalEquations; kwds...) -> c

Return the solution of the normal equations `eqs`. This may be faster than calling
[`solve`](@ref) but the content of `eqs` may be destroyed which means that `eqs` may no
longer be used by [`update`](@ref), [`solve`](@ref), or `solve!`.

"""
function solve!(eqs::AbstractNormalEquations; kwds...) end

"""
    Indexable{T}

Type alias for indexable objects (i.e. tuples or vectors) with elements of type `T`.

"""
const Indexable{T} = Union{AbstractVector{T},Tuple{Vararg{T}}}

const Value{T<:Real} = Union{T,Neutral}
const Weight{T<:Real} = Union{T,typeof(ZERO),typeof(ONE)}

#------------------------------------------------ NormalEquations: Static normal equations -

"""
    eqs = NormalEquations{N,T}(A, b)

Return an immutable object `eqs` whose LHS matrix has packed coefficients given by `A` and
RHS vector has coefficients given by `b`. `N` is the number of unknowns in the normal
equations, it is also the length of `b`. `T` is the type of the coefficients stored by
`eqs`. The number of packed coefficients in `A` is equal to `(N*(N + 1)) ÷ 2`.

"""
struct NormalEquations{N,T<:AbstractFloat,L} <: StaticNormalEquations{N}
    # The coefficients of the LHS matrix are packed but can be in row-major or column-major
    # order.
    A::NTuple{L,T} # packed coefficients of the LHS matrix
    b::NTuple{N,T} # coefficients of the RHS vector
    # Return instantiated equations.
    @inline function NormalEquations{N,T}(A::NTuple{L,Real},
                                          b::NTuple{N,Real}) where {N,T<:AbstractFloat,L}
        L == packed_symmetrix_length(N) || throw(DimensionMismatch(
            "expecting $(packed_symmetrix_length(N)) packed coefficient(s) for `A`, got `$L`"))
        return new{N,T,L}(A, b)
    end
end

Base.eltype(eqs::NormalEquations) = eltype(typeof(eqs))
Base.eltype(::Type{<:NormalEquations{N,T}}) where {N,T} = T

"""
    eqs = NormalEquations{N,T}()
    eqs = zero(NormalEquations{N,T})

Return an immutable object `eqs` with coefficients of type `T` of `N` normal equations.
In the returned objects, all coefficients are `zero(T)`.

A typical usage of `eqs` is to update it by [`eqs = update(eqs, ...)`](@ref update) for each
data and eventually call [`solve(eqs)`])(@ref solve) to compute the solution of the normal
equations.

"""
NormalEquations{N,T}() where {N,T<:AbstractFloat} = zero(NormalEquations{N,T})
NormalEquations{N,T,L}() where {N,T<:AbstractFloat,L} = zero(NormalEquations{N,T,L})

Base.zero(eqs::NormalEquations) = zero(typeof(eqs))
function Base.zero(::Type{NormalEquations{N,T,L}}) where {N,T,L}
    N::Int
    L::Int
    L == packed_symmetrix_length(N) || throw(DimensionMismatch(
        "with `N=$N`, expecting `L=$(packed_symmetrix_length(N))`, got `L=$L`"))
    return zero(NormalEquations{N,T})
end

@inline function Base.zero(::Type{NormalEquations{N,T}}) where {N,T}
    N::Int
    A = ntuple(Returns(zero(T)), Val(packed_symmetrix_length(N)))
    b = ntuple(Returns(zero(T)), Val(N))
    return NormalEquations{N,T}(A, b)
end

"""
    k = packed_symmetrix_index(i::Int, j::Int, n::Int)

Return the index of element `A[i,j]` for a symmetric matrix in packed storage. All indices
are 1-based.

"""
function packed_symmetrix_index(i::Int, j::Int, n::Int)
    i, j = minmax(i, j)
    return div((2n - i)*(i - 1), 2) + j
end

packed_symmetrix_length(n::Int) = div(n*(n + 1), 2)

"""
    eqs = update(eqs::NormalEquations, ΔA, Δb)

Return the normal equations in `eqs` with the the LHS matrix incremented by `ΔA` and the RHS
vector incremented by `Δb`.

"""
update(eqs::NormalEquations{N,T,L}, A::NTuple{L,Real}, b::NTuple{N,Real}) where {N,T,L} =
    NormalEquations{N,T}(map(_update, eqs.A, A), map(_update, eqs.b, b))

# Add an increment to a value, preserving the type of the value.
_update(val::T, adj::Number) where {T<:Number} = (val + lazy_convert(T, adj))::T
_update(val::T, adj::T) where {T<:Number} = val + adj

"""
    eqs = update(eqs::NormalEquations, yₖ, fxₖ...; weight=wₖ)
    eqs = update(eqs::NormalEquations, yₖ, fxₖ; weight=wₖ)
    eqs = update(eqs::NormalEquations, wₖ, yₖ, fxₖ)

Update the coefficients of the normal equations stored by `eqs` for a new data value `yₖ`
and corresponding components `fxₖ = (f₁(xₖ), f₂(xₖ), ...)` of the linear model. `fxₖ` is
specified by the trailing arguments, by a tuple, or by a vector. In words, the assumed
linear model is:

```
yₖ ≈ c₁⋅f₁(xₖ) + c₂⋅f₂(xₖ) + ...
```

for some unknown parameters `c = (c₁, c₂, ...)`.

Keyword `weight` is to specify a statistical weight `wₖ` for `yₖ`. Typically, the weight is the
reciprocal of the variance of `yₖ`. If not specified, `weight=𝟙` is assumed.

For convenience, `fxₖ` can also be specified as a tuple of model functions followed by the
independent variable `xₖ`:

```
eqs = update(eqs, yₖ, (f₁, f₂, ...), xₖ; weight=wₖ)
eqs = update(eqs, wₖ, yₖ, (f₁, f₂, ...), xₖ) # equivalent
```

which are both the same as specifying `fxₖ` as `(f₁(xₖ), f₂(xₖ), ...)` or, if `xₖ` is a
tuple, as `(f₁(xₖ...), f₂(xₖ...), ...)`. For type inference, it is purposely not supported
to have a vector of model functions, they must be provided by a tuple.

"""
update(eqs::NormalEquations, y::Real, fx::Real...; weight::Real=ONE) = update(eqs, weight, y, fx)
update(eqs::NormalEquations, y::Real, fx::Indexable{<:Real}; weight::Real=ONE) =
    update(eqs, weight, y, fx)

function update(eqs::NormalEquations{N,T}, w::Union{Real,Neutral{1}}, y::Real,
                fx::NTuple{N,Real}) where {N,T<:AbstractFloat}
    # NOTE to speed-up some computations, neutrals are kept in `fx`
    return update(eqs, lazy_convert(T, w), convert(T, y), map(lazy_convert(T), fx))
end

function update(eqs::NormalEquations{N,T}, w::Union{Real,Neutral{1}}, y::Real,
                fx::Indexable{<:Real}) where {N,T<:AbstractFloat}
    length(fx) == N || throw(DimensionMismatch(
        "expecting $N model component(s) in `fx`, got $(length(fx))"))
    off = firstindex(fx) - 1
    return update(eqs, lazy_convert(T, w), convert(T, y),
                  # NOTE using lazy_convert below would be overkill
                  ntuple(i -> convert(T, @inbounds(fx[off + i])), Val(N)))
end

@generated function update(eqs::NormalEquations{N,T}, w::Weight{T}, y::T,
                           # NOTE to speed-up some operations, `fx` may contain neutrals
                           fx::NTuple{N,Value{T}}) where {N,T<:AbstractFloat}
    init = Expr[]
    A = Expr(:tuple)
    b = Expr(:tuple)
    k = 0 # index in packed eqs.A
    for i in 1:N
        wfx_i = Symbol("wfx_",i)
        push!(init, :($wfx_i = w*fx[$i]))
        for j in i:N
            k += 1
            push!(A.args, :(eqs.A[$k] + $wfx_i*fx[$j]))
        end
        push!(b.args, :(eqs.b[$i] + $wfx_i*y))
    end
    return quote
        if isfinite(w) && w > zero(w)
            $(init...)
            return NormalEquations{N,T}($(A), $(b))
        elseif iszero(w)
            return eqs
        else
            throw_invalid_weights(w)
        end
    end
end

@noinline throw_invalid_weights(w::Real) =
    throw(ArgumentError("weights must all be finite and nonnegative, got $w"))

# Model components specified as a tuple (or vector) of function followed by the independent
# variable.
function update(eqs::NormalEquations{N,T}, y::Real,
                fns::Tuple{Vararg{Function}}, x; weight::Real=ONE) where {N,T<:AbstractFloat}
    return update(eqs, weight, y, fns, x)
end
function update(eqs::NormalEquations{N,T}, w::Real, y::Real,
                fns::Tuple{Vararg{Function}}, x) where {N,T<:AbstractFloat}
    return update(eqs, w, y, mcall(NTuple{N,T}, fns, x))
end

@propagate_inbounds function _mcall(::Type{T}, fns::Indexable{<:Function},
                                    i::Int, x) where {T}
    return convert(T, fns[i](x))::T
end
@propagate_inbounds function _mcall(::Type{T}, fns::Indexable{<:Function},
                                    i::Int, x::Tuple) where {T}
    return convert(T, fns[i](x...))::T
end

function mcall(::Type{NTuple{N,T}},
               fns::NTuple{N,Function}, x) where {N,T<:AbstractFloat}
    return ntuple(i -> _mcall(T, fns, i, x), Val(N))::NTuple{N,T}
end

function mcall(::Type{NTuple{N,T}},
               fns::Indexable{<:Function}, x) where {N,T<:AbstractFloat}
    length(fns) == N || throw(DimensionMismatch(
        "expecting $N model function(s), got $(length(fns))"))
    return ntuple(i -> @inbounds(_mcall(T, fns, i, x)), Val(N))::NTuple{N,T}
end

@generated function lhs_matrix(eqs::NormalEquations{N,T}; readonly::Bool=false) where {N,T}
    A = Expr(:tuple)
    for i in 1:N
        for j in 1:N
            k = packed_symmetrix_index(i, j, N)
            push!(A.args, :(eqs.A[$k]))
        end
    end
    quote
        return SMatrix{N,N,T,N*N}($(A))
    end
end

rhs_vector(eqs::NormalEquations{N,T}; readonly::Bool=false) where {N,T} =
    SVector{N,T}(eqs.b)

solve!(eqs::NormalEquations) = solve(eqs)

"""
    LinearLeastSquares.lazy_convert(T, x) -> x′

Convert `x` to type `T` unless `x` is a neutral number in which case `x` is returned.

"""
lazy_convert(::Type{T}, x) where {T<:Number} = convert(T, x)::T
lazy_convert(::Type{T}, x::Neutral) where {T<:Number} = x

"""
    c = LinearLeastSquares.lazy_convert(T)

Return a callable object `c` such that `c(x)` yields `lazy_convert(T, x)`.

"""
lazy_convert(::Type{T}) where {T<:Number} = TypeUtils.Converter(lazy_convert, T)

end # module
