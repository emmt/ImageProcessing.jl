"""
    center_of_gravity([f,] A)
    center_of_gravity([f,] A, B)

yield the center of gravity of the values in `N`-dimensional array `A`. The result is a
`N`-dimensional point computed by the formula:

    (Σᵢ mass(i)*Point(i))/(Σᵢ mass(i))

with `Σᵢ` the sum over the indices `i` of `A` (and `B` if specified), `Point(i)` the
`N`-dimensional position at `i`, and `mass(i)` the *mass* given by:

```julia
mass(i) = f(A[i])           # if `B` is not specified
mass(i) = f(A[i], B[i])     # if `B` is specified
```

The value returned by the *mass function* `f` should be non-negative. If `f` is not
specified, the default *mass function* is similar to the following definition:

```julia
f(x, w=one(x)) = nonnegative_part(w*x)
```

With the default *mass function*, `A` typically represents measured intensities while `B`
represents weights.

To restrict the computations to an hyper-rectangular sub-region, use a view or a boxed
array of `A` (and `B`).

For efficiency, a user-defined *mass function* should avoid branching to favor
vectorization of computations. If you want to compute the center of gravity for, possibly
weighted, values above a level `lvl`, the *mass function* `f` may be defined by:

```julia
f(x) = soft_thresholder(x, lvl)
f(x, w) = soft_thresholder(w*x, lvl)
```

"""
center_of_gravity(A::AbstractArray) = center_of_gravity(default_mass, A)
center_of_gravity(A::AbstractArray{<:Any,N}, B::AbstractArray{<:Any,N}) where {N} =
    center_of_gravity(default_mass, A, B)

function center_of_gravity(f, A::AbstractArray{<:Any,N}) where {N}
    zero_mass = zero(f(zero(eltype(A))))
    num = zero_mass*Point(zero(CartesianIndex{N}))
    den = zero_mass
    @inbounds @simd for i in CartesianIndices(A)
        mass = f(A[i])
        num += mass*Point(i)
        den += mass
    end
    return num/den
end

function center_of_gravity(f, A::AbstractArray{<:Any,N},
                           B::AbstractArray{<:Any,N}) where {N}
    inds = axes(A)
    axes(B) == inds || throw(DimensionMismatch("arrays must have the same axes"))
    zero_mass = zero(f(zero(eltype(A)), zero(eltype(B))))
    num = zero_mass*Point(zero(CartesianIndex{N}))
    den = zero_mass
    @inbounds @simd for i in CartesianIndices(inds)
        mass = f(A[i], B[i])
        num += mass*Point(i)
        den += mass
    end
    return num/den
end

# A Boolean weight is considered as a strong zero (in the sense that `false*NaN -> 0`).
default_mass(x) = nonnegative_part(x)
default_mass(x, y) = default_mass(x*y)
default_mass(x, y::Bool) = ifelse(y, default_mass(x), zero(x))
default_mass(x::Bool, y) = default_mass(y, x)
default_mass(x::Bool, y::Bool) = x & y

"""
    ImageProcessing.default_origin(dim)
    ImageProcessing.default_origin(rng)
    ImageProcessing.default_origin(args...)
    ImageProcessing.default_origin((args...,))

yield the origin assumed by default in the `ImageProcessing` package. Arguments may be a
dimension length `dim` or an index range `rng` to yield the index of the central pixel
along this dimension or range. Arguments `args...` may also be any number of array
dimensions and/or array index ranges to yield the multi-dimensional index of the central
pixel.

The same conventions as in `fftshift` and `ifftshift` are made for dimensions of even
length.

"""
default_origin(inds::eltype(RelaxedArrayShape)...) = default_origin(inds)
default_origin(inds::RelaxedArrayShape) =
    CartesianIndex(map(default_origin, as_array_shape(inds)))
default_origin(dim::Integer) = _default_origin(1, dim)
default_origin(rng::AbstractUnitRange{<:Integer}) = _default_origin(first(rng), length(rng))

_default_origin(firstindex::Int, length::Int) =
    as(Int, firstindex) + div(as(Int, length), 2)

"""
    I = ImageProcessing.locate_maximum(A)

yields the `N`-dimensional index of the first maximal entry in the `N`-dimensional array
`A`.

"""
function locate_maximum(A::AbstractArray{T,N}) where {T,N}
    imax, vmax = firstindex(A), typemin(T)
    @inbounds for i in eachindex(A)
        v = A[i]
        if v > vmax
            imax, vmax = i, v
        end
    end
    if imax isa Node{N}
        return imax
    else
        return CartesianIndices(A)[imax]
    end
end
