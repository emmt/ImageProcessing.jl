"""
    center_of_gravity(A; kwds...)

yields the center of gravity of the values in array `A`. The result is a `N`-dimensional
point with `N = ndims(A)` and computed by the formula:

    (sum_{i ∈ region} mass(i)*Point(i))/(sum_{i ∈ region} mass(i))

with `i ∈ region` the indices in the region of interest, `Point(i)` the `N`-dimensional
position at `i`, and:

    mass(i) = thresholder(weights[i]*A[i], threshold)

Keywords:

- `weights` is an array of element-wise nonnegative weights. By default, unit weights are
  assumed (i.e., as if `weights` is an array of ones).

- `region` is a `N`-tuple of index ranges to restrict the computation of the center of
  gravity to a rectangular sub-region of `A` (and of `weights` if specified).

- `threshold` is a threshold level, more specifically, the 2nd argument of the thresholder
  function. By default, `threshold = 0`. The threshold is converted to a suitable type.

- `thresholder` is a callable object to compute the *mass* of node at index `i` as
  `thresholder(weights[i]*A[i], threshold)` and which should yields a nonnegative result.
  By default, a hard-thresholder is used (see [`hard_thresholder`](@ref)). For efficiency,
  a user-defined thresholder should avoid branching to favor vectorization of
  computations.

"""
function center_of_gravity(A::AbstractArray{<:Any,N};
                           weights::AbstractArray{<:Any,N} = default_weights(A),
                           region::Union{NTuple{N,AbstractRange{<:Integer}},Unspecified} = unspecified,
                           threshold::Number = zero(eltype(weights))*zero(eltype(A)),
                           thresholder = hard_thresholder) where {N}
    # Define and check the region for computing the center of gravity.
    if region === unspecified
        region = axes(A)
    else
        is_subregion_of(region, A) || error("sub-region is not within array of values")
    end
    is_subregion_of(region, weights) || error("sub-region is not within array of weights")

    # Accumulate the numerator and the denominator of the center of gravity. There are 2
    # cases to consider depending on whether all weights are equal to one. The variables
    # to sum the numerator and the denominator must be initialized as zeros of suitable
    # types. Using a thresholder function avoids branching and thus favors vectorization.
    if quick_all_ones(weights)
        zero_mass = zero(eltype(weights))*zero(eltype(A))
        num = zero_mass*Point(zero(CartesianIndex{N}))
        den = zero_mass
        threshold = oftype(zero_mass, threshold)
        @inbounds @simd for i in CartesianIndices(region)
            mass = thresholder(A[i], threshold)
            num += mass*Point(i)
            den += mass
        end
    else
        zero_mass = zero(eltype(A))
        num = zero_mass*Point(zero(CartesianIndex{N}))
        den = zero_mass
        threshold = oftype(zero_mass, threshold)
        @inbounds @simd for i in CartesianIndices(region)
            mass = thresholder(weights[i]*A[i], threshold)
            num += mass*Point(i)
            den += mass
        end
    end
    return num/den
end

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
