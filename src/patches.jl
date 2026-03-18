"""
    patch = ImagePatch{T}(mask[, origin])

Return an object representing a *patch* in a `N`-dimensional array.

# Arguments

- `mask` is a `N`-dimensional array of `Bool` which indicates whether a cell is part of the
  patch or not.

- `origin` gives the coordinates of a point representing the *origin* of the patch. `origin`
  is a Cartesian index in `mask` with possibly factional units. If not specified, `origin`
  is the geometrical center of `mask`.

- `T` is the type of the coordinates of `origin`. If omitted it is inferred from `origin` if
  this argument is specified and it is assumed to be `Float64` otherwise.

Depending on the context, a *patch* may be called a *structuring element* (in mathematical
morphology), a *neighborhood*, or a *sliding window*.

# Properties

- `patch.mask` is the Boolean mask.

- `patch.origin` is the point considered as the origin (or center) of the patch.

"""
function ImagePatch(mask::AbstractArray{Bool,N}, origin::PointLike{N}) where {N}
    return ImagePatch(mask, Point(origin))
end
function ImagePatch(mask::AbstractArray{Bool,N}, origin::Point{N,T}) where {N,T}
    return ImagePatch{T}(mask, origin)
end

ImagePatch(mask::AbstractArray{Bool}) = ImagePatch{Float64}(mask)
function ImagePatch{T}(mask::AbstractArray{Bool,N}) where {T<:AbstractFloat, N}
    origin = Point{N,T}(map(r -> (T(first(r))::T + T(last(r))::T)/2, axes(mask)))
    return ImagePatch{T}(mask, origin)
end

# Conversion constructors.
ImagePatch(patch::ImagePatch) = patch
ImagePatch{T}(patch::ImagePatch{T}) where {T} = patch
ImagePatch{T}(patch::ImagePatch) where {T} = ImagePatch{T}(patch.mask, patch.origin)
ImagePatch{T,N}(patch::ImagePatch{<:Any,N}) where {T,N} = ImagePatch{T}(patch)
ImagePatch{T,N,A}(patch::ImagePatch{T,N,A}) where {T,N,A<:AbstractArray{Bool,N}} = patch
ImagePatch{T,N,A}(patch::ImagePatch{<:Any,N}) where {T,N,A<:AbstractArray{Bool,N}} =
    ImagePatch{T}(convert(A, patch.mask), patch.origin)

Base.convert(::Type{T}, x::T) where {T<:ImagePatch} = x
Base.convert(::Type{T}, x::ImagePatch) where {T<:ImagePatch} = T(x)::T

Base.ndims(A::ImagePatch) = ndims(typeof(A))
Base.ndims(::Type{<:ImagePatch{T,N}}) where {T,N} = N

for f in (:isequal, :(==))
    @eval begin
        function Base.$f(A::ImagePatch{<:Any,N}, B::ImagePatch{<:Any,N}) where {N}
            return $f(A.origin, B.origin) && $f(A.mask, B.mask)
        end
        Base.$f(A::ImagePatch, B::ImagePatch) = false
    end
end

TypeUtils.get_precision(A::ImagePatch) = get_precision(typeof(A))
TypeUtils.get_precision(::Type{<:ImagePatch{T,N}}) where {T,N} = get_precision(T)

TypeUtils.adapt_precision(::Type{T}, A::ImagePatch{S}) where {T<:TypeUtils.Precision,S} =
    ImagePatch{adapt_precision(T, S)}(A)

"""
    ImagePatch{T,N}(form::Val, shape...)

Return a `N`-dimensional *patch* with a given geometric `form` and array `shape`. The origin
of the patch is at its geometric center.

Arguments:

- `form` is `Val(:rectangular)` for a rectangular patch or `Val(:circular)` for a circular
  patch.

- `shape...` specifies a `N`-dimensional array shape. Each of `shape...` may be an integer
  or an index range. If `N` is specified, `shape` may also be a single dimension length or
  index range to assume the same `shape` for all `N` axes of the patch.

- `T` is the type of the coordinates of the patch origin. `T = Float64` by default.

- `N` is the number of dimensions of the patch. It must only be specified for a
  multi-dimensional patch and a single `shape`.

"""
ImagePatch(::Val{S}, shape::eltype(ArrayShape)...) where {S} = ImagePatch(Val(S), shape)
ImagePatch(::Val{S}, shape::ArrayShape{N}) where {S,N} = ImagePatch{Float64}(Val(S), shape)

ImagePatch{T,N}(::Val{S}, dim::eltype(ArrayShape)) where {S,T<:AbstractFloat,N} =
    ImagePatch{T}(Val(S), to_array_shape(Dims{N}, dim))
ImagePatch{T,N}(::Val{S}, shape::eltype(ArrayShape)...) where {S,T<:AbstractFloat,N} =
    ImagePatch{T,N}(Val(S), shape)
ImagePatch{T,N}(::Val{S}, shape::ArrayShape{N}) where {S,T<:AbstractFloat,N} =
    ImagePatch{T}(Val(S), shape)

function ImagePatch{T}(::Val{:rectangular}, shape::ArrayShape{N}) where {T<:AbstractFloat,N}
    return ImagePatch{T}(FastUniformArray(true, shape))
end

function ImagePatch{T}(::Val{:circular}, shape::ArrayShape{N}) where {T<:AbstractFloat,N}
    mask = new_array(Bool, shape)
    patch = ImagePatch{T}(mask)
    origin = patch.origin.coords
    radius = map(len -> T(len)/2, size(mask))
    for i in CartesianIndices(mask)
        mask[i] = mapreduce((i, c, r) -> abs2((i - c)/r), +, Tuple(i), origin, radius) ≤ ONE
    end
    return patch
end

function ImagePatch{T}(::Val{S}, shape::ArrayShape{N}) where {S,T<:AbstractFloat,N}
    throw(ArgumentError("unknown patch geometric form `$S`"))
end

to_array_shape(::Type{Dims{N}}, len::Integer) where {N} =
    ntuple(Returns(as_array_dim(len)), Val(N))
to_array_shape(::Type{Dims{N}}, rng::AbstractUnitRange{<:Integer}) where {N} =
    ntuple(Returns(as_array_axis(rng)), Val(N))
