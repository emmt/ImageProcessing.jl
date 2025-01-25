"""
    const BoxRange = UnitRange{Int}

alias for the type of index ranges in a box. Boxes have a single parameter: their number
of dimensions. This is needed to simplify having type stable vectors of boxes.

"""
const BoxRange = UnitRange{Int}
@assert isconcretetype(BoxRange)
const BoxRanges{N} = NTuple{N,BoxRange}

"""
    B = IndexBox(rngs...)

builds an object representing a `N`-dimensional hyper-rectangular region of contiguous
array indices given by the `N` unit-ranges `rngs...`. The `N`-tuple of ranges can be
retrieved by `B.indices` or `Tuple(B)`.

A box can be built from or converted into a `CartesianIndices` object but has unit-step
ranges.

`IndexBox` cannot just be an alias to `CartesianIndices` (with unit range indices) because
some operations (like scaling or shifting) are implemented for `IndexBox` that do not
apply for `CartesianIndices`.

A box may also be built given the first and last of its multi-dimensioanl indices:

    B = IndexBox(start, stop)

where `start` and `stop` are `N`-dimensional Cartesian indices or points with integer
coordinates. The box is empty if `start ≤ stop` does not hold.

"""
IndexBox(rngs::ArrayAxisLike...) = IndexBox(rngs)
IndexBox{N}(rngs::Vararg{ArrayAxisLike,N}) where {N} = IndexBox(rngs)

# Constructor from another box.
IndexBox(A::IndexBox) = A
IndexBox{N}(A::IndexBox{N}) where {N} = A

# Conversion to `IndexBox` falls back to calling the constructor.
Base.convert(::Type{T}, x::T) where {T<:IndexBox} = x
Base.convert(::Type{T}, x) where {T<:IndexBox} = T(x)

# Constructor from CartesianIndices.
IndexBox(A::CartesianIndices) = IndexBox(A.indices)
IndexBox{N}(A::CartesianIndices{N}) where {N} = IndexBox(A)

# Convert to CartesianIndices.
Base.CartesianIndices(A::IndexBox) = CartesianIndices(A.indices)
Base.CartesianIndices{N}(A::IndexBox{N}) where {N} = CartesianIndices(A)
Base.convert(::Type{CartesianIndices}, x::IndexBox) = CartesianIndices(x)
Base.convert(::Type{T}, x::IndexBox{N}) where {N,T<:CartesianIndices{N}} = T(x)

"""
    B = IndexBox(arr)

build a box representing the axes of the array `arr`.

"""
IndexBox(A::AbstractArray) = IndexBox(axes(A))
IndexBox{N}(A::AbstractArray{<:Any,N}) where {N} = IndexBox(A)

# IndexBox constructors from a pair of Cartesian indices or of points.
for type in (:(CartesianIndex{N}), :(Point{N,<:Integer}), :(NTuple{N,Integer}))
    @eval begin
        IndexBox(start::$(type), stop::$(type)) where {N} =
            IndexBox(map(UnitRange{Int}, Tuple(start), Tuple(stop)))
        IndexBox{N}(start::$(type), stop::$(type)) where {N} =
            IndexBox(map(UnitRange{Int}, Tuple(start), Tuple(stop)))
    end
end

for func in (:first, :last)
    @eval begin
        Base.$func(A::IndexBox) = $func(CartesianIndex, A)
        Base.$func(::Type{Point{N,T}}, A::IndexBox{N}) where {N,T} =
            Point{N,T}(map($func, A.indices))
    end
    for class in (:Point, :CartesianIndex)
        @eval begin
            Base.$func(::Type{$class}, A::IndexBox) = $class(map($func, A.indices))
            Base.$func(::Type{$class{N}}, A::IndexBox{N}) where {N} = $func($class, A)
        end
    end
end

Base.show(io::IO, x::IndexBox) = show(io, MIME"text/plain"(), x)
function Base.show(io::IO, ::MIME"text/plain", x::IndexBox{N}) where {N}
    show(io, typeof(x))
    print(io, "(")
    flag = false
    for rng in x.indices
        flag && print(io, ", ")
        print(io, rng)
        flag = true
    end
    print(io, ")")
end

Base.isempty(A::IndexBox) = mapreduce(isless, |, Tuple(last(A)), Tuple(first(A)))

Base.view(A::AbstractArray{<:Any,N}, B::IndexBox{N}) where {N} =
    view(A, B.indices...)

Base.getindex(A::AbstractArray{<:Any,N}, B::IndexBox{N}) where {N} =
    getindex(A, B.indices...)

Base.ndims(B::IndexBox) = ndims(typeof(B))
Base.ndims(::Type{<:IndexBox{N}}) where {N} = N
Base.size(B::IndexBox) = map(length, B.indices)
Base.length(B::IndexBox) = prod(size(B))

OffsetArrays.OffsetArray{T}(::typeof(undef), B::IndexBox) where {T} =
    OffsetArray(Array{T}(undef, size(B)), B.indices)
OffsetArrays.OffsetArray{T,N}(::typeof(undef), B::IndexBox{N}) where {T,N} =
    OffsetArray{T}(undef, B)

Base.fill(val::T, B::IndexBox) where {T} = fill!(OffsetArray{T}(undef, B), val)

Base.zeros(::Type{T}, B::IndexBox) where {T} = fill(zero(T), B)
Base.zeros(B::IndexBox) = zeros(Float64, B)

Base.ones(::Type{T}, B::IndexBox) where {T} = fill(one(T), B)
Base.ones(B::IndexBox) = ones(Float64, B)

# `CartesianIndices` is more general than `IndexBox`. Both have `Int`-valued ranges.
Base.promote_rule(::Type{T}, ::Type{IndexBox{N}}) where {N,T<:CartesianIndices{N}} = T

# Inclusion `A ∈ B`.
Base.in(A::Point, B::Union{IndexBox,CartesianIndices}) = false
Base.in(A::Point{N,<:Integer}, B::Union{IndexBox{N},CartesianIndices{N}}) where {N} =
    mapreduce(in, &, A.coords, B.indices; init=true)
Base.in(A::CartesianIndex, B::IndexBox) = false
Base.in(A::CartesianIndex{N}, B::IndexBox{N}) where {N} =
    mapreduce(in, &, Tuple(A), B.indices; init=true)

# Intersection `A ∩ B` when, at least, one of `A` or `B` is an index box. When `A` or `B`
# is a `CartesianIndinces`, the result must be a `CartesianIndices` as it is more general.
# Note that `A ∩ B = B ∩ A` must hold.
Base.intersect(A::IndexBox{N}, B::IndexBox{N}) where {N} =
    IndexBox(map(intersect, A.indices, B.indices))
Base.intersect(A::CartesianIndices{N}, B::IndexBox{N}) where {N} = intersect(B, A)
Base.intersect(A::IndexBox{N}, B::CartesianIndices{N}) where {N} =
    CartesianIndices(map(intersect, A.indices, B.indices)) # FIXME: forward?

# `A ⊆ B`, when `A` is an `IndexBox` and `B` an `IndexBox` or a `CartesianIndices`.
Base.issubset(A::IndexBox, B::Union{IndexBox,CartesianIndices}) = false
Base.issubset(A::IndexBox{N}, B::Union{IndexBox{N},CartesianIndices{N}}) where {N} =
    mapreduce(issubset, &, A.indices, B.indices; init=true)

# Shifting of Cartesian index ranges by adding/subtracting a point.
Base.:(+)(A::Point{N,<:Integer}, B::CartesianIndices{N}) where {N} = B + A
Base.:(+)(A::Union{CartesianIndex{N},Point{N,<:Integer}}, B::IndexBox{N}) where {N} = B + A
Base.:(+)(A::IndexBox{N}, B::Union{CartesianIndex{N},Point{N,<:Integer}}) where {N} =
    IndexBox(map(EasyRanges.plus, A.indices, Tuple(B)))
Base.:(+)(A::CartesianIndices{N}, B::Point{N,<:Integer}) where {N} =
    IndexBox(map(EasyRanges.forward∘EasyRanges.plus, A.indices, Tuple(B)))

Base.:(-)(A::IndexBox{N}, B::Union{CartesianIndex{N},Point{N,<:Integer}}) where {N} =
    IndexBox(map(EasyRanges.minus, A.indices, Tuple(B)))
Base.:(-)(A::Union{CartesianIndex{N},Point{N,<:Integer}}, B::IndexBox{N}) where {N} =
    IndexBox(map(EasyRanges.forward∘EasyRanges.minus, Tuple(B), A.indices))

Base.:(-)(A::CartesianIndices{N}, B::Point{N,<:Integer}) where {N} =
    CartesianIndices(map(EasyRanges.forward∘EasyRanges.minus, A.indices, Tuple(B)))
Base.:(-)(A::Point{N,<:Integer}, B::CartesianIndices{N}) where {N} =
    CartesianIndices(map(EasyRanges.forward∘EasyRanges.minus, Tuple(A), B.indices))

# Extend `EasyRanges` package.
EasyRanges.normalize(B::IndexBox) = CartesianIndices(B)
EasyRanges.ranges(B::IndexBox) = B.indices
