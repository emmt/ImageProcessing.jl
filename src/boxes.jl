"""
    const BoxRange = UnitRange{Int}

alias for the type of index ranges in a box. Boxes have a single parameter: their number
of dimensions. This is needed to simplify having type stable vectors of boxes.

"""
const BoxRange = UnitRange{Int}
const BoxRanges{N} = NTuple{N,BoxRange}
@assert isconcretetype(BoxRange)

"""
    B = IndexBox(rngs...)

builds an object representing a `N`-dimensional hyper-rectangular region of contiguous
array indices given by the `N` unit-ranges `rngs...`. The `N`-tuple of ranges can be
retrieved by `B.indices` or `Tuple(B)`.

A box can be built from or converted into a `CartesianIndices` object but has unit-step
ranges.

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

# Conversion to/from `IndexBox` fall back to calling the constructor.
Base.convert(::Type{T}, x::T) where {T<:IndexBox} = x
Base.convert(::Type{T}, x) where {T<:IndexBox} = T(x)

# Constructor from CartesianIndices.
IndexBox(A::CartesianIndices) = IndexBox(A.indices)
IndexBox{N}(A::CartesianIndices{N}) where {N} = IndexBox(A)

# Convert to CartesianIndices
Base.CartesianIndices(A::IndexBox) = CartesianIndices(A.indices)
Base.CartesianIndices{N}(A::IndexBox{N}) where {N} = CartesianIndices(A)
Base.convert(::Type{T}, x::IndexBox) where {T<:CartesianIndices} = T(x)

"""
    B = IndexBox(arr)

build a box representing the axes of the array `arr`.

"""
IndexBox(A::AbstractArray) = IndexBox(axes(A))
IndexBox{N}(A::AbstractArray{<:Any,N}) where {N} = IndexBox(A)

# IndexBox constructors from a pair of Cartesian indices or of points.
for type in (:(CartesianIndex{N}), :(Point{N,<:Integer}))
    @eval begin
        #IndexBox((start,stop)::NTuple{2,$(type)}) where {N} = IndexBox(start, stop)
        IndexBox(start::$(type), stop::$(type)) where {N} =
            IndexBox(map(UnitRange{Int}, Tuple(start), Tuple(stop)))
        #IndexBox{N}((start,stop)::NTuple{2,$(type)}) where {N} = IndexBox(start, stop)
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

Base.isempty(A::IndexBox) = mapreduce(isless, |, Tuple(last(A)), Tuple(first(A)))

Base.view(A::AbstractArray{<:Any,N}, B::IndexBox{N}) where {N} =
    view(A, B.indices...)

Base.getindex(A::AbstractArray{<:Any,N}, B::IndexBox{N}) where {N} =
    getindex(A, B.indices...)

Base.ndims(B::IndexBox) = ndims(typeof(B))
Base.ndims(::Type{<:IndexBox{N}}) where {N} = N
Base.size(B::IndexBox) = map(length, B.indices)
Base.length(B::IndexBox) = prod(size(B); init=1)

OffsetArrays.OffsetArray{T}(::typeof(undef), B::IndexBox) where {T} =
    OffsetArray(Array{T}(undef, size(B)), B.indices)
OffsetArrays.OffsetArray{T,N}(::typeof(undef), B::IndexBox{N}) where {T,N} =
    OffsetArray{T}(undef, B)

Base.fill(val::T, B::IndexBox) where {T} = fill!(OffsetArray{T}(undef, B), val)

Base.zeros(::Type{T}, B::IndexBox) where {T} = fill(zero(T), B)
Base.zeros(B::IndexBox) = zeros(Float64, B)

Base.ones(::Type{T}, B::IndexBox) where {T} = fill(one(T), B)
Base.ones(B::IndexBox) = ones(Float64, B)
