"""
    B = BoundingBox{N,T}(start, stop)

yields an object representing the set of `N`-dimensional points `x` whose coordinates are
of type `T` and such that:

    start[i] ≤ x[i] ≤ stop[i]

holds for all `i ∈ 1:N`. If omitted, type parameters `T` and `N` are inferred form the
arguments.

A bounding-box can be seen as an hyper-rectangular region whose axes are aligned with the
Cartesian axes. Except for `N = 1`, the interval of points `Interval(start, stop)` is not
a bounding-box as it countains all points `x` such that `start ≤ x ≤ stop`.

If `B isa BoundingBox{N,T}`, `ndims(B)` yields `N` and `eltype(B)` yields
[`Point{N,T}`](@ref).

Arguments `start` and `stop` can be anything that can be `convert`ed to a `Point`: points,
`N`-tuples of coordinates, or Cartesian indices. If `start` and `stop` are both points,
the following syntax is supported:

    B = start:stop

If `ImageProcessing.compare_coordinates(start, ≤, stop)` does not hold, the bounding-box
is empty. This can be checked by calling `isempty(B)`.

There are several ways to retrieve the extreme points of a bounding-box `B`:

```julia
start = B.start
start = first(B)
start = minimum(B) # throws if `B` is empty

stop = B.stop
stop = last(B)
stop = maximum(B) # throws if `B` is empty

start, stop = endpoints(B)
start, stop = extrema(B) # throws if `B` is empty
```

The syntax `B.intervals` yields the list of intervals that define the bounding-box `B`.

"""
BoundingBox(   start::Point{N,T}, stop::Point{N,T}) where {N,T} = BoundingBox{N,T}(start, stop)
BoundingBox(   start::Point{N},   stop::Point{N})   where {N}   = BoundingBox(promote(start, stop)...)
BoundingBox{N}(start::Point{N},   stop::Point{N})   where {N}   = BoundingBox(start, stop)

Base.:(:)(start::Point{N}, stop::Point{N}) where {N} = BoundingBox(start, stop)

for P in (:(NTuple{N,Any}), :(CartesianIndex{N}))
    @eval begin
        BoundingBox(     start::$P, stop::$P) where {N}   = Point(start):Point(stop)
        BoundingBox{N}(  start::$P, stop::$P) where {N}   = Point(start):Point(stop)
        BoundingBox{N,T}(start::$P, stop::$P) where {N,T} = Point{N,T}(start):Point{N,T}(stop)
    end
end

"""
    B = BoundingBox(R::CartesianIndices{N})

yields the bounding-box representing a `N`-dimensional hyper-rectangular continuous region
that most tightly contains the Cartesian indices in `R`.

A bounding-box may not be automatically converted into a range of Cartesian indices (the
`convert` method cannot be used for that) because a range is a discrete set while a
bounding-box is a continuous set. However, to retrieve the discrete range of Cartesian
indices `R` that belong to a bounding-box `B`, you may intersect the bounding-box with the
type `CartesianIndices` (which is meant to represent the set of all possible ranges in
this context):

```julia
R = B ∩ CartesianIndices
```

Calling `EasyRanges.ranges(B)` from the `EasyRanges` package also yields this result.

"""
BoundingBox(     R::CartesianIndices)                = BoundingBox(endpoints(R)...)
BoundingBox{N}(  R::CartesianIndices{N}) where {N}   = BoundingBox(R)
BoundingBox{N,T}(R::CartesianIndices{N}) where {N,T} = BoundingBox{N,T}(endpoints(R)...)

"""
    B = BoundingBox(rngs...)

yields the bounding-box representing a `N`-dimensional hyper-rectangular continuous region
that most tightly contains the points whose coordinates are given by the `N` ranges or
intervals `rngs...`.

 array indices given by the `N` integer-valued unit-ranges
`rngs...`. Arguments may also be an instance of `CartesianIndices`.

As an example, the bounding-box of the Cartesian indices of an array `A` can be built by
`BoundingBox(axes(A))`.

A bounding-box may not be automatically converted into a range of Cartesian indices (the
`convert` method cannot be used for that) because a range is a discrete set while a
bounding-box is a continuous set. However, to retrieve the discrete range of Cartesian
indices `R` that belong to a bounding-box `B`, you may intersect the bounding-box with the
type `CartesianIndices` (which is meant to represent the set of all possible ranges in
this context):

```julia
R = B ∩ CartesianIndices
```

Calling `EasyRanges.ranges(B)` from the `EasyRanges` package also yields this result.

"""
BoundingBox(rngs::IntervalLike...) = BoundingBox(rngs)
BoundingBox(rngs::NTuple{N,IntervalLike}) where {N} =
    BoundingBox(endpoints(Point, rngs)...)

BoundingBox{N}(rngs::Vararg{IntervalLike,N}) where {N} = BoundingBox(rngs)
BoundingBox{N}(rngs::NTuple{N,IntervalLike}) where {N} = BoundingBox(rngs)

BoundingBox{N,T}(rngs::Vararg{IntervalLike,N}) where {N,T} = BoundingBox{N,T}(rngs)
BoundingBox{N,T}(rngs::NTuple{N,IntervalLike}) where {N,T} =
    BoundingBox(endpoints(Point{N,T}, rngs)...)

Base.propertynames(::BoundingBox) = (:intervals, :start, :stop,)
@inline Base.getproperty(B::BoundingBox, key::Symbol) =
    key === :intervals ? intervals(B)        :
    key === :start     ? getfield(B, :start) :
    key === :stop      ? getfield(B, :stop)  : throw(KeyError(key))

Base.show(io::IO, B::BoundingBox) = show(io, MIME"text/plain"(), B)
function Base.show(io::IO, ::MIME"text/plain", B::BoundingBox{N}) where {N}
    show(io, typeof(B))
    write(io, "(")
    flag = false
    for i in intervals(B)
        flag && print(io, ", ")
        print(io, first(i), ":", last(i))
        flag = true
    end
    write(io, ")")
end

# As a facility, an integer-valued bounding-box can be used to define the index ranges
# corresponding to this region in a discrete set of Cartesian indices.

TypeUtils.as_array_axes(B::BoundingBox{N,<:Integer}) where {N} =
    map(UnitRange{Int}, Tuple(first(B)), Tuple(last(B)))

@propagate_inbounds Base.view(A::AbstractArray{<:Any,N}, B::BoundingBox{N,<:Integer}) where {N} =
    view(A, as_array_axes(B)...)

@propagate_inbounds Base.getindex(A::AbstractArray{<:Any,N}, B::BoundingBox{N,<:Integer}) where {N} =
    getindex(A, as_array_axes(B)...)

TypeUtils.new_array(::Type{T}, B::BoundingBox{N,<:Integer}) where {N,T} =
    new_array(T, as_array_axes(B))

OffsetArrays.OffsetArray{T,N}(::typeof(undef), B::BoundingBox{N,<:Integer}) where {T,N} =
    OffsetArray{T}(undef, B)

function OffsetArrays.OffsetArray{T}(::typeof(undef), B::BoundingBox{N,<:Integer}) where {N,T}
    rngs = as_array_axes(B)
    return OffsetArray(Array{T}(undef, map(length, rngs)), rngs)
end

Base.fill(x::T, B::BoundingBox{N,<:Integer}) where {N,T} = fill!(new_array(T, B), x)

# `zeros` and `ones` from a bounding-box with integer coordinates.
for (f, fs) in ((:zero, :zeros), (:one, :ones))
    @eval begin
        Base.$fs(B::BoundingBox{N,<:Integer}) where {N} = $fs(Float64, B)
        Base.$fs(::Type{T}, B::BoundingBox{N,<:Integer}) where {N,T} = fill($f(T), B)
    end
end
