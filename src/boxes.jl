"""
    B = BoundingBox{N,T}(start, stop)

yields an object representing the set of `N`-dimensional points `pnt` such that `start ≤
pnt ≤ stop` and whose coordinates are of type `T`. If omitted, type parameters `T` and `N`
are inferred form the arguments. A bounding-box can be seen as an hyper-rectangular region
whose axes are aligned with the Cartesian axes. If
`ImageProcessing.compare_coordinates(start, ≤, stop)` does not hold, the bounding-box is
empty. This can be checked by calling `isempty(B)`.

Arguments `start` and `stop` can be anything that can be `convert`ed to a `Point`: points,
`N`-tuples of coordinates, or Cartesian indices. If `start` and `stop` are both points,
the following syntax is supported:

    B = start:stop

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
BoundingBox(start::Point{N,T}, stop::Point{N,T}) where {N,T} = BoundingBox{N,T}(start, stop)
BoundingBox(start::Point{N}, stop::Point{N}) where {N} = BoundingBox(promote(start, stop)...)

Base.:(:)(start::Point{N}, stop::Point{N}) where {N} = BoundingBox(start, stop)

for P in (:(NTuple{N,Any}), :(CartesianIndex{N}))
    @eval begin
        BoundingBox(     start::$P, stop::$P) where {N}   = Point(start):Point(stop)
        BoundingBox{N}(  start::$P, stop::$P) where {N}   = Point(start):Point(stop)
        BoundingBox{N,T}(start::$P, stop::$P) where {N,T} = Point{N,T}(start):Point{N,T}(stop)
    end
end

# Extend base methods for bounding-boxes.
Base.first(box::BoundingBox) = box.start
Base.last( box::BoundingBox) = box.stop
Base.isempty(box::BoundingBox) = !(box.start ≤ box.stop)
Base.minimum(box::BoundingBox) = (assert_nonempty(box); return box.start)
Base.maximum(box::BoundingBox) = (assert_nonempty(box); return box.stop)
Base.extrema(box::BoundingBox) = (assert_nonempty(box); return (box.start, box.stop))
assert_nonempty(box::BoundingBox) =
    isempty(box) ? throw(ArgumentError("box must be non-empty")) : nothing

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

@inline function endpoints(::Type{T}, rngs::NTuple{N,IntervalLike}) where {N,T<:Point}
    e = map(endpoints, rngs)
    return T(map(first, e)), T(map(last, e))
end

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

@propagate_inbounds Base.view(A::AbstractArray{<:Any,N}, B::BoundingBox{N,<:Integer}) where {N} =
    view(A, ranges(B)...)

@propagate_inbounds Base.getindex(A::AbstractArray{<:Any,N}, B::BoundingBox{N,<:Integer}) where {N} =
    getindex(A, ranges(B)...)

new_array(::Type{T}, B::BoundingBox{N,<:Integer}) where {N,T} = new_array(T, ranges(B))

OffsetArrays.OffsetArray{T,N}(::typeof(undef), B::BoundingBox{N,<:Integer}) where {T,N} =
    OffsetArray{T}(undef, B)
function OffsetArrays.OffsetArray{T}(::typeof(undef), B::BoundingBox{N,<:Integer}) where {N,T}
    rngs = ranges(B)
    return OffsetArray(Array{T}(undef, map(length, rngs)), rngs)
end

Base.fill(val::T, box::BoundingBox{N,<:Integer}) where {N,T} = fill!(new_array(T, box), val)

# `zeros` and `ones` from a bounding-box with integer coordinates.
for (f, fs) in ((:zero, :zeros), (:one, :ones))
    @eval begin
        Base.$fs(box::BoundingBox{N,<:Integer}) where {N} = $fs(Float64, box)
        Base.$fs(::Type{T}, box::BoundingBox{N,<:Integer}) where {N,T} = fill($f(T), box)
    end
end

foo1(A::Point{N}, B::BoundingBox{N}) where {N} =
    mapreduce(in, &, Tuple(A), intervals(B); init=true)
# FIXME Base.in(A::Point{N}, B::Union{BoundingBox{N},CartesianIndices{N}}) where {N} =
# FIXME     mapreduce(in, &, Tuple(A), ranges(B); init=true)
# FIXME Base.in(A::CartesianIndex{N}, B::BoundingBox{N}) where {N} =
# FIXME     mapreduce(in, &, Tuple(A), ranges(B); init=true)

# Since Julia 1.6 CartesianIndices may have non-unit steps.
const ContiguousCartesianIndices{N} =
    CartesianIndices{N,<:NTuple{N,AbstractUnitRange{<:Integer}}}

intervals(B::Union{BoundingBox,ContiguousCartesianIndices}) =
    map(Interval, Tuple(first(B)), Tuple(last(B)))

# Shifting of Cartesian index ranges by adding/subtracting a point.
Base.:(+)(A::Point{N,<:Integer}, B::CartesianIndices{N}) where {N} = B + A
Base.:(+)(A::Union{CartesianIndex{N},Point{N,<:Integer}}, B::BoundingBox{N}) where {N} = B + A
Base.:(+)(A::BoundingBox{N}, B::Union{CartesianIndex{N},Point{N,<:Integer}}) where {N} =
    BoundingBox(map(EasyRanges.plus, ranges(A), Tuple(B)))
Base.:(+)(A::CartesianIndices{N}, B::Point{N,<:Integer}) where {N} =
    BoundingBox(map(EasyRanges.forward∘EasyRanges.plus, ranges(A), Tuple(B)))

Base.:(-)(A::BoundingBox{N}, B::Union{CartesianIndex{N},Point{N,<:Integer}}) where {N} =
    BoundingBox(map(EasyRanges.minus, ranges(A), Tuple(B)))
Base.:(-)(A::Union{CartesianIndex{N},Point{N,<:Integer}}, B::BoundingBox{N}) where {N} =
    BoundingBox(map(EasyRanges.forward∘EasyRanges.minus, Tuple(B), ranges(A)))

Base.:(-)(A::CartesianIndices{N}, B::Point{N,<:Integer}) where {N} =
    CartesianIndices(map(EasyRanges.forward∘EasyRanges.minus, ranges(A), Tuple(B)))
Base.:(-)(A::Point{N,<:Integer}, B::CartesianIndices{N}) where {N} =
    CartesianIndices(map(EasyRanges.forward∘EasyRanges.minus, Tuple(A), ranges(B)))
