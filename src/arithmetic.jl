Base.float(p::Point{N,<:AbstractFloat}) where {N} = p
Base.float(p::Point) = Point(map(float, Tuple(p)))

Base.float(i::Interval{<:AbstractFloat}) = i
Base.float(i::Interval) = Interval(float(i.start), float(i.stop))

Base.float(b::BoundingBox{N,<:AbstractFloat}) where {N} = b
Base.float(b::BoundingBox) = BoundingBox(float(b.start), float(b.stop))

# For points and bounding-boxes, constructors from an instance of the same class,
# `convert`, `convert_eltype`, and `promote_rule` are similar.
for class in (:Point, :BoundingBox)
    @eval begin
        $class(     x::$class)                  = x
        $class{N}(  x::$class{N})   where {N}   = x
        $class{N,T}(x::$class{N,T}) where {N,T} = x

        Base.convert(::Type{$class},      x::$class)                  = x
        Base.convert(::Type{$class{N}},   x::$class{N})   where {N}   = x
        Base.convert(::Type{$class{N,T}}, x::$class{N,T}) where {N,T} = x
        Base.convert(::Type{$class{N,T}}, x::$class{N})   where {N,T} = $class{N,T}(x)

        TypeUtils.convert_eltype(::Type{T}, ::Type{<:$class{N}}) where {N,T} = $class{N,T}

        Base.promote_rule(::Type{$class{N,T1}}, ::Type{$class{N,T2}}) where {N,T1,T2} =
            $class{N,promote_type(T1,T2)}
    end
end
# ... we just have to provide conversion to a different element type.
Point{N,T}(pnt::Point{N}) where {N,T} = Point{N,T}(Tuple(pnt))
BoundingBox{N,T}(box::BoundingBox{N}) where {N,T} = BoundingBox{N,T}(box.start, box.stop)
# ... in addition a cartesian index can be promoted to a point.
Base.promote_rule(::Type{CartesianIndex{N}}, ::Type{Point{N,T}}) where {N,T} =
    Point{N,promote_type(T,Int)}

# For intervals, conversions and promote rules are also similar but there is no
# dimensionality to consider.
Interval(   x::Interval)              = x
Interval{T}(x::Interval{T}) where {T} = x
Interval{T}(x::Interval)    where {T} = Interval{T}(x.start, x.stop)
#
Base.convert(::Type{Interval},    x::Interval)              = x
Base.convert(::Type{Interval{T}}, x::Interval{T}) where {T} = x
Base.convert(::Type{Interval{T}}, x::Interval)    where {T} = Interval{T}(x)
#
TypeUtils.convert_eltype(::Type{T}, ::Type{<:Interval}) where {T} = Interval{T}
#
Base.promote_rule(::Type{Interval{T1}}, ::Type{Interval{T2}}) where {T1,T2} =
    Interval{to_same_concrete_type(T1, T2)}

#-----------------------------------------------------------------------------------------
# Inclusion of a value in an interval.
Base.in(x, I::Interval) = (I.start ≤ x)&(x ≤ I.stop)

# Inclusion of a point in anything that is iterable. More specific cases are handles in
# what follows.
Base.in(A::Point, B) = _in(A, B)
@inline _in(A::Point, B) = any(==(A), B)

# A point cannot be part of a collection if it is a collection of points of different
# dimensionality. This provides a shortcase in some common cases. It just have to be done
# with a signatue a bit more specific than above.

# Inclusion of a point in an array of points.
Base.in(A::Point{N}, B::AbstractArray{<:Point{M}}) where {N,M} = false
Base.in(A::Point{N}, B::AbstractArray{<:Point{N}}) where {N} = _in(A, B)

# Inclusion of a point in a tuple of points.
Base.in(A::Point{N}, B::Tuple{Vararg{Point{M}}}) where {N,M}  = false
Base.in(A::Point{N}, B::Tuple{Point{N},Vararg{Point{N}}}) where {N} = _in(A, B)

# Inclusion of a point in a bounding-box or in Cartesian indices.
Base.in(A::Point, B::BoundingBox) = false
Base.in(A::Point{N}, B::BoundingBox{N}) where {N} = first(B) ≤ A ≤ last(B)

# Inclusion of a point in Cartesian indices.
Base.in(A::Point, B::CartesianIndices) = false
Base.in(A::Point{N}, B::CartesianIndices{N}) where {N} =
    has_integer_coordinates(A) && Point(first(B)) ≤ A ≤ Point(last(B))

# Inclusion of a Cartesian index in a bounding-box.
Base.in(A::CartesianIndex, B::BoundingBox) = false
Base.in(A::CartesianIndex{N}, B::BoundingBox{N}) where {N} = Point(A) ∈ B

#-----------------------------------------------------------------------------------------

Base.issubset(A::Interval, B::Interval) =
    isempty(A) | ((first(B) ≤ first(A)) & (last(A) ≤ last(B)))

function Base.issubset(A::AbstractRange, B::Interval)
    a, b = endpoints(A)
    return !(a ≤ b) || ((first(B) ≤ a) & (b ≤ last(B)))
end

# Continous interval must be empty or a singleton to be possibly a subset of a discrete
# range.
Base.issubset(A::Interval, B::AbstractRange) =
    isempty(A) || (first(A) == last(A) && first(A) ∈ B)

# `A ⊆ B` for bounding-boxes `A` and `B`.
Base.issubset(A::BoundingBox,     B::BoundingBox)      = false
Base.issubset(A::BoundingBox,     B::BoundingBoxLike) = false
Base.issubset(A::BoundingBoxLike, B::BoundingBox)      = false

Base.issubset(A::BoundingBox{N}, B::BoundingBox{N}) where {N} =
    isempty(A) || (first(B) ≤ first(A) && last(A) ≤ last(B))

Base.issubset(A::CartesianIndices{N}, B::BoundingBox{N}) where {N} =
    isempty(A) || (first(B) ≤ Point(first(A)) && Point(last(A)) ≤ last(B))

# A bounding-box, being a continuous set, can only be a subset of a discrete set if it is
# empty or if it consists in a single point that belongs to the discrete set. NOTE The
# same rule could be applied for `B` being any iterable or collection of points or
# cartesian indices but it's not possible or desirable to foresee every case.
function Base.issubset(A::BoundingBox{N}, B::CartesianIndices{N}) where {N}
    start, stop = endpoints(A)
    return !(start ≤ stop) || (start == stop && start ∈ B)
end

#-----------------------------------------------------------------------------------------

"""
    infimum(I::Interval) -> minimum(I)

yields the minimal value of the interval.

---
    infimum(args::Interval...)    -> I::Interval
    infimum(Interval, args...)    -> I::Interval
    infimum(Interval{T}, args...) -> I::Interval{T}

yield the minimal value of the interval. For multiple arguments yield the largest interval
that is contained by all the intervals defined by each of `args...`. The element type `T`
of the resulting interval may be explicitly specified.

The *infimum* is also given the intersection of all the intervals.

See also `Base.intersect`, `∩`, and [`supremum`](@ref).

"""
infimum(I::Interval) = minimum(I) # NOTE this definition also is used for further doc. references
# FIXME infimum(A::AbstractRange) = A
# FIXME infimum(A::AbstractRange, B::AbstractRange...) = intersect(A, B...)

"""
    infimum(B::BoundingBox)            -> minimum(B)
    infimum(args::BoundingBox...)      -> B::BoundingBox
    infimum(BoundingBox, args...)      -> B::BoundingBox
    infimum(BoundingBox{N}, args...)   -> B::BoundingBox{N}
    infimum(BoundingBox{N,T}, args...) -> B::BoundingBox{N,T}

yield the largest bounding-box that is contained by all the bounding-boxes defined by each
of `args...`. The dimensionality `N` and the element type `T` of the resulting
bounding-box may be explicitly specified.

The *infimum* is also given the intersection of all the bounding-boxes.

See also `Base.intersect`, `∩`, and [`supremum`](@ref).

"""
infimum(B::BoundingBox) = minimum(B) # NOTE this definition also is used for further doc. references

"""
    supremum(I::Interval)          -> minimum(I)
    supremum(args::Interval...)    -> I::Interval
    supremum(Interval, args...)    -> I::Interval
    supremum(Interval{T}, args...) -> I::Interval{T}

yield the smallest interval that contains all the intervals defined by each of `args...`.
The element type `T` of the resulting interval may be explicitly specified.

See also [`infimum`](@ref).

"""
supremum(A::Interval) = A # NOTE this definition also is used for further doc. references

"""
    B = supremum(args::BoundingBox...)
    B = supremum(BoundingBox, args...)
    B = supremum(BoundingBox{N}, args...)
    B = supremum(BoundingBox{N,T}, args...)

yield the smallest bounding-box that contains all the bounding-boxes defined by each of
`args...`. The dimensionality `N` and the element type `T` of the resulting bounding-box
may be explicitly specified.

See also [`infimum`](@ref).

"""
supremum(A::BoundingBox) = A # NOTE this definition also is used for further doc. references

# The following definitions work for `A` and `B` instances of the same type of interval or
# bounding-box.
infimum(A::T, B::T) where {T<:Union{Interval,BoundingBox}} =
    T(max(first(A), first(B)), min(last(A), last(B)))
supremum(A::T, B::T) where {T<:Union{Interval,BoundingBox}} =
    isempty(A) ? B :
    isempty(B) ? A :
    T(min(first(A), first(B)), max(last(A), last(B)))

for f in (:infimum, :supremum)
    @eval begin
        # Convert arguments into intervals.
        @inline $f(::Type{Interval}, A::IntervalLike, B::IntervalLike...) =
            $f(map(Interval, A, B...)...)
        @inline $f(::Type{Interval{T}}, A::IntervalLike, B::IntervalLike...) where {T} =
            $f(map(Interval{T}, A, B...)...)

        # When all arguments are intervals, make sure that all intervals have a common
        # element type to limit the propagation of rounding errors, then, proceed by
        # associativity from left to right.
        @inline $f(A::Interval, B::Interval, C::Interval...) = $f(promote(A, B, C...)...)
        @inline $f(A::Interval{T}, B::Interval{T}, C::Interval{T}...) where {T} =
            $f($f(A, B), C...)

        # Convert arguments into bounding-boxes.
        @inline $f(::Type{BoundingBox}, A::BoundingBoxLike{N}, B::BoundingBoxLike{N}...) where {N} =
            $f(map(BoundingBox, A, B...)...)
        @inline $f(::Type{BoundingBox{N}}, A::BoundingBoxLike{N}, B::BoundingBoxLike{N}...) where {N} =
            $f(map(BoundingBox, A, B...)...)
        @inline $f(::Type{BoundingBox{N,T}}, A::BoundingBoxLike{N}, B::BoundingBoxLike{N}...) where {N,T} =
            $f(map(BoundingBox{N,T}, A, B...)...)

        # When all arguments are bounding-boxes, make sure that all bounding-boxes have a
        # common element type to limit the propagation of rounding errors; then, proceed
        # by associativity from left to right.
        @inline $f(A::BoundingBox{N}, B::BoundingBox{N}, C::BoundingBox{N}...) where {N} =
            $f(promote(A, B, C...)...)
        @inline $f(A::BoundingBox{N,T}, B::BoundingBox{N,T}, C::BoundingBox{N,T}...) where {N,T} =
            $f($f(A, B), C...)
    end
end

# Intersections of continuous intervals (or similar) and bounding-boxes (or similar) of
# same dimensionality yield their infimum.
Base.intersect(A::Interval,     B::Interval)     = infimum(A, B)
Base.intersect(A::IntervalLike, B::Interval)     = infimum(A, B)
Base.intersect(A::Interval,     B::IntervalLike) = infimum(A, B)

Base.intersect(A::BoundingBox{N},     B::BoundingBox{N})     where {N} = infimum(A, B)
Base.intersect(A::BoundingBoxLike{N}, B::BoundingBox{N})     where {N} = infimum(A, B)
Base.intersect(A::BoundingBox{N},     B::BoundingBoxLike{N}) where {N} = infimum(A, B)

# Intersection of intervals with the set of unit-ranges.
Base.intersect(::Type{R}, I::Interval) where {R<:UnitRange} = I ∩ R
Base.intersect(I::Interval{T}, ::Type{UnitRange}) where {T} = I ∩ UnitRange{T}
function Base.intersect(I::Interval, ::Type{UnitRange{T}}) where {T}
    a, b = endpoints(I)
    return UnitRange{S}(_ceil(S, a), _floor(S, b))
end

# Intersection of bounding-boxes with the set of Cartesian indices.
Base.intersect(::Type{R}, B::BoundingBox) where {R<:CartesianIndices} = B ∩ R
Base.intersect(B::BoundingBox{N}, ::Type{CartesianIndices{N}}) where {N} = B ∩ CartesianIndices
function Base.intersect(B::BoundingBox{N,<:Real}, ::Type{CartesianIndices}) where {N}
    # Even though the conversion of the bounding-box end-points to integer coordinates
    # cannot result in a non-empty region if the box is initially empty, first check that
    # the bounding-box is not empty before attempting to convert coordinates to avoid
    # conversion errors due to overflows or NaNs.
    a, b = endpoints(B)
    if !(a ≤ b)
        # Empty bounding-box.
        a = ntuple(Returns(1), Val(N))
        b = ntuple(Returns(0), Val(N))
    else
        a = map(Fix1(_ceil,  Int), Tuple(a))
        b = map(Fix1(_floor, Int), Tuple(b))
    end
    return CartesianIndices(map(UnitRange, a, b))
end

# Conversion of continuous sets can be done automatically for `EasyRange` because the
# objective is to obtain a range of indices that belong to the interval.
EasyRanges.normalize(I::Interval) = I ∩ UnitRange{Int}
EasyRanges.normalize(B::BoundingBox) = B ∩ CartesianIndices
EasyRanges.ranges(I::Interval{<:Integer}) = (EasyRanges.normalize(I),)
EasyRanges.ranges(B::BoundingBox{N,<:Integer}) where {N} =
    map(UnitRange{Int}, B.start, B.stop) # FIXME: check this!

# Yields floor/ceil even though argument is integer.
_floor(::Type{T}, x::Integer) where {T} = as(T, x)
_floor(::Type{T}, x) where {T} = floor(T, x)

_ceil(::Type{T}, x::Integer) where {T} = as(T, x)
_ceil(::Type{T}, x) where {T} = ceil(T, x)

# Intersection of range and interval
# ----------------------------------
#
# When intersecting a discrete set, here a range R, with a continuous one, here an
# interval `I`, the result is of the same type of the discrete one and is given by
# `R[j:k]` where `j` and `k` are the first and last indices in `R` of the resulting range.
# However, `j` and `k` may be out of range (if the result is empty) and, to find `j` and
# `k` we need to precisely estimate `R[i]` for indices `i` that may be out of bounds.
# Hence, to simplify the code, we assume that `R[i]` can be computed for any `i` if we
# skip bounds check in all computations. This amounts to calling `unsafe_getindex` in
# `base/range.jl` but should be more general for other types of range.

Base.intersect(I::Interval, R::AbstractRange) = intersect(R, I)

function Base.intersect(R::AbstractRange, I::Interval)
    isempty(R) && return R
    ifirst = firstindex(R)
    j, k = ifirst, ifirst - 1 # initially assume an empty result
    @inbounds begin # computations must be done without bounds check
        a, b = first(I), last(I)
        a ≤ b || return R[j:k] # interval I is empty, so is the result
        rfirst = first(R)
        Rmin, Rmax = minmax(rfirst, last(R))
        ((Rmin > b) | (Rmax < a)) && return R[j:k]
        s = step(R)
        if s > zero(s)
            # Method for positive step.
            #
            # Find index j of first and lower bound such that R[j-1] < a ≤ R[j].
            if a ≤ Rmin
                j = ifirst
            else
                j = ceil(Int, (a - rfirst)/s) + ifirst
                while R[j] < a # adjust computed index to cope with rounding errors
                    j += 1
                end
                while a ≤ R[j-1] # adjust computed index to cope with rounding errors
                    j -= 1
                end
            end
            # Find index k of last and upper bound such that R[k] ≤ b < R[k+1].
            if Rmax ≤ b
                k = lastindex(R)
            else
                k = floor(Int, (b - rfirst)/s) + ifirst
                while b < R[k] # adjust computed index to cope with rounding errors
                    k -= 1
                end
                while R[k+1] ≤ b # adjust computed index to cope with rounding errors
                    k += 1
                end
            end
        else
            # Method for negative step.
            #
            # Find index k of last and lower bound such that R[k+1] < a ≤ R[k].
            if a ≤ Rmin
                k = lastindex(R)
            else
                k = floor(Int, (a - rfirst)/s) + ifirst
                while R[k] < a # adjust computed index to cope with rounding errors
                    k -= 1
                end
                while a ≤ R[k+1] # adjust computed index to cope with rounding errors
                    k += 1
                end
            end
            # Find index j of first and upper bound such that R[j] ≤ b < R[j-1].
            if Rmax ≤ b
                j = ifirst
            else
                j = ceil(Int, (b - rfirst)/s) + ifirst
                while b < R[j] # adjust computed index to cope with rounding errors
                    j += 1
                end
                while R[j-1] ≤ b # adjust computed index to cope with rounding errors
                    j -= 1
                end
            end
        end
        return R[j:k]
    end
end

function Base.intersect(R::AbstractUnitRange, I::Interval)
    # NOTE This is a special case of the method for non-unit step.
    isempty(R) && return R
    ifirst = firstindex(R)
    j, k = ifirst, ifirst - 1 # initially assume an empty result
    @inbounds begin # computations must be done without bounds check
        a, b = first(I), last(I)
        a ≤ b || return R[j:k] # interval I is empty, so is the result
        rfirst = first(R)
        Rmin, Rmax = minmax(rfirst, last(R))
        ((Rmin > b) | (Rmax < a)) && return R[j:k]
        # Find index j of first and lower bound such that R[j-1] < a ≤ R[j].
        if a ≤ Rmin
            j = ifirst
        else
            t = a - rfirst
            if t isa Integer
                t = as(Int, t) + ifirst
            else
                # Maybe adjust computed index to cope with rounding errors
                j = ceil(Int, t) + ifirst
                while R[j] < a
                    j += 1
                end
                while a ≤ R[j-1]
                    j -= 1
                end
            end
        end
        # Find index k of last and upper bound such that R[k] ≤ b < R[k+1].
        if Rmax ≤ b
            k = lastindex(R)
        else
            t = b - rfirst
            if t isa Integer
                t = as(Int, t) + ifirst
            else
                # Maybe adjust computed index to cope with rounding errors
                k = floor(Int, t) + ifirst
                while b < R[k]
                    k -= 1
                end
                while R[k+1] ≤ b
                    k += 1
                end
            end
        end
        return R[j:k]
    end
end

#-----------------------------------------------------------------------------------------
