# NOTE: This file collects code for computations involving points, intervals, and
#       bounding-boxes. Indeed, in may cases, the operations are similar and having the
#       code side-by-side helps to ensure consistency.

# Dimensionality.
Base.ndims( ::Type{<:Point{N,T}}) where {N,T} = N
Base.ndims( ::Type{<:Interval{<:Number}}) = 1
Base.ndims( ::Type{<:Interval{<:Point{N}}}) where {N} = N
Base.ndims( ::Type{<:BoundingBox{N,T}}) where {N,T} = N

# Element type.
Base.eltype(::Type{<:Interval{T}}) where {T} = T
Base.eltype(::Type{<:BoundingBox{N,T}}) where {N,T} = Point{N,T}

# Extend base methods for intervals and bounding-boxes.
for class in (:Interval, :BoundingBox)
    @eval begin
        Base.first(  I::$class) = getfield(I, :start)
        Base.last(   I::$class) = getfield(I, :stop)
        Base.minimum(I::$class) = (assert_nonempty(I); return first(I))
        Base.maximum(I::$class) = (assert_nonempty(I); return last(I))
        Base.extrema(I::$class) = (assert_nonempty(I); return (first(I), last(I)))
    end
end

assert_nonempty(I::Interval) =
    isempty(I) ? throw(ArgumentError("interval must be non-empty")) : nothing

assert_nonempty(B::BoundingBox) =
    isempty(B) ? throw(ArgumentError("box must be non-empty")) : nothing

# Check for emptyness and build empty set of the given type.
Base.isempty(I::Interval) = !(first(I) ≤ last(I))
Base.empty(::Type{Interval{T}}) where {T} = Interval{T}()
Interval{T}() where {T} = Interval(endpoints_of_empty_set(T)...)

Base.isempty(B::BoundingBox) = any(isempty, intervals(B))
Base.empty(::Type{BoundingBox{N,T}}) where {N,T} = BoundingBox{N,T}()
BoundingBox{N,T}() where {N,T} = BoundingBox(endpoints_of_empty_set(Point{N,T})...)

# NOTE The result is irrelevant if interval is empty.
Base.clamp(x::T, I::Interval{T}) where {T} = clamp(x, I.start, I.stop)
Base.clamp(x, I::Interval) = clamp(promote(x, I.start, I.stop)...)

# Convert to floating-point.
Base.float(p::AbstractPoint{N,<:AbstractFloat}) where {N} = p
Base.float(p::AbstractPoint) = Point(map(float, Tuple(p)))

Base.float(i::Interval{<:AbstractFloat}) = i
Base.float(i::Interval) = Interval(float(first(i)), float(last(i)))

Base.float(b::BoundingBox{N,<:AbstractFloat}) where {N} = b
Base.float(b::BoundingBox) = BoundingBox(float(first(b)), float(last(b)))

# Constructors from an instance of the same class and `convert` are similar for points and
# bounding-boxes.
for type in (:Point, :BoundingBox)
    @eval begin
        $type(     x::$type)                  = x
        $type{N}(  x::$type{N})   where {N}   = x
        $type{N,T}(x::$type{N,T}) where {N,T} = x

        Base.convert(::Type{$type},      x::$type)                  = x
        Base.convert(::Type{$type{N}},   x::$type{N})   where {N}   = x
        Base.convert(::Type{$type{N,T}}, x::$type{N,T}) where {N,T} = x
        Base.convert(::Type{$type{N,T}}, x::$type{N})   where {N,T} = $type{N,T}(x)
    end
end
# ... we just have to provide conversion to a different element type.
BoundingBox{N,T}(b::BoundingBox{N}) where {N,T} = BoundingBox{N,T}(first(b), last(b))

# For intervals, constructors and conversions are also similar but there is no
# dimensionality to consider.
Interval(   i::Interval)              = i
Interval{T}(i::Interval{T}) where {T} = i
Interval{T}(i::Interval)    where {T} = Interval{T}(first(i), last(i))
#
Base.convert(::Type{Interval},    i::Interval)              = i
Base.convert(::Type{Interval{T}}, i::Interval{T}) where {T} = i
Base.convert(::Type{Interval{T}}, i::Interval)    where {T} = Interval{T}(i)

# Rules for type promotion.
Base.promote_rule(::Type{Interval{T1}}, ::Type{Interval{T2}}) where {T1,T2} =
    Interval{promote_type(T1, T2)}
Base.promote_rule(::Type{BoundingBox{N,T1}}, ::Type{BoundingBox{N,T2}}) where {N,T1,T2} =
    BoundingBox{N, promote_type(T1, T2)}
Base.promote_rule(::Type{<:AbstractPoint{N,T1}}, ::Type{<:AbstractPoint{N,T2}}) where {N,T1,T2} =
    Point{N, promote_type(T1, T2)}
Base.promote_rule(::Type{CartesianIndex{N}}, ::Type{<:AbstractPoint{N,T}}) where {N,T} =
    Point{N, promote_type(Int, T)}

# Rules for converting element type.
TypeUtils.convert_eltype(::Type{T}, ::Type{<:BoundingBox{N}}) where {N,T} = BoundingBox{N,T}
TypeUtils.convert_eltype(::Type{T}, ::Type{<:Interval}) where {T} = Interval{T}
TypeUtils.convert_eltype(::Type{T}, ::Type{<:Point{N}}) where {N,T} = Point{N,T}
TypeUtils.convert_eltype(::Type{T}, p::AbstractPoint) where {T} =
    # We must override the rule for abstract arrays in `TypeUtils`.
    as(convert_eltype(T, typeof(p)), p)

# Unary plus.
Base.:(+)(p::AbstractPoint) = p
Base.:(+)(i::Interval) = i
Base.:(+)(b::BoundingBox) = b

# Unary minus.
Base.:(-)(p::AbstractPoint) = Point(map(-, Tuple(p)))
Base.:(-)(i::Interval) = Interval(-last(i), -first(i))
Base.:(-)(b::BoundingBox) = BoundingBox(-last(b), -first(b))

# Consistency
# -----------
#
# Addition and subtraction of points, intervals, or bounding-boxes must be consistent with
# mutiplication and division by a scalar and with the neutral elements (`one(x)` and
# `zero(x)`) for the multiplication and for the addition/subtraction.

# `one(x)` yields a multiplicative identity for x.
Base.one(::Type{<:AbstractPoint{N,T}}) where {N,T} = one(T)
Base.one(::Type{Interval{T}}) where {T} = one(T)
Base.one(::Type{BoundingBox{N,T}}) where {N,T} = one(T)

# `zero(x)` is the neutral element for the addition.
Base.zero(::Type{<:AbstractPoint{N,T}}) where {N,T} = Point{N,T}(ntuple(Returns(zero(T)), Val(N)))
Base.zero(::Type{Interval{T}}) where {T} = Interval{T}(zero(T), zero(T))
Base.zero(::Type{BoundingBox{N,T}}) where {N,T} = BoundingBox{N,T}(zero(Point{N,T}), zero(Point{N,T}))

# For points, `oneunit(x)` follows the same semantics as for Cartesian indices.
Base.oneunit(::Type{<:AbstractPoint{N,T}}) where {N,T} = Point{N,T}(ntuple(Returns(oneunit(T)), Val(N)))
Base.oneunit(::Type{Interval{T}}) where {T} = Interval{T}(zero(T), oneunit(T))
Base.oneunit(::Type{BoundingBox{N,T}}) where {N,T} = BoundingBox{N,T}(zero(Point{N,T}), oneunit(Point{N,T}))

# Some methods are "traits" that only depend on the type.
for type in (:AbstractPoint, :Interval, :BoundingBox)
    for f in (:zero, :one, :oneunit, :eltype)
        f === :eltype && type === :AbstractPoint && continue
        @eval Base.$f(x::$type) = $f(typeof(x))
    end
end

# Multiplication of points, intervals and bounding-boxes by a scalar.
#
# NOTE: Multiplication and division of intervals or bounding-boxes by a scalar follow the
#       same rules.
Base.:(*)(p::AbstractPoint, alpha::Number) = alpha*p
Base.:(*)(alpha::Number, p::AbstractPoint) = Point(map(Base.Fix1(*, alpha), Tuple(p)))

Base.:(*)(i::Interval, alpha::Real) = alpha*i
function Base.:(*)(alpha::Real, i::Interval)
    x = alpha*first(i)
    isempty(i) && return empty(Interval{typeof(x)})
    y = alpha*last(i)
    return Interval(minmax(x, y)...)
end

Base.:(*)(b::BoundingBox, alpha::Real) = alpha*b
function Base.:(*)(alpha::Real, b::BoundingBox{N}) where {N}
    x = alpha*first(b)
    isempty(b) && return empty(BoundingBox{N,eltype(x)})
    y = alpha*last(b)
    return BoundingBox(minmax(x, y)...)
end

# Division of points, intervals and bounding-boxes by a scalar.
Base.:(\)(alpha::Number, p::AbstractPoint) = p/alpha
Base.:(/)(p::AbstractPoint, alpha::Number) = Point(map(Base.Fix2(/, alpha), Tuple(p)))

Base.:(\)(alpha::Real, i::Interval) = i/alpha
function Base.:(/)(i::Interval, alpha::Real)
    x = first(i)/alpha
    isempty(i) && return empty(Interval{typeof(x)})
    y = last(i)/alpha
    return Interval(minmax(x, y)...)
end

Base.:(\)(alpha::Real, b::BoundingBox) = b/alpha
function Base.:(/)(b::BoundingBox{N}, alpha::Real) where {N}
    x = first(b)/alpha
    isempty(b) && return empty(BoundingBox{N,eltype(x)})
    y = last(b)/alpha
    return BoundingBox(minmax(x, y)...)
end

# Binary operations between point-like objects.
for (A, B) in ((:AbstractPoint, :AbstractPoint),
               (:AbstractPoint, :CartesianIndex),
               (:CartesianIndex, :AbstractPoint))
    # Addition.
    @eval Base.:(+)(a::$A{N}, b::$B{N}) where {N} = Point(map(+, Tuple(a), Tuple(b)))

    # Subtraction.
    @eval Base.:(-)(a::$A{N}, b::$B{N}) where {N} = Point(map(-, Tuple(a), Tuple(b)))

    # Equality and ordering.
    @eval @inline Base.isequal(a::$A, b::$B) = isequal(Tuple(a), Tuple(b))
    @eval @inline Base.isless(a::$A{N}, b::$B{N}) where {N} =
        isless(reverse(Tuple(a)), reverse(Tuple(b)))

    # Comparison operators.
    for op in (:(==), :(!=), :(<), :(<=), :(>), :(>=))
        @eval begin
            Base.$op(a::$A,    b::$B) = false
            Base.$op(a::$A{N}, b::$B{N}) where {N} = compare_coordinates(a, $op, b)
        end
    end
end

# Addition and subtraction of intervals or bounding-boxes follow the same rules.
Base.:(+)(A::Interval, B::Interval) = +(promote(A, B)...)
Base.:(-)(A::Interval, B::Interval) = -(promote(A, B)...)
Base.:(+)(A::BoundingBox{N}, B::BoundingBox{N}) where {N} = +(promote(A, B)...)
Base.:(-)(A::BoundingBox{N}, B::BoundingBox{N}) where {N} = -(promote(A, B)...)
for type in (:Interval, :BoundingBox)
    @eval begin
        Base.:(+)(A::T, B::T) where {T<:$type} =
            isempty(A) ? B :
            isempty(B) ? A : $type(first(A) + first(B), last(A) + last(B))
        Base.:(-)(A::T, B::T) where {T<:$type} =
            isempty(A) ? B :
            isempty(B) ? A : $type(first(A) - last(B), last(A) - first(B))
    end
end

# Check equality of intervals and bounding-boxes. Note that `isequal` falls back to `==`.
Base.:(==)(A::Interval, B::Interval) =
    (isempty(A) & isempty(B)) | ((first(A) == first(B)) & (last(A) == last(B)))
Base.:(==)(A::BoundingBox{N}, B::BoundingBox{N}) where {N} =
    (isempty(A) && isempty(B)) || ((first(A) == first(B)) && (last(A) == last(B)))

# Addition and subtraction of a value to an interval or a point to a bounding-box yields
# the set resulting from the elementwise operation.
#
# NOTE: Emptyness must be checked otherwise adding a value to an empty interval could
#       yield a non-empty result. For instance, `Interval(1,0) + Inf` would yield
#       `Interval(Inf,Inf)` which is not considered as empty.
Base.:(+)(x, i::Interval) = i + x
function Base.:(+)(i::Interval, x)
    start, x, stop = promote(first(i), x, last(i))
    return isempty(i) ? Interval{typeof(x)}() : Interval(start + x, stop + x)
end
function Base.:(-)(i::Interval, x)
    start, x, stop = promote(first(i), x, last(i))
    return isempty(i) ? Interval{typeof(x)}() : Interval(start - x, stop - x)
end
function Base.:(-)(x, i::Interval)
    start, x, stop = promote(first(i), x, last(i))
    return isempty(i) ? Interval{typeof(x)}() : Interval(x - stop, x - start)
end

Base.:(+)(p::PointLike{N}, b::BoundingBox{N}) where {N} = b + p
function Base.:(+)(b::BoundingBox{N}, p::PointLike{N}) where {N}
    start, p, stop = promote(first(b), p, last(b))
    return isempty(b) ? BoundingBox{N,eltype(p)}() : BoundingBox(start + p, stop + p)
end
function Base.:(-)(b::BoundingBox{N}, p::PointLike{N}) where {N}
    start, p, stop = promote(first(b), p, last(b))
    return isempty(b) ? BoundingBox{N,eltype(p)}() : BoundingBox(start - p, stop - p)
end
function Base.:(-)(p::PointLike{N}, b::BoundingBox{N}) where {N}
    start, p, stop = promote(first(b), p, last(b))
    return isempty(b) ? BoundingBox{N,eltype(p)}() : BoundingBox(p - stop, p - start)
end

# Shifting of Cartesian index ranges by adding/subtracting a point. The result is an
# instance of `CartesinaIndices`.
Base.:(+)(p::AbstractPoint{N,<:Integer}, R::CartesianIndices{N}) where {N} = R + p
Base.:(+)(R::CartesianIndices{N}, p::AbstractPoint{N,<:Integer}) where {N} =
    EasyRanges.plus(R, CartesianIndex(p))
Base.:(-)(p::AbstractPoint{N,<:Integer}, R::CartesianIndices{N}) where {N} =
    EasyRanges.minus(CartesianIndex(p), R)
Base.:(-)(R::CartesianIndices{N}, p::AbstractPoint{N,<:Integer}) where {N} =
    EasyRanges.minus(R, CartesianIndex(p))

# `min()`, `max()`, and `minmax()` for points work as for Cartesian indices.
@inline Base.min(a::AbstractPoint{N}, b::AbstractPoint{N}) where {N} =
    Point(map(min, Tuple(a), Tuple(b)))
@inline Base.max(a::AbstractPoint{N}, b::AbstractPoint{N}) where {N} =
    Point(map(max, Tuple(a), Tuple(b)))
@inline function Base.minmax(a::AbstractPoint{N}, b::AbstractPoint{N}) where {N}
    t = map(minmax, Tuple(a), Tuple(b))
    return Point(map(first, t)), Point(map(last, t))
end

# Some math functions.
# NOTE `Base.hypot(Tuple(x::Point)...)` is a bit faster than
#      `LinearAlgebra.norm2(Tuple(x::Point))`.
LinearAlgebra.norm(a::AbstractPoint) = hypot(a)
LinearAlgebra.norm(a::AbstractPoint, p::Real) = LinearAlgebra.norm(Tuple(a), p)
Base.hypot(a::AbstractPoint) = hypot(Tuple(a)...)
Base.abs(a::AbstractPoint) = hypot(a)
Base.abs2(a::AbstractPoint) = mapreduce(abs2, +, Tuple(a))
Base.Math.atan(a::AbstractPoint{2}) = atan(a[1], a[2])
LinearAlgebra.dot(a::AbstractPoint{N}, b::AbstractPoint{N}) where {N} = mapreduce(*, +, Tuple(a), Tuple(b))
LinearAlgebra.cross(a::AbstractPoint{2}, b::AbstractPoint{2}) = a[1]*b[2] - a[2]*b[1]

#-----------------------------------------------------------------------------------------
# Inclusion of a value in an interval.
Base.in(x, i::Interval) = is_between(x, first(i), last(i))

# Inclusion of a point `p` in any set `S` that is iterable. More specific cases are
# handled in what follows.
Base.in(p::AbstractPoint, S) = _any_is_equal(p, S)
@inline _any_is_equal(p::AbstractPoint, S) = any(==(p), S)

# A point cannot be part of a collection if it is a collection of points of different
# dimensionality. This provides a shortcase in some common cases. It just have to be done
# with a signatue a bit more specific than above.

# Inclusion of a point `p` in `A`, an array of points.
Base.in(p::AbstractPoint{N}, A::AbstractArray{<:Point{M}}) where {N,M} = false
Base.in(p::AbstractPoint{N}, A::AbstractArray{<:Point{N}}) where {N} = _any_is_equal(p, A)

# Inclusion of a point `p` in `A`, a tuple of points.
Base.in(p::AbstractPoint{N}, A::Tuple{Vararg{Point{M}}}) where {N,M}  = false
Base.in(p::AbstractPoint{N}, A::Tuple{Point{N},Vararg{Point{N}}}) where {N} = _any_is_equal(p, A)

# Inclusion of a point in a bounding-box or in Cartesian indices.
Base.in(p::AbstractPoint,    B::BoundingBox) = false
Base.in(p::AbstractPoint{N}, B::BoundingBox{N}) where {N} =
    all_between(Tuple(p), Tuple(first(B)), Tuple(last(B)))

# Inclusion of a point in Cartesian indices. FIXME: uses R.indices.
Base.in(x::AbstractPoint,    R::CartesianIndices) = false
Base.in(x::AbstractPoint{N}, R::CartesianIndices{N}) where {N} =
    has_integer_coordinates(A) && all_between(Tuple(x), Tuple(first(R)), Tuple(last(R)))

# Inclusion of a Cartesian index in a bounding-box.
Base.in(A::CartesianIndex,    B::BoundingBox) = false
Base.in(A::CartesianIndex{N}, B::BoundingBox{N}) where {N} = Point(A) ∈ B

#-----------------------------------------------------------------------------------------

Base.issubset(A::Interval, B::Interval) =
    isempty(A) | ((first(B) ≤ first(A)) & (last(A) ≤ last(B)))

function Base.issubset(A::AbstractRange, B::Interval)
    start, stop = endpoints(A) # extract end-points once to save computations
    return !(start ≤ stop) | ((first(B) ≤ start) & (stop ≤ last(B)))
end

# Continous interval must be empty or a singleton to be possibly a subset of a discrete
# range.
Base.issubset(A::Interval, B::AbstractRange) =
    isempty(A) || (first(A) == last(A) && first(A) ∈ B)

# `A ⊆ B` for bounding-boxes `A` and `B`.
Base.issubset(A::BoundingBox,     B::BoundingBox)     = false
Base.issubset(A::BoundingBox,     B::BoundingBoxLike) = false
Base.issubset(A::BoundingBoxLike, B::BoundingBox)     = false

Base.issubset(A::Union{BoundingBox{N},CartesianIndices{N}}, B::BoundingBox{N}) where {N} =
    isempty(A) || (first(A) ∈ B && last(A) ∈ B)

Base.issubset(A::NTuple{N,IntervalLike}, B::BoundingBox{N}) where {N} =
    BoundingBox(A) ⊆ B

# A bounding-box, being a continuous set, can only be a subset of a discrete set if it is
# empty or if it consists in a single point that belongs to the discrete set. NOTE The
# same rule could be applied for `I` being any iterable or collection of points or
# cartesian indices but it's not possible or desirable to foresee every case.
Base.issubset(B::BoundingBox{N}, I::CartesianIndices{N}) where {N} =
    isempty(B) || (first(B) == last(B) && first(B) ∈ I)

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
EasyRanges.normalize(p::AbstractPoint{N,<:Integer}) where {N} = CartesianIndex(p)
EasyRanges.normalize(I::Interval) = I ∩ UnitRange{Int}
EasyRanges.normalize(B::BoundingBox) = B ∩ CartesianIndices
EasyRanges.ranges(I::Interval{<:Integer}) = (EasyRanges.normalize(I),)
EasyRanges.ranges(B::BoundingBox{N,<:Integer}) where {N} =
    map(UnitRange{Int}, first(B), last(B)) # FIXME: check this!

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
# Rounding and Co.

struct Round{T,R<:RoundingMode}
    Round(::Type{T}, r::R) where {T,R<:RoundingMode} = new{T,R}()
end
(::Round{T,R})(p) where {T,R} = round(T, p, R())

# Rounding coordinates to a tuple.
Base.round(::Type{Tuple}, p::AbstractPoint{N,T}, r::RoundingMode = RoundNearest) where {N,T} =
    round(NTuple{N,T}, p, r)
Base.round(::Type{NTuple}, p::AbstractPoint{N,T}, r::RoundingMode = RoundNearest) where {N,T} =
    round(NTuple{N,T}, p, r)
Base.round(::Type{NTuple{N}}, p::AbstractPoint{N,T}, r::RoundingMode = RoundNearest) where {N,T} =
    round(Point{N,T}, p, r)
Base.round(::Type{NTuple{N,T}}, p::AbstractPoint{N}, r::RoundingMode = RoundNearest) where {N,T} =
    map(Round(T,r), Tuple(p))
Base.round(::Type{NTuple{N,T}}, p::AbstractPoint{N,T}, r::RoundingMode = RoundNearest) where {T<:Integer,N} =
    Tuple(p)

# Rounding coordinates to a point.
Base.round(p::AbstractPoint{N,T}, r::RoundingMode = RoundNearest) where {N,T} = round(Point{N,T}, p, r)
Base.round(p::AbstractPoint{N,<:Integer}, r::RoundingMode = RoundNearest) where {N} = p
Base.round(::Type{Point}, p::AbstractPoint{N,T}, r::RoundingMode = RoundNearest) where {N,T} =
    round(Point{N,T}, p, r)
Base.round(::Type{Point{N}}, p::AbstractPoint{N,T}, r::RoundingMode = RoundNearest) where {N,T} =
    round(Point{N,T}, p, r)
Base.round(::Type{Point{N,T}}, p::AbstractPoint{N}, r::RoundingMode = RoundNearest) where {N,T} =
    Point(round(NTuple{N,T}, p, r))

# Rounding coordinates to a Cartesian index.
Base.round(::Type{CartesianIndex}, p::AbstractPoint{N}, r::RoundingMode = RoundNearest) where {N} =
    CartesianIndex(round(NTuple{N,Int}, p, r))
Base.round(::Type{CartesianIndex{N}}, p::AbstractPoint{N}, r::RoundingMode = RoundNearest) where {N} =
    round(CartesianIndex, p, r)

for (f, r) in ((:ceil, :RoundUp), (:floor, :RoundDown))
    @eval begin
        Base.$f(p::AbstractPoint) = round(p, $r)

        Base.$f(::Type{Point},      p::AbstractPoint{N,T}) where {N,T} = round(Point{N,T}, p, $r)
        Base.$f(::Type{Point{N}},   p::AbstractPoint{N,T}) where {N,T} = round(Point{N,T}, p, $r)
        Base.$f(::Type{Point{N,T}}, p::AbstractPoint{N}  ) where {N,T} = round(Point{N,T}, p, $r)

        Base.$f(::Type{Tuple},       p::AbstractPoint{N,T}) where {N,T} = round(NTuple{N,T}, p, $r)
        Base.$f(::Type{NTuple},      p::AbstractPoint{N,T}) where {N,T} = round(NTuple{N,T}, p, $r)
        Base.$f(::Type{NTuple{N}},   p::AbstractPoint{N,T}) where {N,T} = round(NTuple{N,T}, p, $r)
        Base.$f(::Type{NTuple{N,T}}, p::AbstractPoint{N}  ) where {N,T} = round(NTuple{N,T}, p, $r)

        Base.$f(::Type{CartesianIndex},    p::AbstractPoint{N}) where {N} = round(CartesianIndex, p, $r)
        Base.$f(::Type{CartesianIndex{N}}, p::AbstractPoint{N}) where {N} = round(CartesianIndex, p, $r)
    end
end

# To nearest Cartesian index.
nearest(::Type{CartesianIndex{N}}, p::CartesianIndex{N}) where {N} = p
nearest(::Type{CartesianIndex{N}}, p::NTuple{N}        ) where {N} = nearest(CartesianIndex, p)
nearest(::Type{CartesianIndex{N}}, p::AbstractPoint{N}         ) where {N} = nearest(CartesianIndex, p)
nearest(::Type{CartesianIndex},    p::CartesianIndex   ) = p
nearest(::Type{CartesianIndex},    p::AbstractPoint            ) = nearest(CartesianIndex, Tuple(p))
nearest(::Type{CartesianIndex},    p::NTuple{N,Integer}) where {N} = CartesianIndex(p)
nearest(::Type{CartesianIndex},    p::NTuple{N,Real}   ) where {N} =
    CartesianIndex(map(nearest(Int), p))

# To nearest point.
nearest(::Type{Point},      p::AbstractPoint) = p
nearest(::Type{Point},      p::Union{CartesianIndex,NTuple}) = Point(p)
nearest(::Type{Point{N}},   p::AbstractPoint{N}) where {N} = p
nearest(::Type{Point{N}},   p::Union{CartesianIndex{N},NTuple{N}}) where {N} = Point(p)
nearest(::Type{Point{N,T}}, p::AbstractPoint{N,T}       ) where {N,T} = p
nearest(::Type{Point{N,T}}, p::AbstractPoint{N}         ) where {N,T} = Point{N,T}(map(nearest(T), Tuple(p)))
nearest(::Type{Point{N,T}}, p::NTuple{N,T}      ) where {N,T} = Point{N,T}(p)
nearest(::Type{Point{N,T}}, p::NTuple{N}        ) where {N,T} = Point{N,T}(map(nearest(T), p))
nearest(::Type{Point{N,T}}, p::CartesianIndex{N}) where {N,T} = Point{N,T}(p)

# To nearest N-tuple.
nearest(::Type{Tuple},         p::AbstractPoint            ) = Tuple(p)
nearest(::Type{Tuple},         i::CartesianIndex   ) = Tuple(i)
nearest(::Type{Tuple},         t::Tuple            ) = t

nearest(::Type{NTuple},        p::AbstractPoint            ) = Tuple(p)
nearest(::Type{NTuple},        i::CartesianIndex   ) = Tuple(i)
nearest(::Type{NTuple},        t::NTuple           ) = t

nearest(::Type{NTuple{N}},     p::AbstractPoint{N}         ) where {N} = Tuple(p)
nearest(::Type{NTuple{N}},     i::CartesianIndex{N}) where {N} = Tuple(i)
nearest(::Type{NTuple{N}},     t::NTuple{N}        ) where {N} = t

nearest(::Type{NTuple{N,T}},   p::AbstractPoint{N,T}       ) where {N,T} = Tuple(p)
nearest(::Type{NTuple{N,T}},   p::AbstractPoint{N}         ) where {N,T} = nearest(NTuple{N,T}, Tuple(p))
nearest(::Type{NTuple{N,Int}}, i::CartesianIndex{N}) where {N}   = Tuple(i)
nearest(::Type{NTuple{N,T}},   i::CartesianIndex{N}) where {N,T} = nearest(NTuple{N,T}, Tuple(i))
nearest(::Type{NTuple{N,T}},   t::NTuple{N,T}      ) where {N,T} = t
nearest(::Type{NTuple{N,T}},   t::NTuple{N}        ) where {N,T} = map(nearest(T), t)
