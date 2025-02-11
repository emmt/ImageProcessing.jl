"""
    I = Interval{T}(start, stop)

builds an interval object representing the continuous range of values `x` of type `T` such
that `start ≤ x ≤ stop`. Type parameter `T` is inferred from the arguments if omitted. If
`start ≤ stop` does not hold, the interval is considered as empty which can be tested by
`isempty(I)`.

There are several ways to retrieve the endpoints of an interval `I`:

```julia
start = I.start
start = first(I)
start = minimum(I) # throws if `I` is empty

stop = I.stop
stop = last(I)
stop = maximum(I) # throws if `I` is empty

start, stop = endpoints(I)
start, stop = extrema(I) # throws if `I` is empty
```

"""
Interval(start::T, stop::T) where {T} = Interval{T}(start, stop)
Interval(start, stop) = Interval(promote(start, stop)...)

"""
    I = Interval(rng::AbstractRange)

yields the interval that most tightly contains the values of the range `rng`.

An interval may not be automatically converted into a range (the `convert` method cannot
be used for that) because a range is a discrete set while an interval is a continuous set.
However, to retrieve the discrete range of values `R` that belong to an interval `I`, you
may intersect the interval with the expected range type (which is meant to represent the
set of all possible ranges in this context):

```julia
R = I ∩ UnitRange
R = I ∩ UnitRange{T}
```

Calling `EasyRanges.ranges(I)` from the `EasyRanges` package also yields this result. It
can be noted that this kind of conversion is restricted to unit-step ranges.

"""
Interval(rng::AbstractRange) = Interval(endpoints(rng)...)
