@static if !isdefined(Base, :Returns)
    """
        f = Returns(val)

    yields a callable object `f` such that `f(args...; kwds...) === val` always
    holds. This is similar to `Returns` which appears in Julia 1.7.

    You may call:

        f = Returns{T}(val)

    to force the returned value to be of type `T`.

    """
    struct Returns{T}
        value::T
        Returns{T}(value) where {T} = new{T}(value)
        Returns(value::T) where {T} = new{T}(value)
    end
    (obj::Returns)(@nospecialize(args...); @nospecialize(kwds...)) = getfield(obj, :value)
end

@static if !isdefined(Base, :Memory)
    const Memory{T} = Vector{T}
end

@static if !isdefined(Base, :ispositive)
    ispositive(x) = x > zero(x)
end
@static if !isdefined(Base, :isnegative)
    isnegative(x) = x < zero(x)
end
@static if !isdefined(Base, :isnonpositive)
    isnonpositive(x) = !isnegative(x)
end
@static if !isdefined(Base, :isnonnegative)
    isnonnegative(x) = !ispositive(x)
end
