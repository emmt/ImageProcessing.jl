if VERSION < v"1.2.0-rc1"
    # `mapreduce` is not implemented for multiple iterators prior to Julia 1.2.
    mapreduce(f, op, itr; kw...) = Base.mapreduce(f, op, itr; kw...)
    mapreduce(f, op, itrs...; kw...) = Base.reduce(op, Base.Generator(f, itrs...); kw...)
    mapreduce(args...; kwds...) = Base.mapreduce(args...; kwds...)
end

if VERSION < v"1.7.0-beta1"
    # Returns is not defined prior to Julia 1.7.
    struct Returns{T}
        value::T
        Returns{T}(value) where {T} = new{T}(value)
        Returns(value::T) where {T} = new{T}(value)
    end
    (obj::Returns)(@nospecialize(args...); @nospecialize(kwds...)) = obj.value
end
