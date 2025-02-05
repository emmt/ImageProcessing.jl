const ArraySizeLike{N} = NTuple{N,Integer}
const ArraySize{N} = NTuple{N,Int} # FIXME: Dims{N}

const ArrayAxisLike = AbstractUnitRange{<:Integer}
const ArrayAxis = AbstractUnitRange{Int}

const ArrayAxesLike{N} = NTuple{N,ArrayAxisLike}
const ArrayAxes{N} = NTuple{N,ArrayAxis}

const ArraySizeOrAxesLike{N} = Union{ArraySizeLike{N},ArrayAxesLike{N}}

"""
    ImageProcessing.unspecified
    ImageProcessing.Unspecified()

singleton object to represent an unspecified optional argument or keyword.

""" unspecified
struct Unspecified end
const unspecified = Unspecified()
@public Unspecified unspecified
@doc unspecified Unspecified

struct Point{N,T}
    coords::NTuple{N,T}

    # The following inner constructor relies on the `convert` base method to convert the
    # coordinates if needed.
    Point{N,T}(coords::NTuple{N,Any}) where {N,T} = new{N,T}(coords)
end

struct IndexBox{N}
    # The member name is the same as for `CartesianIndices` to help generalize code. The
    # other possibility is to call `EasyRanges.ranges`.
    indices::NTuple{N,UnitRange{Int}}

    # The following inner constructor relies on the `convert` base method to convert the
    # index ranges if needed.
    IndexBox(rngs::ArrayAxesLike{N}) where {N} = new{N}(rngs)

    # The following inner constructor just drops the type parameter `N`.
    IndexBox{N}(rngs::ArrayAxesLike{N}) where {N} = IndexBox(rngs)
end

# The following must hold for `IndexBox{N}` to be a concrete type for a given `N` provided
# it is consistent.
@assert isconcretetype(UnitRange{Int})

struct OnlineSum{T,N,A<:AbstractArray{<:Any,N},B<:AbstractArray{<:Any,N}}
    den::A # array to store the integrated denominator
    num::B # array to store the integrated numerator
    bad::T # value of bad pixels in the (weighted) mean
    org::NTuple{N,Int}
end
