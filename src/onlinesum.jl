"""
    S = OnlineSum{T}(dims...; kwds...)

yields an object to store a sum of, possibly weighted and/or recentered, *images* of size
`dims` and with values of type `T`.

It is also possible to build an instance of `OnlineSum` by providing the arrays to store
the denominator `den` and numerator `num` for each pixels of the weighted and/or
recentered image:

    S = OnlineSum{T}(den, num; kwds...)

In this case and if it is specified, `T` is the type of the division of an element of
`num` by one element of `den`.

Keyword `bad` is to specify the value of unmeasured pixels in the recentered image
(`zero(T)` by default). Keyword `origin` is to specify the Cartesian coordinates of the
*origin* in the recentered image.

To add one more image, say `A`, in `S` call:

    push!(S, A)

optionally with keywords `origin` and `weights` to specify the position of the origin in
`A` and pixelwise nonnegative weights. The origin is rounded to the nearest pixel
position.

Call `mean(S)`, to retrieve the (weighted) mean of the recentered images.

To reset `S`, call `zerofill!(S)`.

"""
OnlineSum{T}(dims::Integer...; kwds...) where {T} = OnlineSum{T}(dims; kwds...)
OnlineSum{T}(dims::NTuple{N,Integer}; kwds...) where {T,N} =
    OnlineSum{T}(as(Dims{N}, dims); kwds...)
function OnlineSum{T}(dims::Dims{N}; kwds...) where {T,N}
    typeof(oneunit(T)/oneunit(T)) === T || error("pixel type `$T` is not stable by the division")
    return OnlineSum(zeros(T, dims), zeros(T, dims); kwds...)
end
function OnlineSum(den::AbstractArray{<:Any,N},
                       num::AbstractArray{<:Any,N};
                       bad = nothing,
                       origin::ArrayNode{N} = default_origin(axes(num))) where {N}
    axes(num) == axes(den) || throw(DimensionMismatch("numerator and denominator have different axes"))
    T = typeof(oneunit(eltype(num))/oneunit(eltype(den)))
    bad = bad === nothing ? zero(T) : as(T, bad)
    return OnlineSum{T,N,typeof(den),typeof(num)}(den, num, bad, origin)
end

function zerofill!(A::OnlineSum)
    zerofill!(A.num)
    zerofill!(A.den)
    return A
end

function Statistics.mean(A::OnlineSum)
    num, den, bad = A.num, A.den, A.bad
    T = typeof(oneunit(eltype(num))/oneunit(eltype(den)))
    B = similar(num, T)
    @inbounds for i in eachindex(B, num, den)
        B[i] = ifelse(den[i] > zero(eltype(den)), as(T, num[i]/den[i]), bad)
    end
    return B
end

function Base.push!(A::OnlineSum{<:Any,N},
                    B::AbstractArray{<:Any,N};
                    weights::AbstractArray{<:Any,N} = default_weights(B),
                    origin = default_origin(axes(B))) where {N}
    num, den = A.num, A.den
    axes(num) == axes(den) || throw(DimensionMismatch("numerator and denominator have different axes"))
    axes(weights) == axes(B) || throw(DimensionMismatch("weights and data have different axes"))
    off = nearest(CartesianIndex{N}, origin) - CartesianIndex(A.org)
    R = @range CartesianIndices(num) ∩ (CartesianIndices(B) - off)
    if quick_all_ones(weights)
        @inbounds for i in R
            num[i] += B[i + off]
            den[i] += one(eltype(A.den))
        end
    else
        @inbounds for i in R
            j = i + off
            w = weights[j]
            num[i] += w*B[j]
            den[i] += w
        end
    end
    return A
end
