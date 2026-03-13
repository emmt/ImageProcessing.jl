module CentroidingBenchmarks
using ImageProcessing, LinearAlgebra, BenchmarkTools

typeof_sum_f_A(f::Function, A::AbstractArray) =
    typeof(1*f(zero(eltype(A))))

typeof_sum_f_A_B(f::Function, A::AbstractArray, B::AbstractArray) =
    typeof(1*f(zero(eltype(A))*zero(eltype(B))))

#---------------------------------------------------- Reference center of gravity, no SIMD -

function center_of_gravity_ref(f::Function, A::AbstractArray{<:Any,1})
    sw = sw1 = zero(typeof_sum_f_A(f, A))
    I1 = axes(A)[1]
    @inbounds for i1 in I1
        w = f(A[i1])
        sw1 += w*i1
        sw += w
    end
    return Point(sw1/sw)
end

function center_of_gravity_ref(f::Function, A::AbstractArray{<:Any,1},
                               B::AbstractArray{<:Any,1})
    @assert axes(A) == axes(B)
    sw = sw1 = zero(typeof_sum_f_A_B(f, A, B))
    I1 = axes(A)[1]
    @inbounds for i1 in I1
        w = f(A[i1], B[i1])
        sw1 += w*i1
        sw += w
    end
    return Point(sw1/sw)
end

function center_of_gravity_ref(f::Function, A::AbstractArray{<:Any,2})
    sw = sw1 = sw2 = zero(typeof_sum_f_A(f, A))
    I1, I2 = axes(A)
    @inbounds for i2 in I2
        for i1 in I1
            w = f(A[i1,i2])
            sw1 += w*i1
            sw2 += w*i2
            sw += w
        end
    end
    return Point(sw1/sw, sw2/sw)
end

function center_of_gravity_ref(f::Function, A::AbstractArray{<:Any,2},
                                  B::AbstractArray{<:Any,2})
    @assert axes(A) == axes(B)
    sw = sw1 = sw2 = zero(typeof_sum_f_A_B(f, A, B))
    I1, I2 = axes(A)
    @inbounds for i2 in I2
        for i1 in I1
            w = f(A[i1,i2], B[i1,i2])
            sw1 += w*i1
            sw2 += w*i2
            sw += w
        end
    end
    return Point(sw1/sw, sw2/sw)
end

function center_of_gravity_ref(f::Function, A::AbstractArray{<:Any,3})
    sw = sw1 = sw2 = sw3 = zero(typeof_sum_f_A(f, A))
    I1, I2, I3 = axes(A)
    @inbounds for i3 in I3
        for i2 in I2
            for i1 in I1
                w = f(A[i1,i2,i3])
                sw1 += w*i1
                sw2 += w*i2
                sw3 += w*i3
                sw += w
            end
        end
    end
    return Point(sw1/sw, sw2/sw, sw3/sw)
end

function center_of_gravity_ref(f::Function, A::AbstractArray{<:Any,3},
                               B::AbstractArray{<:Any,3})
    @assert axes(A) == axes(B)
    sw = sw1 = sw2 = sw3 = zero(typeof_sum_f_A_B(f, A, B))
    I1, I2, I3 = axes(A)
    @inbounds for i3 in I3
        for i2 in I2
            for i1 in I1
                w = f(A[i1,i2,i3], B[i1,i2,i3])
                sw1 += w*i1
                sw2 += w*i2
                sw3 += w*i3
                sw += w
            end
        end
    end
    return Point(sw1/sw, sw2/sw, sw3/sw)
end

#-------------------------------------------------- Reference center of gravity, with SIMD -

function center_of_gravity_simd(f::Function, A::AbstractArray{<:Any,1})
    sw = sw1 = zero(typeof_sum_f_A(f, A))
    I1 = axes(A)[1]
    @inbounds @simd for i1 in I1
        w = f(A[i1])
        sw1 += w*i1
        sw += w
    end
    return Point(sw1/sw)
end

function center_of_gravity_simd(f::Function, A::AbstractArray{<:Any,1},
                               B::AbstractArray{<:Any,1})
    @assert axes(A) == axes(B)
    sw = sw1 = zero(typeof_sum_f_A_B(f, A, B))
    I1 = axes(A)[1]
    @inbounds @simd for i1 in I1
        w = f(A[i1], B[i1])
        sw1 += w*i1
        sw += w
    end
    return Point(sw1/sw)
end

function center_of_gravity_simd(f::Function, A::AbstractArray{<:Any,2})
    sw = sw1 = sw2 = zero(typeof_sum_f_A(f, A))
    I1, I2 = axes(A)
    @inbounds for i2 in I2
        @simd for i1 in I1
            w = f(A[i1,i2])
            sw1 += w*i1
            sw2 += w*i2
            sw += w
        end
    end
    return Point(sw1/sw, sw2/sw)
end

function center_of_gravity_simd(f::Function, A::AbstractArray{<:Any,2},
                                B::AbstractArray{<:Any,2})
    @assert axes(A) == axes(B)
    sw = sw1 = sw2 = zero(typeof_sum_f_A_B(f, A, B))
    I1, I2 = axes(A)
    @inbounds for i2 in I2
        @simd for i1 in I1
            w = f(A[i1,i2], B[i1,i2])
            sw1 += w*i1
            sw2 += w*i2
            sw += w
        end
    end
    return Point(sw1/sw, sw2/sw)
end

function center_of_gravity_simd(f::Function, A::AbstractArray{<:Any,3})
    sw = sw1 = sw2 = sw3 = zero(typeof_sum_f_A(f, A))
    I1, I2, I3 = axes(A)
    @inbounds for i3 in I3
        for i2 in I2
            @simd for i1 in I1
                w = f(A[i1,i2,i3])
                sw1 += w*i1
                sw2 += w*i2
                sw3 += w*i3
                sw += w
            end
        end
    end
    return Point(sw1/sw, sw2/sw, sw3/sw)
end

function center_of_gravity_simd(f::Function, A::AbstractArray{<:Any,3},
                                B::AbstractArray{<:Any,3})
    @assert axes(A) == axes(B)
    sw = sw1 = sw2 = sw3 = zero(typeof_sum_f_A_B(f, A, B))
    I1, I2, I3 = axes(A)
    @inbounds for i3 in I3
        for i2 in I2
            @simd for i1 in I1
                w = f(A[i1,i2,i3], B[i1,i2,i3])
                sw1 += w*i1
                sw2 += w*i2
                sw3 += w*i3
                sw += w
            end
        end
    end
    return Point(sw1/sw, sw2/sw, sw3/sw)
end

#-------------------------------------------------------------------------------------------

Base.isapprox(a::Point{<:Any,N}, b::Point{<:Any,N}; kwds...) where {N} =
    Base.isapprox(collect(a), collect(b); kwds...)

function runtests(; T::Type=Float32, shape=(100,100))
    A = rand(T, shape)
    cog1 = center_of_gravity_ref(identity, A)
    cog2 = center_of_gravity(identity, A)
    @assert cog1 ≈ cog2
    nops = (1 + 2*ndims(A))*length(A)
    println("center of gravity with shape=$shape, T=$T, f=identity, nops=$nops")
    print("- ref."); @btime center_of_gravity_ref(identity, $A);
    print("- SIMD"); @btime center_of_gravity_simd(identity, $A);
    print("-     "); @btime center_of_gravity(identity, $A);
    println()
    cog1 = center_of_gravity_ref(nonnegative_part, A)
    cog2 = center_of_gravity(nonnegative_part, A)
    @assert cog1 ≈ cog2
    nops = (2 + 2*ndims(A))*length(A)
    println("center of gravity with shape=$shape, T=$T, f=nonnegative_part, nops=$nops")
    print("- ref."); @btime center_of_gravity_ref(nonnegative_part, $A);
    print("- SIMD"); @btime center_of_gravity_simd(nonnegative_part, $A);
    print("-     "); @btime center_of_gravity(nonnegative_part, $A);
    nothing
end

end # module
