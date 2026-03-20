"""
    Parabola2D(c1, c2, c3, c4, c5, c6) -> P
    Parabola2D{T}(c1, c2, c3, c4, c5, c6) -> P

Return the 2-dimensional parabola whose equation writes:

    P(x,y) = c1 + c2*x + c3*y + c4*x^2 + c5*x*y + c6*y^2

which is also the result of calling `P(x,y)`.

Optional type parameter `T` is the type of the stored coefficients.

The 6-tuple of coefficients can be retrieved by `P.coefs` or by `Tuple(P)`.

"""
Parabola2D

(P::Parabola2D)(pnt::Point{2}) = P(Tuple(pnt))
(P::Parabola2D)((x, y)::NTuple{2,T}) where {T} = P(x, y)
function (P::Parabola2D)(x, y)
    c1, c2, c3, c4, c5, c6 = P.coefs
    return c1 + (c2 + c4*x)*x + (c3 + c6*y)*y + c5*x*y
end

"""
    IsotropicParabola2D(c1, c2, c3, c4) -> P
    IsotropicParabola2D{T}(c1, c2, c3, c4) -> P

Return the 2-dimensional isotropic parabola whose equation writes:

    P(x,y) = c1 + c2*x + c3*y + c4*x^2 + c4*y^2

which is also the result of calling `P(x,y)`.

Optional type parameter `T` is the type of the stored coefficients.

The 4-tuple of coefficients can be retrieved by `P.coefs` or by `Tuple(P)`.

""" IsotropicParabola2D

(P::IsotropicParabola2D)(pnt::Point{2}) = P(Tuple(pnt))
(P::IsotropicParabola2D)((x, y)::NTuple{2,T}) where {T} = P(x, y)
function (P::IsotropicParabola2D)(x, y)
    c1, c2, c3, c4 = P.coefs
    return c1 + (c2 + c4*x)*x + (c3 + c4*y)*y
end

for (type, N) in ((:Parabola2D, 6), (:IsotropicParabola2D, 4))
    @eval begin
        # Accessor.
        Base.Tuple(f::$type) = getfield(f, :coefs)

        # Constructors.
        $type(c::Vararg{Any,$N}) = $type(c)
        $type(c::NTuple{$N,Any}) = $type(promote(c...))
        $type(c::NTuple{$N,T}) where {T} = $type{T}(c)
        $type{T}(c::Vararg{Any,$N}) where {T} = $type{T}(c)

        # Copy/convert constructors.
        $type(x::$type) = x
        $type{T}(x::$type{T}) where {T} = x
        $type{T}(x::$type) where {T} = $type{T}(x.coefs)

        Base.convert(::Type{T}, x::T) where {T<:$type} = x
        Base.convert(::Type{T}, x) where {T<:$type} = T(x)::T

        TypeUtils.get_precision(::Type{$type{T}}) where {T} = get_precision(T)
        TypeUtils.adapt_precision(::Type{T}, f::$type{S}) where {T<:TypeUtils.Precision,S} =
            $type{adapt_precision(T, S)}(f)

        # Broadcasting and mapping of some functions on polynomials.
        Broadcast.broadcasted(::Type{T}, f::$type) where {T} = $type{T}(f)
        Broadcast.broadcasted(::typeof(float), f::$type) = float(f)
        Base.float(f::$type) = $type(map(float, f.coefs))

        # Multiplication and division of a polynomial by a scalar.
        Base.:(*)(x::Number, f::$type) = $type(map(Base.Fix1(*, x), f.coefs))
        Base.:(*)(f::$type, x::Number) = x*f
        Base.:(/)(f::$type, x::Number) = $type(map(Base.Fix2(/, x), f.coefs))
        Base.:(\)(x::Number, f::$type) = f/x

        # Unary plus and minus for polynomials.
        Base.:(+)(f::$type) = f
        Base.:(-)(f::$type) = $type(map(-, f.coefs)) # FIXME unsigned coefs!

        # Addition and subtraction of polynomials.
        Base.:(+)(a::$type, b::$type) = $type(map(+, a.coefs, b.coefs))
        Base.:(-)(a::$type, b::$type) = $type(map(-, a.coefs, b.coefs))

        # f[i] syntax to index the coefficients.
        @propagate_inbounds Base.getindex(A::$type, i) = getindex(Tuple(f), i)

        # Neutral element for addition of polynomials.
        Base.zero(f::$type{T}) where {T} = $type(ntuple(Returns(zero(T)), Val{$N}()))

        # Multiplicative identity for polynomials.
        Base.one(f::$type{T}) where {T} = one(T)
    end
end

Base.show(io::IO, f::AbstractPolynomial) = show(io, MIME"text/plain"(), f)
function Base.show(io::IO, m::MIME"text/plain", f::AbstractPolynomial)
    show(io, m, typeof(f))
    write(io, "(")
    for (i, c) in enumerate(f.coefs)
        i > 1 && write(io, ", ")
        show(io, m, c)
    end
    write(io, ")")
end

"""
    is_strictly_concave(f)

Return whether model function `f` is strictly concave and therefore has a single maximum.

"""
function is_strictly_concave(f::IsotropicParabola2D)
    return isnegative(f[4])
end
function is_strictly_concave(f::Parabola2D)
    c4, c5, c6 = f[4:6]
    return isnegative(c4) & isnegative(c6) & (4*c4*c6 > c5^2)
end

"""
    is_strictly_convex(f)

Return whether model function `f` is strictly convex and therefore has a single minimum.

"""
function is_strictly_convex(f::IsotropicParabola2D)
    return ispositive(f[4])
end
function is_strictly_convex(f::Parabola2D)
    c4, c5, c6 = f[4:6]
    return ispositive(c4) & ispositive(c6) & (4*c4*c6 > c5^2)
end

"""
    eqs = ImageProcessing.IsotropicParabola2DNormalEquations{T}()
    eqs = StaticNormalEquations(IsotropicParabola2D{T})

Return a zero-filled object storing the coefficients of type `T` of the normal equations for
fitting a 2-dimensional isotropic parabola.

Updating `eqs` is done by one of:

```julia
eqs = update(eqs, x, y, z; weight=w)
eqs = update(eqs, w, x, y, z)
```

with `(x,y)` the 2-dimensional coordinate of observed value `z` and `w ≥ 0` an associated
weight. If not specified, `w` is assumed to be one.

"""
function IsotropicParabola2DNormalEquations{T}() where {T}
    return IsotropicParabola2DNormalEquations(StaticNormalEquations{4,T,10}())
end

function LinearLeastSquares.StaticNormalEquations(::Type{<:IsotropicParabola2D{T}}) where {T}
    return IsotropicParabola2DNormalEquations{T}()
end

function LinearLeastSquares.update(eqs::IsotropicParabola2DNormalEquations{T},
                                   x, y, z; weight=ONE) where {T}
    return update(eqs, weight, x, y, z)
end

function LinearLeastSquares.update(eqs::IsotropicParabola2DNormalEquations{T}, w::Real,
                                   x::Real, y::Real, z::Real) where {T<:AbstractFloat}
    # x, y, and z must be of type T, w is converted later if needed
    return update(eqs, w, convert(T, x), convert(T, z), convert(T, z))
end

function LinearLeastSquares.update(eqs::IsotropicParabola2DNormalEquations{T}, w::Real,
                                   x::T, y::T, z::T) where {T<:AbstractFloat}
    return IsotropicParabola2DNormalEquations{T}(
        # isotropic parabola model writes: c₁ + c₂*x + c₃*y + c₄*(x^2 + y^2)
        update(eqs.parent, lazy_convert(T, w), z, (ONE, x, y, x^2 + y^2)))
end

function LinearLeastSquares.lhs_matrix(eqs::IsotropicParabola2DNormalEquations{T}) where {T}
    return lhs_matrix(eqs.parent)
end

function LinearLeastSquares.rhs_vector(eqs::IsotropicParabola2DNormalEquations{T}) where {T}
    return rhs_vector(eqs.parent)
end

function LinearLeastSquares.solve(eqs::IsotropicParabola2DNormalEquations{T}) where {T}
    return IsotropicParabola2D{T}(Tuple(solve(eqs.parent)))
end

"""
    eqs = ImageProcessing.Parabola2DNormalEquations{T}()
    eqs = StaticNormalEquations(Parabola2D{T})

Return a zero-filled object storing the coefficients of type `T` of the normal equations for
fitting a 2-dimensional parabola.

Updating `eqs` is done by one of:

```julia
eqs = update(eqs, x, y, z; weight=w)
eqs = update(eqs, w, x, y, z)
```

with `(x,y)` the 2-dimensional coordinate of observed value `z` and `w ≥ 0` an associated
weight. If not specified, `w` is assumed to be one.

"""
function Parabola2DNormalEquations{T}() where {T}
    z = zero(T)
    return Parabola2DNormalEquations{T}(z, z, z, z, z, z, z,
                                        z, z, z, z, z, z, z,
                                        z, z, z, z, z, z, z)
end

function LinearLeastSquares.StaticNormalEquations(::Type{<:Parabola2D{T}}) where {T}
    return Parabola2DNormalEquations{T}()
end

function LinearLeastSquares.update(eqs::Parabola2DNormalEquations{T},
                                   x, y, z; weight=ONE) where {T}
    return update(eqs, weight, x, y, z)
end

function LinearLeastSquares.update(eqs::Parabola2DNormalEquations{T},
                                   w, x, y, z) where {T}
    return update(eqs, lazy_convert(T, w), convert(T, x), convert(T, z), convert(T, z))
end

function LinearLeastSquares.update(eqs::Parabola2DNormalEquations{T},
                                   w::LinearLeastSquares.Weight{T},
                                   x::T, y::T, z::T) where {T}
    (w > zero(w) && isfinite(z)) || return eqs
    wx = w*x
    wy = w*y
    wz = w*convert(T, z)
    x² = x*x
    xy = x*y
    y² = y*y
    wx² = w*x²
    wxy = w*xy
    wy² = w*y²
    return Parabola2DNormalEquations{T}(
        # Integrate coefficients of LHS matrix A.
        eqs.sw     + w,
        eqs.swx    + wx,
        eqs.swxy   + wxy,
        eqs.swxy²  + wxy*y,
        eqs.swxy³  + wxy*y²,
        eqs.swx²   + wx²,
        eqs.swx²y  + wx²*y,
        eqs.swx²y² + wx²*y²,
        eqs.swx³   + wx²*x,
        eqs.swx³y  + wx²*xy,
        eqs.swx⁴   + wx²*x²,
        eqs.swy    + wy,
        eqs.swy²   + wy²,
        eqs.swy³   + wy²*y,
        eqs.swy⁴   + wy²*y²,
        # Integrate coefficients of RHS vector b.
        eqs.swz    + wz,
        eqs.swzx   + wz*x,
        eqs.swzy   + wz*y,
        eqs.swzx²  + wz*x²,
        eqs.swzxy  + wz*xy,
        eqs.swzy²  + wz*y²)
end

function LinearLeastSquares.lhs_matrix(eqs::Parabola2DNormalEquations{T}) where {T}
    return SMatrix{6,6,T,36}(
        #=          1         x          y           x²          xy          y²    =#
        #= 1  =# eqs.sw,   eqs.swx,   eqs.swy,    eqs.swx²,   eqs.swxy,   eqs.swy²,
        #= x  =# eqs.swx,  eqs.swx²,  eqs.swxy,   eqs.swx³,   eqs.swx²y,  eqs.swxy²,
        #= y  =# eqs.swy,  eqs.swxy,  eqs.swy²,   eqs.swx²y,  eqs.swxy²,  eqs.swy³,
        #= x² =# eqs.swx², eqs.swx³,  eqs.swx²y,  eqs.swx⁴,   eqs.swx³y,  eqs.swx²y²,
        #= xy =# eqs.swxy, eqs.swx²y, eqs.swxy²,  eqs.swx³y,  eqs.swx²y², eqs.swxy³,
        #= y² =# eqs.swy², eqs.swxy², eqs.swy³,   eqs.swx²y², eqs.swxy³,  eqs.swy⁴,
    )
end

function LinearLeastSquares.rhs_vector(eqs::Parabola2DNormalEquations{T}) where {T}
    return SVector{6,T}(
        #=    =# eqs.swz,  eqs.swzx,  eqs.swzy,   eqs.swzx²,  eqs.swzxy,  eqs.swzy²,
    )
end

function LinearLeastSquares.solve(eqs::Parabola2DNormalEquations{T}) where {T}
    return Parabola2D{T}(Tuple(cholesky(lhs_matrix(eqs))\rhs_vector(eqs)))
end

"""
    StationaryPoint2D{T}(origin, point, value, eigvals, angle)

Return an image feature descriptor representing a 2-dimensional stationary point.

`T` is the floating-point type of the values stored by the feature descriptor. If omitted,
it is inferred from the arguments:

- `point` is the position of the stationary point relative to `origin`.

- `value` is the model value at the stationary point.

- `eigvals` is the 2-tuple of the eigenvalues of the Hessian matrix of the model shape at
  the stationary point.

- `angle` is the angle, with respect to the first Cartesian axis of the direction, of the
  largest (in absolute value) eigenvalue.

Arguments may also be specified by keywords. In that case, `origin=Point(0,0)` by default
`eigvals=(NaN,NaN)` by default, and `angle=NaN` by default.

"""
function StationaryPoint2D(origin::PointLike{2}, point::PointLike{2}, value::Real,
                           eigvals::NTuple{2,Real}, angle::Real)
    # Convert position arguments to points.
    return StationaryPoint2D(Point(origin), Point(point), value, eigvals, angle)
end
function StationaryPoint2D(origin::Point{2}, point::Point{2}, value::Real,
                           eigvals::NTuple{2,Real}, angle::Real)
    T = float(promote_type(
        eltype(origin), eltype(point), typeof(value), eltype(eigvals), typeof(angle)))
    isconcretetype(T) || throw(ArgumentError(
        "types of values cannot be converted to a common concrete type"))
    return StationaryPoint2D{T}(origin, point, value, eigvals, angle)
end

function StationaryPoint2D{T}(origin::PointLike{2}, point::PointLike{2}, value::Real,
                              eigvals::NTuple{2,Real}, angle::Real) where {T}
    return StationaryPoint2D{T}(Point(origin), Point(point), value, eigvals, angle)
end

# Keyword-only constructors.
function StationaryPoint2D(; origin::PointLike{2}=Point(false,false),
                           point::PointLike{2}, value::Real,
                           eigvals::NTuple{2,Real}=(NaN,NaN), angle::Real=NaN)
    return StationaryPoint2D(origin, point, value, eigvals, angle)
end
function StationaryPoint2D{T}(; origin::PointLike{2}=Point(false,false),
                              point::PointLike{2}, value::Real,
                              eigvals::NTuple{2,Real}=(NaN,NaN), angle::Real=NaN) where {T}
    return StationaryPoint2D{T}(origin, point, value, eigvals, angle)
end

# Conversion constructors.
StationaryPoint2D(A::StationaryPoint2D) = A
StationaryPoint2D{T}(A::StationaryPoint2D{T}) where {T} = A
StationaryPoint2D{T}(A::StationaryPoint2D) where {T} =
    StationaryPoint2D{T}(A.origin, A.point, A.value, A.eigvals, A.angle)

Base.convert(::Type{T}, x::T) where {T<:StationaryPoint2D} = x
Base.convert(::Type{T}, x::StationaryPoint2D) where {T<:StationaryPoint2D} = T(x)::T

"""
    StationaryPoint2D{T}(origin, f)

Return the stationary point of the 2-dimensional parabola `f`. `(xs,ys)` are the coordinates
of the stationary point, `f(xs,ys)` is the corresponding value of the parabola, and `(λₘᵢₙ,
λₘₐₓ)` are the eigenvalues of the Hessian matrix of the parabola, and `θ` the angle with
respect to the first Cartesian axis of the direction of largest (in absolute value)
eigenvalue. If `f` has no unique stationary point, `(xs,ys)` and `f(xs,ys)` may have
non-finite values.

"""
function StationaryPoint2D(origin::PointLike{2},
                           f::Union{Parabola2D,IsotropicParabola2D})
    return StationaryPoint2D(Point(origin), f)
end

function StationaryPoint2D(origin::Point{2,R},
                           f::Union{Parabola2D{S},IsotropicParabola2D{S}}) where {R,S}
    T = float(promote_type(R, S))
    return StationaryPoint2D{T}(origin, f)
end

function StationaryPoint2D{T}(origin::Point{2}, f::Parabola2D) where {T}
    c2, c3, c4, c5, c6 = map(as(T), f[2:6])

    # Position of stationary point.
    delta = 4*c4*c6 - c5^2
    xs = (c3*c5 - 2*c2*c6)/delta
    ys = (c2*c5 - 2*c3*c4)/delta

    # Eigenvalues of the Hessian matrix.
    beta = c4 + c6
    gamma = sqrt((c4 - c6)^2 + c5^2) # sqrt(beta^2 - delta)
    if beta > zero(beta)
        λₘₐₓ = beta + gamma
        λₘᵢₙ = delta/λₘₐₓ
    elseif iszero(beta)
        λₘᵢₙ = -gamma
        λₘₐₓ = +gamma
    else # beta is negative or NaN
        λₘᵢₙ = beta - gamma
        λₘₐₓ = delta/λₘᵢₙ
    end
    angle = atan(c5, c6 - c4)/2
    return StationaryPoint2D{T}(origin, Point(xs, ys), f(xs, ys), (λₘᵢₙ, λₘₐₓ), angle)
end

function StationaryPoint2D{T}(origin::Point{2}, f::IsotropicParabola2D) where {T}
    c2, c3, c4 = map(as(T), f[2:4])
    λ = 2*c4 # eigenvalue of Hessian matrix
    x = -c2/λ
    y = -c3/λ
    return StationaryPoint2D{T}(origin, Point(x, y), f(x, y), (λ, λ), zero(T))
end
