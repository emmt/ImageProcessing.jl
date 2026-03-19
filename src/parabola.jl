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
    c2, c3, c4, c5, c6 = map(as(T), Tuple(f)[2:6])

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
    c2, c3, c4 = map(as(T), Tuple(f)[2:4])
    λ = 2*c4 # eigenvalue of Hessian matrix
    x = -c2/λ
    y = -c3/λ
    return StationaryPoint2D{T}(origin, Point(x, y), f(x, y), (λ, λ), zero(T))
end
