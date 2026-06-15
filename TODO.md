# Wish list

- Use `Neutrals` package.

- Estimate covariance matrix of center of gravity.

- Array interpolation.

- Add sub-pixel recentering in `OnlineSum` via a given (separable) interpolator.

- Add forgetting factor in `OnlineSum`.

- Deriche filters.

- Find fastest implementation of `min`, `max`, `minmax`, `isless`, `<`, `<=`, etc. for
  points. Maybe have a look at implementation for Cartesian indices.

- Equivalence between points and static vectors.

- Extend `r = Base.broadcasted(op, x, y)` and `Base.materialize(r)` to map things like
  `box + point` or `interval + value`.

- In `lhs_matrix` and `rhs_vector`, keyword `readonly` must be a `Val{Bool}` for
  type-stability. These functions shall return suitable arrays for solving the equations
  (hence dense ones with fast 1-based linear indices).

```julia
struct NormalEquations{T<:AbstractFloat,LHS,RHS}; A::LHS; b::RHS; end
const StaticNormalEquations{N,T,L} = NormalEquations{T,NTuple{T,L},NTuple{T,N}}
const WritableNormalEquations{T,N} = NormalEquations{T,<:AbstractVector{T},<:AbstractArray{T,N}}
```
