# Wish list

- Array interpolation.

- Add sub-pixel recentering in `OnlineSum` via a given (separable) interpolator.

- Add forgetting factor in `OnlineSum`.

- Deriche filters.

- Find fastest implementation of `min`, `max`, `minmax`, `isless`, `<`, `<=`, etc. for
  points. Maybe have a look at implementation for Cartesian indices.

- Equivalence between points and static vectors.

- Extend `r = Base.broadcasted(op, x, y)` and `Base.materialize(r)` to map things like
  `box + point` or `interval + value`.
