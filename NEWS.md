# User visible changes in `ImageProcessing` package

- `oneunit(x)` for a point (or a point type) `x` yields a unitful unit step point.
- Bug fix: `isless(x, y)` for points `x` and `y` compares coordinates in reverse order
  (from last to first) as for Cartesian indices.
- Non-exported public methods `ImageProcessing.front` and `ImageProcessing.tail` to
  discard the last of the first element of a tuple.

# Version 0.2.0

- New exported function `new_array(T, args...)` to build an array with undefined elements
  of type `T` and shape `args...` that can be any number of array dimensions (integers)
  and/or axes (integer-valued unit ranges). If all shape parameters are integers or
  instances of `Base.OneTo`, an ordinary array of type `Array{T}` is returned; otherwise,
  an offset array (wrapped on top of an ordinary array) is returned.

- Non-exported public functions `ImageProcessing.to_dim`, `ImageProcessing.to_axis`,
  `ImageProcessing.to_size`, and `ImageProcessing.to_axes` to deal with array shapes.

- Extend some math functions for instances of `Point`:
  - `abs(A)`, `norm(A)`, `norm(A, 2)`, and `hypot(A)` yield the Euclidean norm of the
    vector of coordinates of point `A` while `norm(A, p)` yields the `p`-norm of the
    vector of coordinates of point `A`;
  - `min(A, B)`, `max(A, B)`, and `minmax(A, B)` work for points `A` and `B` as for
    Cartesian indices;
  - `dot(A, B)` yields the scalar product of the vectors of coordinates of points `A` and `B`;
  - `atan(A)` yields the polar angle of 2-dimensional point `A`;
  - `cross(A, B)` and `A × B` yield the cross product of the vectors of coordinates of
    2-dimensional points `A` and `B`.

# Version 0.1.0

- New `OnlineSum` structure to store the sum of, possibly weighted and/or recentered,
  images.
