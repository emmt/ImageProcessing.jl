# User visible changes in `ImageProcessing` package

- Extend some math functions for instances of `Point`:
  - `abs(A)`, `norm2(A)`, `norm(A, 2)`, and `hypot(A)` yield the Euclidean norm of the
    vector of coordinates of point `A`;
  - `norm(A, p)` yield the `p`-norm of the vector of coordinates of point `A`;
  - `min(A,B)`, `max(A,B)`, and `minmax(A,B)` work for points `A` and `B` as for Cartesian
    indices;
  - `atan(A)` yields the polar angle of 2-dimensional point `A`;
  - `dot(A,B)` yields the scalar product of the vectors of coordinates of points `A` and `B`;
  - `cross(A,B)` and `A × B` yield the cross product of the vectors of coordinates of
    2-dimensional points `A` and `B`.

# Version 0.1.0

- New `OnlineSum` structure to store the sum of, possibly weighted and/or recentered,
  images.
