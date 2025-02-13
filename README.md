# Image processing for Julia

[![Build Status](https://github.com/emmt/ImageProcessing.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/emmt/ImageProcessing.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Build Status](https://ci.appveyor.com/api/projects/status/github/emmt/ImageProcessing.jl?svg=true)](https://ci.appveyor.com/project/emmt/ImageProcessing-jl) [![Coverage](https://codecov.io/gh/emmt/ImageProcessing.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/emmt/ImageProcessing.jl)

The `ImageProcessing` package provides methods and types for processing *images* in
[Julia](https://julialang.org/). Following the conventions in
[`JuliaImages`](https://juliaimages.org), *images* can be any multi-dimensional abstract
arrays with numerical values, not just 2-dimensional arrays.

## Remarks

*Images* may be quite large arrays so many methods of this package are designed for fast
computations. For example, branching in loops is avoided to favor loop vectorization. As a
consequence, special values such as `NaN` or `missing` are not treated specifically. Even
more, `missing` is not an expected value in images. One should use `NaN` to indicate
missing or bad data and rely on IEEE rules for NaNs to produce correct results or, better,
use zero weights for indicating missing or bad data and avoid NaNs.

## Points and bounding-boxes

The `ImageProcessing` provides points and bounding-boxes of respective types `Point{N,T}`
and `BoundingBox{N,T}` with `N` the dimensionality and `T` the type of coordinates. These
object can be seen as generalizations of `CartesianIndex{N}` and `CartesianIndices{N}` to
help working with coordinates and help defining coordinate transforms, region of interest,
etc. With points and bounding boxes, coordinate type `T` may not be integer and a number
of arithmetic operations are supported that may not be implemented for Cartesian indices
(addition or subtraction of points, scaling of point by a scalar, rounding, etc.).

As a facility, even tough points may have continuous coordinates, they may be converted to
`CartesianIndex` which represents discrete positions. For a point, say `pnt`, with integer
coordinates, it is sufficient to call the constructor `CartesianIndex(pnt)`. For
non-integer coordinates, the coordinates must first be rounded (in some direction) to an
integer, for example by one of:

``` julia
nearest(Point{N,Int}, pnt) # round coordinates to nearest `Int`
round(Point{N,Int}, pnt)   # round coordinates to nearest `Int`
floor(Point{N,Int}, pnt)   # round coordinates to nearest `Int` from below
ceil(Point{N,Int}, pnt)    # round coordinates to nearest `Int` from above
```

where `N` is the number of dimensions and then call `CartesianIndex` on the result. To
simplify such conversion and make the code more readable, it is sufficient to call:

``` julia
nearest(CartesianIndex, pnt) # round point to nearest Cartesian index
round(CartesianIndex, pnt) # round point to nearest Cartesian index
floor(CartesianIndex, pnt) # round point to nearest Cartesian index from below
ceil(CartesianIndex, pnt) # round point to nearest Cartesian index from above
```

Similarly, even tough intervals and bounding-boxes represent continuous ranges, they may
be respectively converted to `AbstractRange` or `CartesianIndices` instances which
represent discrete ranges.

Operators `∈` (`in`), `⊆` (`issubset`), and `∩` (`intersect`) may be used with points,
intervals, and bounding-boxes. Integer-valued points, intervals, and bounding-boxes my
also be tested with these operators against `CartesianIndex`, `AbstractRange{<:Integer}`,
and `CartesianIndices` provided the 2 latters have unit-step. The operation will be
performed as if the point, interval, or bounding-box has been converted to its discrete
counterpart.

## Installation

To install `ImageProcessing` so as to follow the main development branch:

``` julia
using Pkg
Pkg.add(url="https://github.com/emmt/ImageProcessing.jl")
```

or at the prompt of Julia's package manager (after typing `]` in Julia's REPL):

``` julia
add https://github.com/emmt/ImageProcessing.jl
```

Another possibility is to install `ImageProcessing` via Julia registry
[`EmmtRegistry`](https://github.com/emmt/EmmtRegistry), from the prompt of Julia's package
manager:

```julia
registry add General
registry add https://github.com/emmt/EmmtRegistry
add ImageProcessing
```

Adding the `General` registry (1st line of the above example) is mandatory to have access
to the official Julia packages if you never have used the package manager before.
