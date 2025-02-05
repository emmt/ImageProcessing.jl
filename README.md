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
