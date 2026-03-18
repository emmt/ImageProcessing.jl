module ImageProcessing

export
    AbstractPoint,
    BoundingBox,
    BoundingBoxLike,
    ImagePatch,
    Interval,
    IntervalLike,
    OnlineSum,
    Point,
    PointLike,
    center_of_gravity,
    endpoints,
    hard_thresholder,
    nonnegative_part,
    soft_thresholder,
    zerofill!,

    # re-exports from TypeUtils
    nearest,
    new_array,

    # re-exports from InterpolationKernels
    infimum,
    supremum,

    # re-exports from LinearAlgebra
    norm,
    cross,
    dot,

    # Re-exports from Base or compat.
    Returns,

    # re-exports from Statistics
    mean

using Backport
@backport

using EasyRanges
using InterpolationKernels
using LinearAlgebra
using Neutrals
using OffsetArrays
using Reexport
using Statistics
using StructuredArrays
using TypeUtils

using EasyRanges: ranges
using Base: @propagate_inbounds, Fix1, Fix2
import InterpolationKernels: infimum, supremum
import TypeUtils: nearest, new_array

include("compat.jl")
include("types.jl")
include("utils.jl")
include("intervals.jl")
include("points.jl")
include("boxes.jl")
include("arithmetic.jl")
include("patches.jl")
include("centroiding.jl")
include("onlinesum.jl")

include("LinearLeastSquares.jl")
@reexport import .LinearLeastSquares:
    LinearLeastSquares,
    AbstractNormalEquations,
    StaticNormalEquations,
    MutableNormalEquations,
    NormalEquations,
    rhs_vector,
    rhs_vector!,
    lhs_matrix,
    lhs_matrix!,
    solve,
    solve!,
    update,
    update!

end # module
