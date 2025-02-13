module ImageProcessing

export
    AbstractPoint,
    BoundingBox,
    BoundingBoxLike,
    Interval,
    IntervalLike,
    OnlineSum,
    Point,
    PointLike,
    center_of_gravity,
    endpoints,
    hard_thresholder,
    nearest,
    new_array,
    nonnegative_part,
    soft_thresholder,
    zerofill!,

    # re-exports from InterpolationKernels
    infimum,
    supremum,

    # re-exports from LinearAlgebra
    norm,
    cross,
    dot,

    # re-exports from Statistics
    mean

using EasyRanges
using InterpolationKernels
using LinearAlgebra
using OffsetArrays
using Statistics
using StructuredArrays
using TypeUtils

using EasyRanges: ranges
using Base: @propagate_inbounds, Fix1, Fix2
import InterpolationKernels: infimum, supremum

include("macros.jl")
include("compat.jl")
include("types.jl")
include("utils.jl")
include("intervals.jl")
include("points.jl")
include("boxes.jl")
include("arithmetic.jl")
include("centroiding.jl")
include("onlinesum.jl")

end # module
