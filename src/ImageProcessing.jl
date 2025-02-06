module ImageProcessing

export
    IndexBox,
    Point,
    OnlineSum,
    center_of_gravity,
    hard_thresholder,
    nearest,
    nonnegative_part,
    soft_thresholder,
    zerofill!,

    # re-exports from Statistics
    mean

using EasyRanges
using LinearAlgebra
using OffsetArrays
using Statistics
using StructuredArrays
using TypeUtils

include("macros.jl")
include("compat.jl")
include("types.jl")
include("utils.jl")
include("points.jl")
include("boxes.jl")
include("centroiding.jl")
include("onlinesum.jl")

end # module
