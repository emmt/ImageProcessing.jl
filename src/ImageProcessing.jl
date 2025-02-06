module ImageProcessing

export
    IndexBox,
    OnlineSum,
    Point,
    center_of_gravity,
    hard_thresholder,
    nearest,
    new_array,
    nonnegative_part,
    soft_thresholder,
    zerofill!,

    # re-exports from LinearAlgebra
    norm,
    cross,
    dot,

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
