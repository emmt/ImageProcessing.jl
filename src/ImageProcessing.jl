module ImageProcessing

export
    IndexBox,
    Point,
    nearest

using TypeUtils
using OffsetArrays

include("types.jl")
include("utils.jl")
include("points.jl")
include("boxes.jl")

end # module
