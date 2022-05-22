module MLOPF

include("./Sample.jl")
include("./Augment.jl")
#include("./Models/Generic.jl")
#include("./Models/FullyConnected.jl")
#include("./Models/Convolutional.jl")
#include("./Models/Graph.jl")
include("./Constraints.jl")
include("./Truncate.jl")
include("./Cache.jl")
include("./Settings.jl")

abstract type NetworkParameter end

struct GenParameter <: NetworkParameter
    key::String
    min::String
    max::String
end

const pg = GenParameter("pg", "pmin", "pmax")
const qg = GenParameter("qg", "qmin", "qmax")

struct BusParameter <: NetworkParameter
    id::String
    min::String
    max::String
end

const pd = BusParameter("pd", "pdmin", "pdmax")
const qd = BusParameter("qd", "qdmin", "qdmax")
const vm = BusParameter("vm", "vmin", "vmax")
const du = BusParameter("lam_kcl_r", "", "")

end
