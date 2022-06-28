module MLOPF

abstract type NetworkParameter end

struct GenParameter <: NetworkParameter
    key::String
    min::String
    max::String
end

const pg = GenParameter("pg", "pmin", "pmax")
const qg = GenParameter("qg", "qmin", "qmax")

struct LoadParameter <: NetworkParameter
    key::String
    min::String
    max::String
end

const pd = LoadParameter("pd", "pdmin", "pdmax")
const qd = LoadParameter("qd", "qdmin", "qdmax")

struct BusParameter <: NetworkParameter
    key::String
    min::String
    max::String
end

const vm = BusParameter("vm", "vmin", "vmax")
const du = BusParameter("lam_kcl_r", "", "")

include("./Sample.jl")
include("./Augment.jl")
include("./Normalise.jl")
include("./Models/Generic.jl")
include("./Models/FullyConnected.jl")
include("./Models/Convolutional.jl")
include("./Models/Graph.jl")
include("./Constraints.jl")
include("./Truncate.jl")
include("./Cache.jl")
include("./Settings.jl")

end
