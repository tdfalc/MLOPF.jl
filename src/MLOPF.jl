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

struct NetworkParameter
    key::String
    min::String
    max::String
end

struct GenParameter <: NetworkParameter end

const pg = GenParameter("pg", "pmin", "pmax")
const qg = GenParameter("qg", "qmin", "qmax")

struct LoadParameter <: NetworkParameter end

const pd = LoadParameter("pd", "pdmin", "pdmax")
const qd = LoadParameter("qd", "qdmin", "qdmax")

struct BusParameter <: NetworkParameter end

const vm = BusParameter("vm", "vmin", "vmax")
const du = BusParameter("lam_kcl_r", "", "")

end
