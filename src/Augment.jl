using PowerModels
using MLOPF

"Returns the total load (specified component), repeated num generator times."
function augmented_parameter(pm::ACPPowerModel, bus::String, parameter::MLOPF.LoadParameter)
    load = map(id -> pm.data["load"]["$id"][parameter.key], MLOPF.reference(pm, :bus_loads)[parse(Int, bus)])
    gens = MLOPF.reference(pm, :bus_gens)[parse(Int, bus)]
    return fill(sum(load, init=0.0), max(1, length(gens)))
end

"Returns the normalised bus parameter, repeated num generator times."
function augmented_parameter(pm::ACPPowerModel, bus::String, parameter::MLOPF.BusParameter)
    gens = MLOPF.reference(pm, :bus_gens)[parse(Int, bus)]
    return fill(pm.solution["bus"][bus][parameter.key], max(1, length(gens)))
end

"Returns the normalised power (specified component) for each generator on the bus."
function augmented_parameter(pm::ACPPowerModel, bus::String, parameter::MLOPF.GenParameter)
    gens = MLOPF.reference(pm, :bus_gens)[parse(Int, bus)]
    return isempty(gens) ? [NaN] : map(gens) do gen
        pm.solution["gen"]["$gen"][parameter.key] 
    end
end

"Repeat specified rows and columns num generator times."
function augment_adjacency_matrix(pm::ACPPowerModel, adj_mat::Matrix, bus_name::String, bus_id::Int64)
    gens = MLOPF.reference(pm, :bus_gens)[parse(Int, bus_name)]
    adj_mat = vcat(adj_mat[1:(bus_id-1), :], repeat(adj_mat[bus_id, :]', max(1, length(gens))), adj_mat[(bus_id+1):end, :])
    adj_mat = hcat(adj_mat[:, 1:(bus_id-1)], repeat(adj_mat[:, bus_id]', max(1, length(gens)))', adj_mat[:, (bus_id+1):end])
    return adj_mat
end
