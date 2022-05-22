using PowerModels
using MLOPF

"Returns the total load, repeated n (= the number of generators) times."
function augmented_bus_load(pm::ACPPowerModel, bus::String, parameter::MLOPF.BusParameter)
    load = map(id -> pm.data["load"]["$id"][parameter.key], MLOPF.reference(pm, :bus_loads)[parse(Int, bus)])
    gens = MLOPF.reference(pm, :bus_gens)[parse(Int, bus)]
    return fill(sum(load, init=0.0), max(1, length(gens)))
end

"Returns the normalised voltage magnitude, repeated n (= the number of generators) times."
function augmented_bus_vm(pm::ACPPowerModel, bus::String)
    vmin, vmax = getindex.(Ref(pm.data["bus"][bus]), (vm.min, vm.max))
    gens = MLOPF.reference(pm, :bus_gens)[parse(Int, bus)]
    return fill((pm.solution["bus"][bus][vm.key] - vmin) / (vmax - vmin), max(1, length(gens)))
end

"Returns the normalised injected power (specified component) for each generator on the bus."
function augmented_bus_gen(pm::ACPPowerModel, bus::String, parameter::MLOPF.GenParameter)
    gens = MLOPF.reference(pm, :bus_gens)[parse(Int, bus)]
    return isempty(gens) ? [NaN] : map(gens) do gen
        (pm.data["gen"]["$gen"][parameter.key] - pm.data["gen"]["$gen"][parameter.min]) /
        (pm.data["gen"]["$gen"][parameter.max] - pm.data["gen"]["$gen"][parameter.min])
    end
end

"Repeat specified rows and columns n (= the number of generators) times."
function augment_adjacency_matrix(pm::ACPPowerModel, adj_mat::Matrix, bus_name::String, bus_id::Int64)
    gens = MLOPF.reference(pm, :bus_gens)[parse(Int, bus_name)]
    adj_mat = vcat(adj_mat[1:(bus_id-1), :], repeat(adj_mat[bus_id, :]', length(gens)), adj_mat[(bus_id+1):end, :])
    adj_mat = hcat(adj_mat[:, 1:(bus_id-1)], repeat(adj_mat[:, bus_id]', length(gens))', adj_mat[:, (bus_id+1):end])
    return adj_mat
end