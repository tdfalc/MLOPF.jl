function augmented_bus_load(pm::ACPPowerModel, bus::String, key::String)
    load = map(id -> pm.data["load"]["$id"][key], reference(pm, :bus_loads)[parse(Int, bus)])
    gens = reference(pm, :bus_gens)[parse(Int, bus)]
    return fill(sum(load, init=0.0), max(1, length(gens)))
end

function augmented_bus_vm(pm::ACPPowerModel, bus::String)
    vm = pm.solution["bus"][bus]["vm"]
    vmin, vmax = getindex.(Ref(pm.data["bus"][bus]), ("vmin", "vmax"))
    gens = reference(pm, :bus_gens)[parse(Int, bus)]
    return fill((vm - vmin) / (vmax - vmin), max(1, length(gens)))
end

function augmented_bus_gen(pm::ACPPowerModel, bus::String, key::String)
    gens = reference(pm, :bus_gens)[parse(Int, bus)]
    return isempty(gens) ? [NaN] : map(gens) do gen
        min, max = first(key) * "min", first(key) * "max"
        (pm.data["gen"]["$gen"][key] - pm.data["gen"]["$gen"][min]) /
        (pm.data["gen"]["$gen"][max] - pm.data["gen"]["$gen"][min])
    end
end

function augment_adjacency_matrix(pm::ACPPowerModel, adj_mat::Matrix, bus::String, id::Int64)
    gens = MLOPF.reference(pm, :bus_gens)[parse(Int, bus)]
    adj_mat = vcat(adj_mat[1:(id-1), :], repeat(adj_mat[id, :]', length(gens)), adj_mat[(id+1):end, :])
    adj_mat = hcat(adj_mat[:, 1:(id-1)], repeat(adj_mat[:, id]', length(gens))', adj_mat[:, (id+1):end])
    return adj_mat
end
