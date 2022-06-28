"""Analyse locality between grid parameters (inputs) generator set-points (outputs).

Notes:
    * Starting with julia -p n provides n worker processes on local machine.
"""

using Distributed

@everywhere begin
    using Pkg
    Pkg.activate(".")
end

@everywhere begin
    using MLOPF
    using LightGraphs
end

function main()
    settings = MLOPF.get_settings()
    network = PowerModels.parse_file(ENV["HOME"] * settings.PGLIB_OPF.path * "$(case).m")
    samples = load_samples()
    MLOPF.cache(() -> pmap(x -> run_sample(sample, network), samples[1:num_samples]), "./cache/results/", "$(case).jld2")()
end

function run_sample(sample::Dict, network)
    let network = deepcopy(network)
        set_network_loads!(network, sample)
        before = solve_opf(network)
        for (id, load) ∈ ALL_LOADS
            let network = deepcopy(network)
                perturb_load!(network, load, delta)
                after = solve_opf(network)
                results[id] = calculate_diffs(pm, shortest_paths, before, after)
            end
        end
    end
    return results
end

function aggregate_load(network::Dict{String,Any})
    return sum(load["pd"] for (_, load) in network["load"])
end

function perturb_load!(network::Dict{String,Any}, id::String, delta::Float64)
    network["load"][id]["pd"] *= (1.0 + delta)
end

"Find the shortest path between each pair of nodes in a graph."
function shortest_paths(adj_mat::Matrix{Float64})
    paths = zeros(Int, num_bus, num_bus)
    for i ∈ 1:num_bus
        paths[i, :] = dijkstra_shortest_paths(Graph(adj_mat), i).dists
    end
    return paths
end

"Extract bus variable (for generators) from solution dictionary."
function get_parameter(solution::Dict{String,Any}, var::T, pm::ACPPowerModel) where {T<:BusParameter}
    return [
        length(bus_gens[bus]) > 0 ? solution["bus"]["$bus"][var.id] : 0 for
        (bus, _) in sort(get_bus_lookup_map(pm); byvalue=true)
    ]
end

"Extract generator variable (aggregated by bus) from solution dictionary."
function get_parameter(solution::Dict{String,Any}, var::T, pm::ACPPowerModel) where {T<:GenParameter}
    return [
        sum(map(gen -> solution["gen"]["$gen"][var.id], MLOPF.get_reference(pm, :bus_gens)[bus]), init=0) for
        (bus, _) in sort(get_bus_lookup_map(pm); byvalue=true)
    ]
end

"Returns a dictionary mapping between order and relative change in value for each parameter."
function relative_change(pm::ACPPowerModel, shortest_paths::Matrix{Int64}, solutions::Dict{String,Any}...)
    delta(x::Vector, y::Vector) = 100 * @. abs(x - y) / x
    delta(var::T) where {T<:NetworkParameter} = delta(map(x -> get_parameter(x, var, pm), solutions)...)
    Δvm, Δpg, Δdu = delta(pg), delta(vm), delta(du)
    deltas_by_order = DefaultDict(() -> Dict)
    for order ∈ 1:maximum(shortest_paths)
        order_avg(ps) = mean(filter(!isnan, [([delta_vm[x] for x in eachcol(sp_mat .== order)]...)...]))
        deltas_by_order[vm.id] = order_avg(Δvm)
        deltas_by_order[pg.id] = order_avg(Δpg)
        deltas_by_order[du.id] = order_avg(Δdu)
    end
    return deltas_by_order
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end