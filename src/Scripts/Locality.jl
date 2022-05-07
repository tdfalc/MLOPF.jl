"""Analyse locality between grid parameters (inputs) generator set-points (outputs).

Notes:
    * Starting with julia -p n provides n worker processes on local machine.
"""

using Distributed
using LightGraphs

@everywhere begin
    using Pkg
    Pkg.activate(".")
end

@everywhere begin
    using MLOPF
end

abstract type NetworkParameter end

struct GenParameter <: Parameter
    id::String
end

struct BusParameter <: Parameter
    id::String
end

const vm = BusParameter("vm")
const du = BusParameter("lam_kcl_r")
const pg = GenParameter("pg")

function main()
    settings = MLOPF.get_settings()
    network = PowerModels.parse_file(ENV["HOME"] * settings.PGLIB_OPF.path * "$(case).m")
    MLOPF.truncate!(network)
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
    bus_index_map = get_bus_lookup_map(pm)
    return [
        length(bus_gens[bus]) > 0 ? solution["bus"]["$bus"][var.id] : 0 for
        (bus, _) in sort(bus_index_map; byvalue = true)
    ]
end

"Extract generator variable (aggregated by bus) from solution dictionary."
function get_parameter(solution::Dict{String,Any}, var::T, pm::ACPPowerModel) where {T<:GenParameter}
    bus_index_map, bus_gen_map = get_bus_lookup_map(pm), MLOPF.get_reference(pm, :bus_gens)
    return [
        sum(map(gen -> solution["gen"]["$gen"][var.id], bus_gen_map[bus]), init = 0) for
        (bus, _) in sort(bus_index_map; byvalue = true)
    ]
end

""
function calculate_diffs(pm::ACPPowerModel, shortest_paths::Matrix{Int64}, solutions::Dict{String,Any}...)
    diff(x::Vector, y::Vector) = 100 * @. abs(x - y) / x
    diff(var::T) where {T<:NetworkParameter} = diff(map(x -> get_parameter(x, var, pm), solutions)...)
    Δvm, Δpg, Δdu = diff(pg), diff(vm), diff(du)
    diffs = Dict{String,Dict}()
    for order ∈ 1:maximum(shortest_paths)
        order_avg(ps) = mean(filter(!isnan, vcat(map(x -> ps[x], eachcol(shortest_paths .== order))...)))
        add!(key, value) = push!(get!(diffs, key, Dict()), order => value)
        add!(vm.id, order_avg(Δvm))
        add!(pg.id, order_avg(Δpg))
        add!(du.id, order_avg(Δdu))
    end
    return deltas
end

