using Distributions
using Distributed
using DataStructures
using Ipopt
using JuMP
using Random
using ProgressMeter
using PowerModels

PowerModels.silence()

"""
    generate_samples(network::Dict{String, Any}, num_samples::Int, alpha::Float64; max_iter::Int = 100, nmo::Bool = false)

This function generates a set of feasible samples by re-scaling each active and reactive load component 
    (relative to nominal values) by factors independently drawn from a Uniform distribution.

# Arguments:
- `network::Dict{String, Any}` -- Grid network in PowerModels.jl format.
- `num_samples::Int` -- Number of (feasible) samples to generate.
- `alpha::Float64` -- Uniform distrubtion parameter used to re-scale inputs.

# Keywords:
- `max_iter::Int` -- Maximum number of iterations the IPOPT algorithm should run before declaring infeasiblity. Defaults to 100.
- `nmo::Bool` -- Flag to randomly silence a branch in each sample to emulate N-1 contingency. Defaults to false.

# Outputs
- `Vector{Dict{String,Any}}`: Vector of feasible samples.
"""
function generate_samples(
    network::Dict{String,Any},
    num_samples::Int,
    alpha::Float64;
    max_iter::Int = 100,
    nmo::Bool = false,
)
    @info "generating $(num_samples) samples using $(nprocs()) process(es)"
    return @showprogress pmap(
        id -> generate_sample(deepcopy(network), alpha; id = id, max_iter = max_iter, nmo = nmo),
        1:num_samples,
    )
end

function generate_sample(network::Dict{String,Any}, alpha::Float64; id::Int = 1, max_iter::Int = 100, nmo::Bool = False)
    pd, qd = get_load(network)
    let network = deepcopy(network)
        # Select a random branch to silence (emulating N-1 contingency).
        nmo && silence_random_branch!(network)

        # Sample re-scaling factors from a Uniform distribution (parameterised by alpha).
        distribution = Uniform(1.0 - alpha, 1.0 + alpha)
        set_load!(network, pd .* rand(distribution, length(pd)), qd .* rand(distribution, length(qd)))

        # Solve the new AC-OPF problem and validate feasibility.
        pm = PowerModels.instantiate_model(network, ACPPowerModel, PowerModels.build_opf)
        output = solve_acopf(pm, max_iter)
        if validate_feasibility(output["termination_status"])
            return process_output(network, output, MLOPF.binding_status(pm), id)
        end
    end
    return generate_sample(network, alpha; id=id, max_iter=max_iter, nmo=nmo)
end

""
function process_output(network::Dict{String,Any},  output::Dict{String,Any}, congestion_regime::Dict{String,Any}, id::Int64)
    adj_mat = get_adjacency_matrix(pm)
    # Extract parameters to a dictionary that maps each parameter name to a vector of floats
    # with length equal to the number of generators in the network (this simplifies the
    # construction of input tensors for local graph neural network architectures).
    parameters = DefaultDict()
    for (bus, i) in get_bus_index_map(pm)
        adj_mat = augment_adjacency_matrix(pm, adj_mat, bus, i)
        append!(parameters["vm"], augmented_bus_vm(pm, bus))
        for g in ("pg", "qg")
            append!(parameters[g], augmented_bus_gen(pm, bus), gen) 
        end
        for l in ("pd", "qd")
            append!(parameters[l], augmented_bus_load(pm, bus), load) 
        end
    end
    return Dict(
        "id" => id,
        "adjacency_matrix" => adj_mat,
        "parameters" => parameters,
        "congestion_regime" => MLOPF.enumerate_constraints(congestion_regime),
    )
end

function validate_feasibility(status::MOI.TerminationStatusCode)
    return (status == MOI.LOCALLY_SOLVED) || (status == MOI.OPTIMAL)
end

function solve_acopf(power_model::ACPPowerModel, max_iter::Int)
    optimizer = optimizer_with_attributes(Ipopt.Optimizer, "max_iter" => max_iter, "print_level" => 0)
    return PowerModels.optimize_model!(power_model, optimizer = optimizer)
end

"Proxy for removing random branch from optimization problem whilst preserving topology."
function silence_branch!(network::Dict{String,Any}; br_r::Float64 = 9e9)
    id = string(rand(1:length(network["branch"])))
    setindex!(network["branch"][id], "b_fr", 0.0)
    setindex!(network["branch"][id], "b_to", 0.0)
    setindex!(network["branch"][id], "br_x", 0.0)
    setindex!(network["branch"][id], "br_r", br_r)
end

function get_network_loads(network::Dict{String,Any}, key::String)
    return [load[key] for (_, load) in sort(network["load"]) ]
end

"Convenience function to get both the active and reactive components of bus loads."
function get_network_loads(network::Dict{String,Any})
    return get_network_loads(network, "pd"), get_load(network, "qd")
end

function set_network_loads!(network::Dict{String,Any}, key::String, values::Array{Float64})
    for (i, (_, load)) in enumerate(sort(network["load"]))
        setindex!(load, key, values[i])
    end
end

"Convenience function to set both the active and reactive components of bus loads."
function set_network_loads!(network::Dict{String,Any}, pd::Array{Float64}, qd::Array{Float64})
    set_network_loads!(network, "pd", pd)
    set_network_loads!(network, "qd", qd)
end


"Get (sparse) adjacency matrix using PowerModels API and convert to dense matrix."
function get_adjacency_matrix(pm::ACPPowerModel)
    adj_mat, _ = PowerModels._adjacency_matrix(pm)
    return Matrix(adj_mat)
end

""
function augmented_adjacency_matrix(pm::ACPPowerModel, adj_mat::Matrix, bus::String, id::Int64)
    gens = MLOPF.get_reference(pm, :bus_gens)[bus]
    adj_mat = vcat(adj_mat[1:(i-1), :], repeat(adj_mat[i, :]', length(gens)), adj_mat[(i+1):end, :])
    adj_mat = hcat(adj_mat[:, 1:(i-1)], repeat(adj_mat[:, i]', length(gens))', adj_mat[:, (i+1):end])
    return adj_mat
end

"Build bus index map - required due to inconsitency between name and id of buses."
function get_bus_index_map(pm::ACPPowerModel)
    return Dict((string(b), i) for (i, b) in enumerate(keys(get_reference(pm, :bus))))
end

"Get specific elements from power model reference map."
function get_reference(pm::ACPPowerModel, key::Symbol; default=nothing)
    ref = pm.ref[:it][:pm][:nw][0][key]
    return ref
end

""
function get_augmented_bus_load(pm::ACPPowerModel, bus::String, key::String)
    load = sum(map(id -> pm.data["load"][id][key], get_reference(pm, :bus_loads)[bus]))
    gens = MLOPF.get_reference(pm, :bus_gens)[bus]
    return fill(load, min(1, length(gens)))
end

""
function get_augmented_bus_vm(pm::ACPPowerModel, bus::String)
    vm, vmin, vmax = getindex.(Ref(pm.solution), ("vm", "vmin", "vmax"))
    gens = MLOPF.get_reference(pm, :bus_gens)[bus]
    return fill((vm - vmin) / (vmax - vmin), min(1, length(gens)))
end

""
function get_augmented_bus_gen(pm::ACPPowerModel, bus::String, key::String)
    gens = MLOPF.get_reference(pm, :bus_gens)[bus]
    return isempty(gens) ? [NaN] : map(gens) do gen
        gen = pm.data["gen"]["$gen"]
        min, max = first(key) * "min", first(key) * "max"
        (gen[key] - gen[min]) / (gen[max] - gen[min])
    end
end