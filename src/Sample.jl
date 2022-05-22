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
    max_iter::Int=100,
    nmo::Bool=false
)
    @info "generating $(num_samples) samples using $(nprocs()) process(es)"
    return @showprogress pmap(
        id -> generate_sample(deepcopy(network), alpha; id=id, max_iter=max_iter, nmo=nmo),
        1:num_samples,
    )
end

function generate_sample(network::Dict{String,Any}, alpha::Float64; id::Int=1, max_iter::Int=100, nmo::Bool=False)
    let network = deepcopy(network), (pd, qd) = get_network_loads(network)
        # Select a random branch to silence (emulating N-1 contingency).
        nmo && silence_random_branch!(network)

        # Sample re-scaling factors from a Uniform distribution (parameterised by alpha).
        distribution = Uniform(1.0 - alpha, 1.0 + alpha)
        set_network_loads!(network, pd .* rand(distribution, length(pd)), qd .* rand(distribution, length(qd)))

        # Solve the new AC-OPF problem and validate feasibility.
        pm = PowerModels.instantiate_model(network, ACPPowerModel, PowerModels.build_opf)
        if solve_acopf!(pm, max_iter)
            return extract_data(pm, MLOPF.binding_status(pm), id)
        end
    end
    return generate_sample(network, alpha; id=id, max_iter=max_iter, nmo=nmo)
end

function extract_data(pm::ACPPowerModel, congestion_regime::Dict{String,Vector{Bool}}, id::Int64)
    adj_mat = MLOPF.get_adjacency_matrix(pm)
    # Extract parameters to a dictionary that maps each parameter name to a vector of floats
    # with length equal to the number of generators in the network (this simplifies the
    # construction of input tensors for local graph neural network architectures).
    parameters = DefaultDict(() -> [])
    for (bus, i) in MLOPF.get_bus_index_map(pm)
        for v in (vm,)
            append!(parameters[v.key], MLOPF.augmented_bus_parameter(pm, bus, v))
        end
        for g in (pg, qg)
            append!(parameters[g.key], MLOPF.augmented_gen_parameter(pm, bus, g))
        end
        for l in (pd, qd)
            append!(parameters[l.key], MLOPF.augmented_load_parameter(pm, bus, l))
        end
        adj_mat = MLOPF.augment_adjacency_matrix(pm, adj_mat, bus, i)
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

"Solve OPF problem for specified power model and return feasibility boolean flag."
function solve_acopf!(power_model::ACPPowerModel, max_iter::Int)
    optimizer = optimizer_with_attributes(Ipopt.Optimizer, "max_iter" => max_iter, "print_level" => 0)
    output = PowerModels.optimize_model!(power_model, optimizer=optimizer)
    return MLOPF.validate_feasibility(output["termination_status"])
end

"Proxy for removing a random branch from optimization problem whilst preserving topology."
function silence_branch!(network::Dict{String,Any}; br_r::Float64=9e9)
    id = string(rand(1:length(network["branch"])))
    for parameter ∈ ("b_fr", "b_to", "br_x", "br_r")
        setindex!(network["branch"][id], parameter == "br_r" ? br_r : 0.0, parameter)
    end
end

function get_network_loads(network::Dict{String,Any}, key::String)
    return [load[key] for (_, load) in sort(network["load"])]
end

"Convenience function to get both the active and reactive components of bus loads."
function get_network_loads(network::Dict{String,Any})
    return get_network_loads(network, pd.key), get_network_loads(network, qd.key)
end

function set_network_loads!(network::Dict{String,Any}, key::String, values::Array{Float64})
    for (i, (_, load)) in enumerate(sort(network["load"]))
        setindex!(load, values[i], key)
    end
end

"Convenience function to set both the active and reactive components of bus loads."
function set_network_loads!(network::Dict{String,Any}, active::Array{Float64}, reactive::Array{Float64})
    set_network_loads!(network, pd.key, active)
    set_network_loads!(network, qd.key, reactive)
end

"Get sparse adjacency matrix using PowerModels API and return dense form."
function get_adjacency_matrix(pm::ACPPowerModel)
    adj_mat, _ = PowerModels._adjacency_matrix(pm)
    return Matrix(adj_mat)
end

"Build bus index map (required due to inconsitency between name and id of buses)."
function get_bus_index_map(pm::ACPPowerModel)
    return Dict((string(b), i) for (i, b) in enumerate(keys(reference(pm, :bus))))
end

"Get specific elements from power model reference map."
function reference(pm::ACPPowerModel, key::Symbol)
    return pm.ref[:it][:pm][:nw][0][key]
end
