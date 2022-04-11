using Distributions
using Distributed
using Ipopt
using JuMP
using Random
using Memoize
using ProgressMeter
using PowerModels

PowerModels.silence()

struct RawSample
    id::Int
    output::Dict{String,Any}
    regime::Dict{String,Vector{Bool}}
    load::Dict{Symbol,Vector{Float64}}
end

"""
    generate_samples(network::Dict{String, Any}, num_samples::Int, alpha::Float64; max_iter::Int = 100)

This function generates feasible samples by re-scaling each active and reactive load component 
    (relative to nominal values) independently drawn from a uniform distribution.

# Arguments:
- `network::Dict{String, Any}` -- Grid network in PowerModels.jl format.
- `num_samples::Int` -- Total number of feasible samples to generate.
- `alpha::Float64` -- Defines parameters of Uniform distrubtion used to re-scale inputs.

# Keywords:
- `max_iter::Int` -- Maximum number of iterations the IPOPT algorithm should run before declaring infeasiblity.

# Outputs
- `Vector{RawSample}`: Vector of feasible samples.
"""
function generate_samples(network::Dict{String,Any}, num_samples::Int, alpha::Float64; max_iter::Int = 100)
    @info "generating $(num_samples) samples using $(nprocs()) process(es)"
    return @showprogress pmap(
        id -> generate_sample(deepcopy(network), alpha; id = id, max_iter = max_iter),
        1:num_samples,
    )
end

function generate_sample(network::Dict{String,Any}, alpha::Float64; id::Int = 1, max_iter::Int = 100, nmo::Bool = False)
    pd, qd = get_load(network)
    power_model, nw = ACPPowerModel, deepcopy(network)
    is_feasible, output = false, Dict{String,Any}
    while !is_feasible
        # First we randomly remove a line from the network to emulate N-1 contingency.
        nmo ? remove_branch!(nw, rand(1:length(nw["branch"]))) : nothing

        # Next we sample re-scaling factors from a Uniform distribution (parameterised by alpha) 
        # and update the respective network parameters.
        distribution = Uniform(1.0 - alpha, 1.0 + alpha)
        set_load!(nw, pd .* rand(distribution, length(pd)), qd .* rand(distribution, length(qd)))

        # Finally we intialise Ipopt and solve the updated OPF problem then check for feasibility.
        optimizer = optimizer_with_attributes(Ipopt.Optimizer, "max_iter" => max_iter, "print_level" => 0)
        power_model = PowerModels.instantiate_model(nw, ACPPowerModel, PowerModels.build_opf)
        output = PowerModels.optimize_model!(power_model, optimizer = optimizer)
        is_feasible = validate_feasibility(output["termination_status"])
        nw = deepcopy(network)
    end
    return RawSample(
        id,
        output,
        binding_status(power_model),
        Dict(:pd => get_load_pd(network), :qd => get_load_qd(network)),
    )
end

function validate_feasibility(status::MOI.TerminationStatusCode)
    return (status == MOI.LOCALLY_SOLVED) || (status == MOI.OPTIMAL)
end

"Proxy for removing branch from AC-OPF problem whilst preserving topology."
function remove_branch!(network::Dict{String,Any}, id::Int; br_r::Float64 = 9e9)
    network["branch"][id]["b_fr"] = 0.0
    network["branch"][id]["b_to"] = 0.0
    network["branch"][id]["br_x"] = 0.0
    network["branch"][id]["br_r"] = br_r
end

function get_load_pd(network::Dict{String,Any})
    pd = []
    for (_, load) in sort(network["load"])
        append!(pd, load["pd"])
    end
    return pd
end

function get_load_qd(network::Dict{String,Any})
    qd = []
    for (_, load) in sort(network["load"])
        append!(qd, load["qd"])
    end
    return qd
end

function set_load_pd!(network::Dict{String,Any}, pd::Array{Float64})
    for (i, (_, load)) in enumerate(sort(network["load"]))
        load["pd"] = pd[i]
    end
end

function set_load_qd!(network::Dict{String,Any}, qd::Array{Float64})
    for (i, (_, load)) in enumerate(sort(network["load"]))
        load["qd"] = qd[i]
    end
end

"Convenience function to get both the active and reactive components of bus loads."
function get_load(network::Dict{String,Any})
    pd = get_load_pd(network)
    qd = get_load_qd(network)
    return pd, qd
end

"Convenience function to set both the active and reactive components of bus loads."
function set_load!(network::Dict{String,Any}, pd::Array{Float64}, qd::Array{Float64})
    set_load_pd!(network, pd)
    set_load_qd!(network, qd)
end
