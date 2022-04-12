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
    result::Dict{String,Any}
end

"""
    generate_samples(network::Dict{String, Any}, num_samples::Int, alpha::Float64; max_iter::Int = 100, nmo::Bool = false)

This function generates feasible samples by re-scaling each active and reactive load component (relative to 
    nominal values) by factors independently drawn from a uniform distribution.

# Arguments:
- `network::Dict{String, Any}` -- Grid network in PowerModels.jl format.
- `num_samples::Int` -- Total number of feasible samples to generate.
- `alpha::Float64` -- Defines parameters of Uniform distrubtion used to re-scale inputs.

# Keywords:
- `max_iter::Int` -- Maximum number of iterations the IPOPT algorithm should run before declaring infeasiblity. Defaults to 100.
- `nmo::Bool` -- Flag to randomly silence a branch in each sample to emulate N-1 contingency. Defaults to false.

# Outputs
- `Vector{RawSample}`: Vector of feasible samples.
"""
function generate_samples(
    network::Dict{String,Any},
    num_samples::Int,
    alpha::Float64;
    max_iter::Int = 100,
    nmo:bool = false,
)
    @info "generating $(num_samples) samples using $(nprocs()) process(es)"
    return @showprogress pmap(
        id -> generate_sample(deepcopy(network), alpha; id = id, max_iter = max_iter, nmo = nmo),
        1:num_samples,
    )
end

function generate_sample(network::Dict{String,Any}, alpha::Float64; id::Int = 1, max_iter::Int = 100, nmo::Bool = False)
    pd, qd = get_load(network)
    result, is_feasible = Dict{String,Any}, false
    while !is_feasible
        let network = deepcopy(network)

            # First, we select a branch to silence at random to emulate N-1 contingency.
            nmo && silence_random_branch!(network)

            # Next, we sample re-scaling factors from a Uniform distribution (parameterised by alpha) 
            # and update to relevant network parameters.
            distribution = Uniform(1.0 - alpha, 1.0 + alpha)
            set_load!(network, pd .* rand(distribution, length(pd)), qd .* rand(distribution, length(qd)))

            # We then initialise the Ipopt optimisation algorithm and solve the new AC-OPF problem.
            optimizer = optimizer_with_attributes(Ipopt.Optimizer, "max_iter" => max_iter, "print_level" => 0)
            power_model = PowerModels.instantiate_model(network, ACPPowerModel, PowerModels.build_opf)
            result = PowerModels.optimize_model!(power_model, optimizer = optimizer)
            result["binding_status"] = binding_status(power_model)
            result["load"] = Dict("pd" => get_load_pd(network), "qd" => get_load_qd(network))

            # Finally, we check for feasibility based on the terminiation status of Ipopt.
            is_feasible = validate_feasibility(result["termination_status"])
        end
    end
    return RawSample(id, result)
end

function validate_feasibility(status::MOI.TerminationStatusCode)
    return (status == MOI.LOCALLY_SOLVED) || (status == MOI.OPTIMAL)
end

"Proxy for randomly removing branch from AC-OPF problem whilst preserving topology."
function silence_branch!(network::Dict{String,Any}; br_r::Float64 = 9e9)
    num_branches = length(network["branch"])
    id = string(rand(1:num_branches))
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
