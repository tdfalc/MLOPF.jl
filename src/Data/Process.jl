using PowerModels
using ProgressMeter
using MLOPF

mutable struct ProcessedSample
    id::Int
    output::Dict{String,Any}
    regime::Vector{Bool}
    bus_type::Vector{Float64}
    bus_name::Vector{Float64}
    pd::Vector{Float64}
    pdmin::Vector{Float64}
    pdmax::Vector{Float64}
    qd::Vector{Float64}
    qdmin::Vector{Float64}
    qdmax::Vector{Float64}
    vm::Vector{Float64}
    vmin::Vector{Float64}
    vmax::Vector{Float64}
    pg::Vector{Float64}
    pmin::Vector{Float64}
    pmax::Vector{Float64}
    qg::Vector{Float64}
    qmin::Vector{Float64}
    qmax::Vector{Float64}
    ProcessedSample() = new(
        0,
        Dict{String,Any}(),
        Vector{Bool}(),
        (Vector{Float64}() for _ = 1:(length(fieldnames(ProcessedSample))-3))...,
    )
end

"""
    process_raw_samples(samples::Vector{MLOPF.Sample}}, network::Dict{String,Any})

This function takes as input a vector of raw samples and the corresponding PowerModels.jl network
    and extracts relevant data into a format ready for modelling.

# Arguments:
- `samples::Vector{MLOPF.Sample}` -- Vector of samples generated using `Sampler.generate_samples`.
- `network::Dict{String,Any}` -- Grid network in PowerModels.jl format.
- `inequality_constraints::Vector{String}` -- Vector of inequality consraints of the relevant AC-OPF problem.

# Keywords
- `normalise::Bool`: Flag whether or not to normalise samples. Defaults to true.

# Output:
- `Vector{ProcessedSample}` -- Vector of data ready to be processed for modelling.
"""
function process_raw_samples(
    samples::Vector{MLOPF.Sample},
    network::Dict{String,Any},
    inequality_constraints::Vector{String};
    normalise::Bool = true,
)
    @info "processing data using $(nprocs()) process(es)"
    processed = @showprogress pmap(
        sample -> process_raw_sample(sample, deepcopy(network), inequality_constraints),
        samples,
    )
    if normalise
        return MLOPF.normalise_samples(processed)
    end
    return processed
end

"Set network active and reactive load then extract processed data from power model."
function process_raw_sample(
    sample::MLOPF.Sample,
    network::Dict{String,Any},
    inequality_constraints::Vector{String},
)
    MLOPF.set_load_pd!(network, Float64.(sample.load[:pd]))
    MLOPF.set_load_qd!(network, Float64.(sample.load[:qd]))
    pm = PowerModels.instantiate_model(network, ACPPowerModel, PowerModels.build_opf)
    pm.solution = sample.output["solution"]
    congestion_regime = MLOPF.enumerate_constraints(sample.regime, inequality_constraints)
    data = process_raw_sample(pm)
    data.id, data.regime = sample.id, congestion_regime
    return data
end

"Extract procssed data from power model."
function process_raw_sample(pm::ACPPowerModel)

    bus_lookup_map = get_bus_lookup_map(pm)

    # First we get the solution lookup maps for each generator and bus.
    gen = Dict(:solution => pm.solution["gen"], :data => pm.data["gen"])
    bus = Dict(:solution => pm.solution["bus"], :data => pm.data["bus"])

    # We also need the reference indicies of each generator and load on every bus.
    parse_keys(d::Dict{Int,Vector{Int64}}) = Dict(string(k) => v for (k, v) in d)
    bus_gens = parse_keys(get_reference(pm, :bus_gens))
    bus_loads = parse_keys(get_reference(pm, :bus_loads))

    # Extract data to struct that maps each parameter to a vector of floats with length equal to the 
    # number of generators in the network.
    processed = ProcessedSample()
    for (b, _) in bus_lookup_map
        bus_type = bus[:data][b]["bus_type"]
        for g in (bus_type != 1 ? bus_gens[b] : [nothing])

            # Bus types:
            # - 1 no generator
            # - 2 at least one generator
            # - 3 slack bus
            append!(processed.bus_type, Float64(bus_type))
            append!(processed.bus_name, Base.parse(Float64, b))

            # TODO: Investigate if using a default load of 0 is problematic, rather than removing 
            # these buses from the input.
            append!(processed.pd, sum(Float64[get_load_pd(pm, id) for id in bus_loads[b]]))
            append!(processed.qd, sum(Float64[get_load_qd(pm, id) for id in bus_loads[b]]))
            append!(processed.vm, bus[:solution][b]["vm"])
            append!(processed.vmin, bus[:data][b]["vmin"])
            append!(processed.vmax, bus[:data][b]["vmax"])

            # If bus has at least one generator, we append the input with active and reactive
            # minimum, maxmium and actual power injections.
            no_generator, g = bus_type == 1, string(g)

            # Active and reactive power components.
            append!(processed.pg, no_generator ? NaN : gen[:solution][g]["pg"])
            append!(processed.qg, no_generator ? NaN : gen[:solution][g]["qg"])

            # We also save out the minimum and maximum values of these components to help us normalise later.
            append!(processed.pmin, no_generator ? NaN : gen[:data][g]["pmin"])
            append!(processed.pmax, no_generator ? NaN : gen[:data][g]["pmax"])
            append!(processed.qmin, no_generator ? NaN : gen[:data][g]["qmin"])
            append!(processed.qmax, no_generator ? NaN : gen[:data][g]["qmax"])

        end
    end

    return processed
end

"Build bus lookup map - required due to inconsitency between name and id of buses."
function get_bus_lookup_map(pm::ACPPowerModel)
    return Dict((string(b), i) for (i, b) in enumerate(keys(get_reference(pm, :bus))))
end

"Get specific elements from power model reference map."
function get_reference(pm::ACPPowerModel, key::Symbol)
    return pm.ref[:it][:pm][:nw][0][key]
end

function get_load_pd(pm::ACPPowerModel, id::Int)
    return get_reference(pm, :load)[id]["pd"]
end

function get_load_qd(pm::ACPPowerModel, id::Int)
    return get_reference(pm, :load)[id]["qd"]
end
