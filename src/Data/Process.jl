using PowerModels
using ProgressMeter
using MLOPF

mutable struct ProcessedSample
    id::Int
    output::Dict{String,Any} # Output dictionary containing information of solved OPF problem.
    regime::Vector{Bool} # Congestion regime of solved OPF problem.   
    adjacency_matrix::Matrix{Float64}  # Network adjacency matrix (dense).
    bus_type::Vector{Float64} # Bus types: 1 - no generator, 2 - at least one generator, 3 - slack bus.
    bus_name::Vector{Float64} # Bus name used for identification.
    pd::Vector{Float64} # Bus load (active component).
    pdmin::Vector{Float64}
    pdmax::Vector{Float64}
    qd::Vector{Float64} # Bus load (reactive component).
    qdmin::Vector{Float64}
    qdmax::Vector{Float64}
    vm::Vector{Float64} # Voltage magnitude.
    vmin::Vector{Float64}
    vmax::Vector{Float64}
    pg::Vector{Float64}  # Generator injected power (active component).
    pmin::Vector{Float64}
    pmax::Vector{Float64}
    qg::Vector{Float64} # Generator injected power (reactive component).
    qmin::Vector{Float64}
    qmax::Vector{Float64}
    ProcessedSample(id::Int, congestion_regime, adjacency_matrix::Matrix{Float64}) = new(
        id,
        Dict{String,Any}(),
        congestion_regime,
        adjacency_matrix,
        (Vector{Float64}() for _ = 1:(length(fieldnames(ProcessedSample))-4))...,
    )
end

"""
    process_raw_samples(samples::Vector{MLOPF.RawSample}}, network::Dict{String,Any})

This function takes as input a vector of raw samples and the corresponding PowerModels.jl network
    and extracts relevant data into a format ready for modelling.

# Arguments:
- `samples::Vector{MLOPF.RawSample}` -- Vector of samples generated using `Sample.generate_samples`.
- `network::Dict{String,Any}` -- Grid network in PowerModels.jl format.
- `inequality_constraints::Vector{String}` -- Vector of inequality consraints of the relevant AC-OPF problem.

# Keywords
- `normalise::Bool`: Flag whether or not to normalise samples. Defaults to true.

# Output:
- `Vector{ProcessedSample}` -- Vector of data ready to be processed for modelling.
"""
function process_raw_samples(
    samples::Vector{MLOPF.RawSample},
    network::Dict{String,Any},
    inequality_constraints::Vector{String};
    normalise::Bool = true,
)
    @info "processing $(length(samples)) samples using $(nprocs()) process(es)"
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
    sample::MLOPF.RawSample,
    network::Dict{String,Any},
    inequality_constraints::Vector{String},
)
    MLOPF.set_load!(network, Float64.(sample.load[:pd]), Float64.(sample.load[:qd]))
    pm = PowerModels.instantiate_model(network, ACPPowerModel, PowerModels.build_opf)
    pm.solution = sample.output["solution"]
    congestion_regime = MLOPF.enumerate_constraints(sample.regime, inequality_constraints)
    data = process_raw_sample(pm, congestion_regime; id = sample.id)
    return data
end

"Extract procssed data from power model."
function process_raw_sample(pm::ACPPowerModel, congestion_regime::Vector{Bool}; id::Int = 0)

    bus_lookup_map = get_bus_lookup_map(pm)

    # First we get the solution lookup maps for each generator and bus.
    gen = Dict(:solution => pm.solution["gen"], :data => pm.data["gen"])
    bus = Dict(:solution => pm.solution["bus"], :data => pm.data["bus"])

    # We also need the reference indicies of each generator and load on every bus.
    parse_keys(d::Dict{Int,Vector{Int64}}) = Dict(string(k) => v for (k, v) in d)
    bus_gens = parse_keys(get_reference(pm, :bus_gens))
    bus_loads = parse_keys(get_reference(pm, :bus_loads))

    # Get network adjacency matrix.
    adj_mat = get_adjacency_matrix(pm)

    # Extract data to struct that maps each parameter to a vector of floats with length equal to the 
    # number of generators in the network.
    processed = ProcessedSample(id, congestion_regime, adj_mat)
    for (b, i) in bus_lookup_map
        bus_type = bus[:data][b]["bus_type"]
        num_to_add = 0
        for g in (bus_type != 1 ? bus_gens[b] : [nothing])

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

            num_to_add += 1

        end
        # Finally, we need to augment the adjacency matrix to contain new rows and columns in accoradance
        # with the rows and columns added for each generator.
        # TODO: Remove code duplication to find length of generators.
        adj_mat = vcat(adj_mat[1:(i-1), :], repeat(adj_mat[i, :]', num_to_add), adj_mat[(i+1):end, :])
        adj_mat = hcat(adj_mat[:, 1:(i-1)], repeat(adj_mat[:, i]', num_to_add)', adj_mat[:, (i+1):end])
    end
    processed.adjacency_matrix = adj_mat
    return processed
end

"Get (sparse) adjacency matrix using PowerModels API and convert to dense matrix."
function get_adjacency_matrix(pm::ACPPowerModel)
    adj_mat, _ = PowerModels._adjacency_matrix(pm)
    return Matrix(adj_mat)
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
