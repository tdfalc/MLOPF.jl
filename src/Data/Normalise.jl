using PowerModels
using ProgressMeter
using MLOPF

"""
    normalise_samples(samples::Vector{ProcessedSample}, network::Dict{String,Any})

This function normalises a vector of samples.

# Arguments:
- `data::Vector{ProcessedSample}` -- Vector of processed data generated using `Process.process_raw_samples`.

# Output:
- `Vector{ProcessedSample}` -- Vector of normalised samples.
"""
function normalise_samples(data::Vector{MLOPF.ProcessedSample})

    @info "normalising $(length(data)) samples using $(nprocs()) process(es)"

    # First we get minimum and maximum values across the dataset for all active and reactive loads.
    pdmin, pdmax = get_min_max(data, :pd)
    qdmin, qdmax = get_min_max(data, :qd)

    # TODO: Move normalisation to after train/test split to avoid data leakage.
    return @showprogress pmap(
        d -> normalise_sample(d, Dict(:pdmin => pdmin, :pdmax => pdmax), Dict(:qdmin => qdmin, :qdmax => qdmax)),
        data,
    )
end

"Transform each parameter using minimum/maxmium normalisation."
function normalise_sample(
    data::MLOPF.ProcessedSample,
    pd::Dict{Symbol,Vector{Float64}},
    qd::Dict{Symbol,Vector{Float64}},
)
    # Store minimum and maximum loads incase we want to apply a reverse transformation later on.
    data.pdmin, data.pdmax = pd[:pdmin], pd[:pdmax]
    data.qdmin, data.qdmax = qd[:qdmin], qd[:qdmax]

    # Normalise each parameter (minimum and maximum values for voltage magnitude, 
    # active and reactive generator injections are already provided by the power model).
    # This ensures implicit satisfaction of lower/upper bound inequality constraints 
    # in the AC-OPF problem.
    normalise!(data, :pd, data.pdmin, data.pdmax)
    normalise!(data, :qd, data.qdmin, data.qdmax)
    normalise!(data, :vm, data.vmin, data.vmax)
    normalise!(data, :pg, data.pmin, data.pmax)
    normalise!(data, :qg, data.qmin, data.qmax)

    return data
end

function get_min_max(data, field)
    values = eachrow(hcat(map(x -> getfield(x, field), data)...))
    return map(minimum, values), map(maximum, values)
end

function normalise!(data, field, vmin, vmax)
    vmax[vmax.==0] .= 1
    setproperty!(data, field, (getfield(data, field) .- vmin) ./ (vmax .- vmin))
end