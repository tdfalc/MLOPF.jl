using PowerModels
using ProgressMeter
using MLOPF

mutable struct MinMaxScaler
    pd::Dict{String,Vector{Float64}}
    qd::Dict{String,Vector{Float64}}
    function MinMaxScaler(data::Vector{Dict{String,Any}})
        function get_min_max(parameter::LoadParameter)
            values = eachrow(hcat(map(x -> x["parameters"][parameter.key], data)...))
            return Dict(parameter.min => map(minimum, values), parameter.max => map(maximum, values))
        end
        return new(get_min_max(pd), get_min_max(qd))
    end
end

"""
    normalise_load(data::Vector{Data{String,Any}})

This function normalises the load components for a vector of samples.

# Arguments:
    - `data::Vector{Dict{String,Any}}` -- Vector of feasible samples.

# Output:
    - `Vector{Dict{String,Any}}` -- Vector of feasible samples with normalised load components.
"""
function normalise_load(data::Vector{Dict{String,Any}}, scaler::MinMaxScaler)
    function normalise!(d, parameter)
        bounds = getproperty(scaler, Symbol(parameter.key))
        vmin, vmax = bounds[parameter.min], bounds[parameter.max]
        vmax[vmax.==0] .= 1
        d["parameters"][parameter.key] = (d["parameters"][parameter.key] .- vmin) ./ (vmax .- vmin)
    end
    return pmap(deepcopy(data)) do d
        normalise!(d, MLOPF.pd)
        normalise!(d, MLOPF.qd)
        d
    end
end
