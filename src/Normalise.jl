using PowerModels
using ProgressMeter
using MLOPF

mutable struct MinMaxScaler
    pd::Dict{String,Vector{Float64}}
    qd::Dict{String,Vector{Float64}}
    vm::Dict{String,Vector{Float64}}
    pg::Dict{String,Vector{Float64}}
    function MinMaxScaler(data::Vector{Dict{String,Any}})
        function get_min_max(parameter)
            values = eachrow(hcat(map(x -> x["parameters"][parameter.key], data)...))
            return Dict(parameter.min => map(minimum, values), parameter.max => map(maximum, values))
        end
        return new(get_min_max(pd), get_min_max(qd), get_min_max(vm), get_min_max(pg))
    end
end

"""
    normalise(data::Vector{Data{String,Any}})

This function normalises the specified parameters for a vector of samples.

# Arguments:
    - `data::Vector{Dict{String,Any}}` -- Vector of feasible samples.
    - `scaler::MinMaxScaler` -- Vector of feasible samples.
    - `parameters::Tuple` -- Vector of feasible samples.

# Output:
    - `Vector{Dict{String,Any}}` -- Vector of feasible samples with normalised parameters.
"""
function normalise(data::Vector{Dict{String,Any}}, scaler::MinMaxScaler, parameters::Tuple)
    function normalise!(d, parameter)
        bounds = getproperty(scaler, Symbol(parameter.key))
        vmin, vmax = bounds[parameter.min], bounds[parameter.max]
        vmax[vmax.==0] .= 1
        d["parameters"][parameter.key] = (d["parameters"][parameter.key] .- vmin) ./ (vmax .- vmin)
    end
    return pmap(deepcopy(data)) do d
        for parameter in parameters
            normalise!(d, parameter)
        end
        d
    end
end
