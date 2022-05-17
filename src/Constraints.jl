using JuMP
using PowerModels
using MathOptInterface

const inequality_constraints =   [
        "(JuMP.AffExpr, MathOptInterface.LessThan{Float64})",
        "(JuMP.AffExpr, MathOptInterface.GreaterThan{Float64})",
        "(JuMP.QuadExpr, MathOptInterface.LessThan{Float64})",
        "(JuMP.VariableRef, MathOptInterface.GreaterThan{Float64})",
        "(JuMP.VariableRef, MathOptInterface.LessThan{Float64})",
    ]

"""
    function binding_status(model::Model; threshold::Float64 = 1.0e-5)

This function returns an enumeration of constraint activity. Specifically, a dictionary mapping between 
    constraint types and a boolean array indicating the binding status of each constraint.

# Arguments
- `model::Model`: JuMP model solved using PowerModels.jl.
- `threshold::Float64`: Threshold for evaluating binding status of inequality constraints.

# Output
- `Dict::{String, Vector{Bool}}`: Map between constraint types and binding statuses.
"""
function binding_status(model::JuMP.Model; threshold::Float64 = 1.0e-5)
    status = Dict{String, Vector{Bool}}()
    for (func, ctype) in list_of_constraint_types(model)
        key = string((func, ctype))
        status[key] = ones(Bool, num_constraints(model, (func, ctype)...))
        if ctype != MathOptInterface.EqualTo{Float64}
            constraints = all_constraints(model, func, ctype)
            sets = map(c -> constraint_object(c).set, constraints)
            diff = value.(constraints) - map(s -> getproperty(s, fieldnames(typeof(s))...), sets)
            if ctype == MathOptInterface.GreaterThan{Float64}
                diff = map(diff) do d
                    d < 0.0 ? 0.0 : abs(d)
                end
            else
                diff = map(diff) do d
                    d > 0.0 ? 0.0 : abs(d)
                end
            end
            status[key][diff.>threshold] .= false
        end
    end
    return status
end

function binding_status(power_model::ACPPowerModel; threshold::Float64 = 1.0e-5)
    return binding_status(power_model.model, threshold = threshold)
end

function enumerate_constraints(congestion_regime::Dict{String,Vector{Bool}})
    return vcat(map(c -> congestion_regime[c], inequality_constraints)...)
end

