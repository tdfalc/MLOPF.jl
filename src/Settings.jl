using YAML

"""
    get_settings(; overrides::Dict{String,Any} = Dict{String,Any}(), filename::String = "./settings.yaml")

This function converts a YAML file into a Dictionary with optional overrides.

# Keywords
    - `overrides::Dict{String,Any}`: Dictionary of overrides used to update the settings.
    - `filename::String`: Path to location of the settings YAML file.

# Outputs
    - `Dict{String,Any}`: Dictionary representation of the pipeline settings.
"""
function get_settings(; overrides::Dict{String,Any}=Dict{String,Any}(), filename::String="./settings.yaml")
    settings = YAML.load_file(filename; dicttype=Dict{String,Any})
    nested_merge!(settings, overrides)
    return nested_parse(settings)
end

function nested_merge!(d::Dict...)
    return merge!(nested_merge!, d...)
end

nested_merge!(x::Any...) = x[end]

function nested_parse(d::Dict)
    return (; Dict(Symbol(k) => nested_parse(v) for (k, v) in d)...)
end

nested_parse(x::Any) = x
