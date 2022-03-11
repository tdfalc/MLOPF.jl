using YAML

"""
    get_settings(; overrides::Dict{Symbol,Any} = Dict{Symbol,Any}(), filename::String = "./settings.yaml")

This function converts a YAML file into a NamedTuple with optional overrides.

# Keywords
- `overrides::Dict{Symbol,Any}`: Dictionary of overrides used to update the settings.
- `filename::String`: Path to location of the settings YAML file.

# Outputs
- `NamedTuple{}`: Named Tuple representation of the pipeline settings.
"""
function get_settings(; overrides::Dict = Dict(), filename::String = "./settings.yaml")
    settings = YAML.load_file(filename; dicttype = Dict{Symbol,Any})
    nested_merge!(settings, overrides)
    return parse(settings)
end

function nested_merge!(d::Dict...)
    return merge!(nested_merge!, d...)
end

nested_merge!(x::Any...) = x[end]

function parse(d::Dict)
    return (; (k => parse(v) for (k, v) in d)...)
end

parse(x::Any) = x
