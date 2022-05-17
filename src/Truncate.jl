"""
    truncate!(network::Dict{String,Any})

This function prunes disabled branches and generators from the network.
    
# Arguments:
- `network::Dict{String,Any}` -- Grid network in PowerModels.jl format.
"""
function truncate!(network::Dict{String,Any})
    remove_disabled_branches!(network)
    remove_disabled_generators!(network)
end

function remove_disabled_branches!(network::Dict{String,Any})
    for (id, branch) in network["branch"]
        if branch["br_status"] == 0
            delete!(network["branch"], id)
        end
    end
end

function remove_disabled_generators!(network::Dict{String,Any})
    for (id, gen) in network["gen"]
        if (gen["gen_status"] == 0) || (gen["pmax"] == gen["pmin"] == gen["qmax"] == gen["qmin"] == 0)
            delete!(network["gen"], id)
        end
    end
end
