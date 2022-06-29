using Flux
using GeometricFlux
using MLOPF

abstract type Graph <: NeuralNetwork end

"""
    graph_neural_network(
        size_in::Tuple{Int64},
        size_out::Int64,
        num_layers::Int64;
        drop_out::Float64 = 0.0,
        act::Function = Flux.relu,
        fact::Function = Flux.sigmoid,
        conv::Type{C} = GeometricFlux.GCNConv,
        encoding::Type{E} = MLOPF.Encoding,
        kwargs...,
    ) where {T<:GeometricFlux.AbstractGraphLayer}

This function builds a graph neural network as a Flux.jl chain type.
    
    # Arguments:
    - `size_in::Int64` -- Network input size (number of channels).
    - `size_out::Int64` -- Network output size. Number of neurons (channels) in the final layer for global 
        (local) encoding.
    - `num_layers::Int64` -- Number of hidden layers.

# Keywords:
    - `drop_out::Float64` -- Probability assigned to drop out layer. Defaults to 0.
    - `act::Function` -- Activation function on hidden layers. Defaults to ReLU.
    - `fact::Function` -- Final activation function on output layer. Defaults to Sigmoid.
    - `conv::Type{C}` -- Type of graph neural network layer from GeometricFlux.jl (src/layers/conv.jl) package.
    - `Encoding::Type{E}` -- Specify between local and global variable encoding architecture.

# Outputs
    - `Flux.Chain`: Graph neural network.
"""
function graph_neural_network(
    size_in::Tuple,
    size_out::Int64,
    num_layers::Int64;
    drop_out::Float64=0.0,
    act=Flux.relu,
    fact=Flux.sigmoid,
    conv::Type{C}=GeometricFlux.GCNConv,
    encoding::Type{E}=MLOPF.Global,
    kwargs...
) where {C<:GeometricFlux.AbstractGraphLayer,E<:MLOPF.Encoding}
    channels, nodes = size_in
    chain = []
    size(i::Int) = i == 0 ? channels : ceil(Int64, channels / 4) * 4 * 2^(i - 1)
    for i in 1:(num_layers)
        push!(chain, conv(size(i - 1) => size(i), act; kwargs...))
        push!(chain, x -> FeaturedGraph(x.graph.S, nf=Flux.BatchNorm(size(i))(x.nf))) # Check this
        push!(chain, x -> FeaturedGraph(x.graph.S, nf=Flux.dropout(x.nf, drop_out)))
    end
    if isa(encoding, Type{MLOPF.Global})
        push!(chain, x -> vec(x.nf))
        push!(chain, Flux.Dense(nodes * size(num_layers), size_out, fact))
    elseif isa(encoding, Type{MLOPF.Local})
        push!(chain, conv(size(num_layers) => size_out, act))
        push!(chain, x -> vec(x.nf))
    else
        error("failed to construct network for unknown encoding")
    end
    return Flux.Chain(chain...)
end

function model_input(::Type{Graph}, data::Vector{Dict{String,Any}})
    return map(x -> FeaturedGraph(x["adjacency_matrix"],
            nf=Float32.(hcat([x["parameters"][pd.key], x["parameters"][qd.key]]...)')), data)
end

function model_factory(::Type{Graph}, size_in::Union{Int64,Tuple}, size_out::Int64, num_layers::Int64; kwargs...)
    return graph_neural_network(size_in, size_out, num_layers; kwargs...)
end
