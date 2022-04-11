using Flux
using GeometricFlux
using MLOPF

abstract type Graph <: NeuralNetwork end

"""
    graph_neural_network(
        size_in::Tuple{Int},
        size_out::Int,
        num_layers::Int;
        drop_out::Float64 = 0.0,
        act::Function = Flux.relu,
        fact::Function = Flux.sigmoid,
        conv::Type{T} = GeometricFlux.GCNConv,
        kwargs...,
    ) where {T<:GeometricFlux.AbstractGraphLayer}

This function builds a graph neural network graph as a Flux.jl chain type.
    
    # Arguments:
    - `size_in::Int` -- Network input size (number of channels).
    - `size_out::Int` -- Network output size. Number of neurons (channels) in the final layer for global 
        (local) encoding.
    - `num_layers::Int` -- Number of hidden layers.

# Keywords:
    - `drop_out::Float64` -- Probability assigned to drop out layer. Defaults to 0.
    - `act::Function` -- Activation function (on hidden layer). Defaults to ReLU.
    - `fact::Function` -- Final activation function (on output layer). Defaults to Sigmoid.
    - `conv::Type{T} ` -- Type of graph neural network layer from GeometricFlux.jl (src/layers/conv.jl) package.

# Outputs
    - `Flux.Chain`: Graph neural network.
"""
function graph_neural_network(
    size_in::Tuple{Int},
    size_out::Int,
    num_layers::Int;
    drop_out::Float64 = 0.0,
    act::Function = Flux.relu,
    fact::Function = Flux.sigmoid,
    conv::Type{C} = GeometricFlux.GCNConv,
    encoding::Type{E} = MLOPF.Global,
    kwargs...,
) where {C<:GeometricFlux.AbstractGraphLayer,E<:MLOPF.Encoding}
    size(i::Int) = i == 0 ? size_in : ceil(Int, size_in / 4) * 4 * 2^(i - 1)
    chain = []
    for i ∈ 1:(num_layers)
        push!(chain, conv(size(i - 1) => size(i), act; kwargs...))
        push!(chain, x -> FeaturedGraph(x.graph.S, nf = Flux.BatchNorm(layer.out)(x.nf))) # Check this
        push!(chain, x -> FeaturedGraph(x.graph.S, nf = Flux.dropout(x.nf, drop_out)))
    end
    if isa(encoding, MLOPF.Global)
        push!(chain, x -> vec(x.nf'))
        push!(chain, x -> Flux.Dense(size(x), size_out, fact)(x))
    elseif isa(encoding, MLOPF.Local)
        push!(chain, x -> conv(size(i - 1) => size_out, fact; kwargs...)(x).nf)
    else
        error("failed to construct network for unknown encoding")
    end
    return Flux.Chain(chain...)
end

function model_input(::Type{Graph}, data::Vector{MLOPF.ProcessedSample})
    return map(x -> FeaturedGraph(x.adjacency_matrix, nf = hcat([x.pd, x.qd]...)'), data)
end

function model_factory(::Type{Graph}, size_in::Int, size_out::Int, num_layers::Int; kwargs...)
    return graph_neural_network(size_in, size_out, num_layers; kwargs...)
end
