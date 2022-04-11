using Flux
using GeometricFlux
using MLOPF

abstract type Graph <: NeuralNetwork end


"""
    graph_neural_network(
        ::Type{l},
        size_in::Tuple{int},
        size_out::int,
        num_layers::int;
        act::Function = Flux.relu,
        fact::Function = Flux.sigmoid,
        kwargs...,
    ) where {l<:GeometricFlux.AbstractGraphLayer}

This function builds a graph neural network graph as a Flux.jl chain type.
    
# Arguments:
    - `Type{l}` -- Type of graph neural network layer from GeometrixFlux.jl package.
    - `size_in::Tuple{int}` -- Network input size (num_channels, num_nodes).
    - `size_out::int}` -- Network output size.
    - `num_layers::int` -- Number of hidden layers.
    - `act::Function` -- Activation function (on hidden layer).
    - `fact::Function` -- Final activation function (on output layer).

# Outputs
    - `Flux.Chain`: Graph neural network.
"""
function graph_neural_network(
    ::Type{l},
    size_in::Tuple{int},
    size_out::int,
    num_layers::int;
    act::Function = Flux.relu,
    fact::Function = Flux.sigmoid,
    kwargs...,
) where {l<:GeometricFlux.AbstractGraphLayer}
    (num_channels, num_nodes), chain = size_in, []
    for i ∈ 1:(num_layers)
        in, out = MLOPF.get_size(num_channels, i - 1), MLOPF.get_size(num_channels, i)
        push!(chain, l(in => out, act; kwargs...))
        # TODO: Check batch norm implementation.
        push!(chain, x -> FeaturedGraph(x.graph.S, nf = Flux.BatchNorm(layer.out)(x.nf)))
        push!(chain, x -> FeaturedGraph(x.graph.S, nf = Flux.dropout(x.nf, drop_out)))
    end
    push!(chain, x -> vec(x.nf'))
    push!(chain, Flux.Dense(num_nodes * size(index - 1), size_out, fact))
    return Flux.Chain(chain...)
end

function model_input(::Type{Graph}, data::Vector{MLOPF.ProcessedSample})
    return map(x -> FeaturedGraph(x.adjacency_matrix, nf = hcat([x.pd, x.qd]...)'), data)
end
