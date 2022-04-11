using Flux
using GeometricFlux
using MLOPF

abstract type Graph <: NeuralNetwork end

function model_input(::Type{Graph}, data::Vector{MLOPF.ProcessedSample})
    return map(x -> FeaturedGraph(x.adjacency_matrix, nf = hcat([x.pd, x.qd]...)'), data)
end

"""
    convolutional_neural_network(layers::Vector{MLOPF.Layer}, drop_out::Float64)

This function builds a convolutional neural network graph using Flux's Chain type.
    
# Arguments:
    - `layers::Vector -- Layers defining the neural network architecture.

# Keywords
    - `conv::` -- Probability assigned to drop out layer. Defaults to 0.
    - `drop_out::Float64{}` -- Probability assigned to drop out layer. Defaults to 0.

# Outputs
    - `Flux.Chain`: Convolutional neural network.
"""
function graph_neural_network(
    ::Type{l},
    size_in::int,
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