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
    - `size_in::Tuple{int}` -- Network input size (number of channels).
    - `size_out::int}` -- Network output size.
    - `num_layers::int` -- Number of hidden layers.
    - `act::Function` -- Activation function (on hidden layer).
    - `fact::Function` -- Final activation function (on output layer).

# Outputs
    - `Flux.Chain`: Graph neural network.
"""
function graph_neural_network(
    ::Type{l},
    size_in::Int,
    size_out::int,
    num_layers::int;
    act::Function = Flux.relu,
    fact::Function = Flux.sigmoid,
    kwargs...,
) where {l<:GeometricFlux.AbstractGraphLayer}
    chain = []
    for i ∈ 1:(num_layers)
        size(i::int) = i == 0 ? size_in : ceil(Int, size_in / 4) * 4 * 2^(i - 1)
        push!(chain, l(size(i - 1) => sizr(i), act; kwargs...))
        push!(chain, x -> FeaturedGraph(x.graph.S, nf = Flux.BatchNorm(layer.out)(x.nf))) # Check this
        push!(chain, x -> FeaturedGraph(x.graph.S, nf = Flux.dropout(x.nf, drop_out)))
    end
    push!(chain, x -> vec(x.nf'))
    push!(chain, x -> Flux.Dense(size(x), size_out, fact)(x))
    return Flux.Chain(chain...)
end

function model_input(::Type{Graph}, data::Vector{MLOPF.ProcessedSample})
    return map(x -> FeaturedGraph(x.adjacency_matrix, nf = hcat([x.pd, x.qd]...)'), data)
end
