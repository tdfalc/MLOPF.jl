using Flux
using GeometricFlux
using MLOPF

abstract type Graph <: NeuralNetwork end
struct GraphLayer <: NeuralNetworkLayer end

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
function graph_neural_network(layers::Vector; conv = ChebConv, drop_out::Float64 = 0.0, kwargs...)
    chain = []
    for layer ∈ layers
        if isa(layer, MLOPF.GraphLayer)
            push!(graph, conv(layer.in => layer.out; kwargs...))
            # Try instance norm
            # Does batchnorm even work here? As we have a single instance.
            # push!(chain, x -> FeaturedGraph(x.graph.S, nf=Flux.BatchNorm(layer.out)(x.nf)))
            push!(chain, x -> FeaturedGraph(x.graph.S, nf = Flux.dropout(x.nf, drop_out)))
        else
            push!(chain, x -> vec(x.nf'))
            push!(chain, Flux.Dense(layer.in, layer.out, layer.act))
        end
    end
    return Flux.Chain(graph...)
end

function model_input(::Type{Graph}, data::Vector{MLOPF.ProcessedSample})
    return map(x -> FeaturedGraph(x.adjacency_matrix, nf = hcat([x.pd, x.qd]...)'), data)
end

function graph_layer(index::Int, num_layers::Int, size_in::Tuple, size_out::Int, act, fact)
    num_channels, num_nodes = size_in
    size(i) = i == 0 ? num_channels : ceil(Int, num_channels / 4) * 4 * 2^(i - 1)
    if index <= num_layers
        return MLOPF.GraphLayer(size(index - 1), size(index), act)
    end
    return MLOPF.FullyConnectedLayer(num_nodes * size(index - 1), size_out, fact)
end

define_layers(
    ::Type{MLOPF.Graph},
    size_in::Tuple,
    size_out::Int,
    num_layers::Int;
    act = Flux.relu,
    fact = Flux.sigmoid,
) = return map(l -> graph_layer(l, num_layers, size_in, size_out, act, fact), 1:(num_layers+1))

