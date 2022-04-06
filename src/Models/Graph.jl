using Flux
using GeometricFlux
using MLOPF

abstract type Graph <: NeuralNetwork end
abstract type ChebConv <: Graph end

struct GraphLayer
    in::Int
    out::Int
    act::Function
end

function graph_neural_network(layers::Vector, drop_out::Float64, kernel::Int)
    graph = []
    for layer ∈ layers
        if isa(layer, MLOPF.GraphLayer)
            push!(graph, GeometricFlux.ChebConv(layer.in => layer.out, kernel))
            # TODO: Figure out how to add batch norm to GNN
            #push!(graph, Flux.BatchNorm(layer.out))
            push!(graph, Flux.Dropout(drop_out))
        else
            push!(graph, x -> vec(x.nf'))
            push!(graph, Flux.Dense(layer.in, layer.out, layer.act))
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

