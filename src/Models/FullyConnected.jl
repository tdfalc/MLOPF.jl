using Flux
using MLOPF

struct FullyConnected <: NeuralNetwork end

struct FullyConnectedLayer
    in::Int
    out::Int
    act::Function
end

"""
    fully_connected_neural_network(layers::Vector{MLOPF.Layer}, drop_out::Float64)

This function builds a fully-connected neural network graph using Flux's Chain type.
    
# Arguments:
    - `layers::Vector{MLOPF.FullyConnectedLayer} -- Layers defining the neural network architecture.
    - `drop_out::Float64{}` -- Probability assigned to drop out layer.
    
# Outputs
    - `Flux.Chain`: Fully-connected neural network.
"""
function fully_connected_neural_network(layers::Vector{MLOPF.FullyConnectedLayer}, drop_out::Float64)
    graph = []
    for (i, layer) in enumerate(layers)
        push!(graph, Flux.Dense(layer.in, layer.out, layer.act))
        if i < length(layers)
            push!(graph, Flux.BatchNorm(layer.out))
            push!(graph, Flux.Dropout(drop_out))
        end
    end
    return Flux.Chain(graph...)
end

function model_input(::Type{FullyConnected}, data::Vector{MLOPF.ProcessedSample})
    return hcat(map(d -> [d.pd..., d.qd...], data)...)
end

function fully_connected_layer(index::Int, num_layers::Int, size_in::Int, size_out::Int, act, fact)
    size(i) = floor(size_in + (i / num_layers) * (size_out - size_in))
    return FullyConnectedLayer(size(index - 1), size(index), index < num_layers ? act : fact)
end

define_layers(
    ::Type{FullyConnected},
    size_in::Int,
    size_out::Int,
    num_layers::Int;
    act = Flux.relu,
    fact = Flux.sigmoid,
) = return map(l -> fully_connected_layer(l, num_layers, size_in, size_out, act, fact), 1:num_layers)

build_model(::Type{FullyConnected}, layers::Vector{FullyConnectedLayer}, drop_out::Float64) =
    fully_connected_neural_network(layers::Vector{MLOPF.FullyConnectedLayer}, drop_out)

