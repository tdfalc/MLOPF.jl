using Flux
using MLOPF

struct FullyConnected <: NeuralNetwork end

"""
    fully_connected_neural_network(layers::Vector{MLOPF.Layer}, drop_out::Float64)

This function builds a fully-connected neural network graph using Flux's Chain type.
    
# Arguments:
    - `layers::Vector{MLOPF.Layer} -- Layers defining the neural network architecture.
    - `drop_out::Float64{}` -- Probability assigned to drop out layer.
    
# Outputs
    - `Flux.Chain`: Fully-connected neural network.
"""
function fully_connected_neural_network(layers::Vector{MLOPF.Layer}, drop_out::Float64)
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

function fcnn_input(data)
    return hcat(map(d -> [d.pd..., d.qd...], data)...)
end

function fcnn_layer(index, num_layers, size_in, size_out, act, fact)
    size(i) = floor(size_in + (i / num_layers) * (size_out - size_in))
    return Layer(size(index - 1), size(index), index < num_layers ? act : fact)
end

function fcnn_layers(size_in, size_out, num_layers; act = Flux.relu, fact = Flux.sigmoid)
    return map(l -> fcnn_layer(l, num_layers, size_in, size_out, act, fact), 1:num_layers)
end

define_layers(::Type{FullyConnected}, size_in, size_out, num_layers; act = Flux.relu, fact = Flux.sigmoid) =
    fcnn_layers(size_in, size_out, num_layers; act = act, fact = fact)

build_model(::Type{FullyConnected}, layers, drop_out) =
    fully_connected_neural_network(layers::Vector{MLOPF.Layer}, drop_out)

model_input(::Type{FullyConnected}, data) = fcnn_input(data)
