using Flux
using LinearAlgebra
using MLOPF

struct Convolutional <: NeuralNetwork end

"""
    convolutional_neural_network(layers::Vector{MLOPF.Layer}, drop_out::Float64)

This function builds a convolutional neural network graph using Flux's Chain type.
    
# Arguments:
    - `layers::Vector{MLOPF.Layer} -- Layers defining the neural network architecture.
    - `drop_out::Float64{}` -- Probability assigned to drop out layer.
    
# Outputs
    - `Flux.Chain`: Convolutional neural network.
"""
function convolutional_neural_network(layers::Vector{MLOPF.Layer}, drop_out::Float64)
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

function model_input(::Type{Convolutional}, data::Vector{MLOPF.ProcessedSample})
    return cat(map(d -> cat(d.adjacency_matrix, diagm(d.pd), diagm(d.qd), dims = 3), data)..., dims = 4)
end

function convolutional_layer(index, num_layers, size_in, size_out, act, fact)
    size(i) = floor(size_in + (i / num_layers) * (size_out - size_in))
    return Layer(size(index - 1), size(index), index < num_layers ? act : fact)
end

define_layers(::Type{Convolutional}, size_in, size_out, num_layers; act = Flux.relu, fact = Flux.sigmoid) =
    return map(l -> convolutional_layer(l, num_layers, size_in, size_out, act, fact), 1:num_layers)

build_model(::Type{Convolutional}, layers, drop_out) =
    convolutional_neural_network(layers::Vector{MLOPF.Layer}, drop_out)

