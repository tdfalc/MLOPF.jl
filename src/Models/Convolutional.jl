using Flux
using LinearAlgebra
using MLOPF

struct Convolutional <: NeuralNetwork end

struct ConvolutionalLayer
    in::Int
    out::Int
    act::Function
end

"""
    convolutional_neural_network(layers::Vector{MLOPF.Layer}, drop_out::Float64)

This function builds a convolutional neural network graph using Flux's Chain type.
    
# Arguments:
    - `layers::Vector -- Layers defining the neural network architecture.
    - `drop_out::Float64` -- Probability assigned to drop out layer.
    - `kernel::Tuple{Int}` -- Size of convolutional filter.
    - `pad::Tuple{Int}` -- Specifies the number of elements added around the image borders.
    - `pool::Tuple{Int}` -- Size of pooling layers used to reduce the dimensions of the feature maps.
    
# Outputs
    - `Flux.Chain`: Convolutional neural network.
"""
function convolutional_neural_network(
    layers::Vector,
    drop_out::Float64,
    kernel::Tuple{Int},
    pad::Tuple{Int},
    pool::Tuple{Int},
)
    graph = []
    for layer ∈ layers
        if isa(layer, ConvolutionalLayer)
            push!(graph, Flux.Conv(kernel, layer.in => layer.out, layer.act; pad = pad))
            push!(graph, x -> Flux.maxpool(x, pool))
            push!(graph, Flux.BatchNorm(layer.out))
            push!(graph, Flux.Dropout(drop_out))
        else
            push!(graph, x -> reshape(x, :, size(x, 4)))
            push!(graph, Flux.Dense(layer.in, layer.out, layer.act))
        end
    end
    return Flux.Chain(graph...)
end

function model_input(::Type{Convolutional}, data::Vector{MLOPF.ProcessedSample})
    return cat(map(d -> cat(d.adjacency_matrix, diagm(d.pd), diagm(d.qd), dims = 3), data)..., dims = 4)
end

function calculate_reshape_size(height::Int, width::Int, num_layers::Int, num_channels::Int, pool_size::Int)
    return Int(prod(vcat(floor.((height + width) / pool_size^(num_layers)), num_channels)))
end

function convolutional_layer(index::Int, num_layers::Int, size_in::Tuple{Int}, size_out::Int, act, fact)
    height, width, num_channels = size_in
    size(i) = i == 0 ? num_channels : ceil(Int, num_channels / 4) * 4 * 2^(i - 1)
    if index <= num_layers
        return MLOPF.ConvolutionalLayer(size(index - 1), size(index), act)
    end
    reshape_size = calculate_reshape_size(height, width, num_layers, size(index - 1), 2)
    return MLOPF.FullyConnectedLayer(reshape_size, size_out, fact)
end

define_layers(
    ::Type{MLOPF.Convolutional},
    size_in::Tuple{Int},
    size_out::Int,
    num_layers::Int;
    act = Flux.relu,
    fact = Flux.sigmoid,
) = return map(l -> convolutional_layer(l, num_layers, size_in, size_out, act, fact), 1:(num_layers+1))


build_model(
    ::Type{Convolutional},
    layers::Vector,
    drop_out::Float64,
    kernel::Tuple{Int},
    pad::Tuple{Int},
    pool::Tuple{Int},
) = convolutional_neural_network(layers::Vector{MLOPF.ConvolutionalLayer}, drop_out, kernel, pad, pool)

