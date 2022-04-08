using Flux
using LinearAlgebra
using MLOPF

struct Convolutional <: NeuralNetwork end
struct ConvolutionalLayer <: NeuralNetworkLayer end

"""
    convolutional_neural_network(layers::Vector{MLOPF.Layer}, drop_out::Float64)

This function builds a convolutional neural network graph using Flux's Chain type.
    
# Arguments:
    - `layers::Vector -- Layers defining the neural network architecture.

# Keywords
    - `drop_out::Float64{}` -- Probability assigned to drop out layer. Defaults to 0.
    - `kernel::Tuple{Int}` -- Size of convolutional filter. Defaults to (3, 3).
    - `pad::Tuple{Int}` -- Specifies the number of elements added around the image borders. Defaults to (1, 1).
    - `pool::Tuple{Int}` -- Size of pooling layers used to reduce the dimensions of the feature maps. Defaults to (2, 2).
     
# Outputs
    - `Flux.Chain`: Convolutional neural network.
"""
function convolutional_neural_network(
    layers::Vector,
    drop_out::Float64;
    kernel::Tuple{Int} = (3, 3),
    pad::Tuple{Int} = (1, 1),
    pool::Tuple{Int} = (2, 2),
)
    chain = []
    for layer ∈ layers
        if isa(layer, ConvolutionalLayer)
            push!(chain, Flux.Conv(kernel, layer.in => layer.out, layer.act; pad = pad))
            push!(chain, x -> Flux.maxpool(x, pool))
            push!(chain, Flux.BatchNorm(layer.out))
            push!(chain, Flux.Dropout(drop_out))
        else
            push!(chain, x -> reshape(x, :, size(x, 4)))
            push!(chain, Flux.Dense(layer.in, layer.out, layer.act))
        end
    end
    return Flux.Chain(chain...)
end

function model_input(::Type{Convolutional}, data::Vector{MLOPF.ProcessedSample})
    return cat(map(d -> cat(d.adjacency_matrix, diagm(d.pd), diagm(d.qd), dims = 3), data)..., dims = 4)
end

function convolutional_layer(index::Int, num_layers::Int, size_in::Tuple{Int}, size_out::Int, act, fact)
    height, width, num_channels = size_in
    size(i) = i == 0 ? num_channels : ceil(Int, num_channels / 4) * 4 * 2^(i - 1)
    if index <= num_layers
        return MLOPF.ConvolutionalLayer(size(index - 1), size(index), act)
    end
    return MLOPF.FullyConnectedLayer(
        Int(prod(vcat(floor.((height + width) / pool_size^(num_layers)), num_channels))),
        size_out,
        fact,
    )
end

define_layers(
    ::Type{MLOPF.Convolutional},
    size_in::Tuple{Int},
    size_out::Int,
    num_layers::Int;
    act = Flux.relu,
    fact = Flux.sigmoid,
) = return map(l -> convolutional_layer(l, num_layers, size_in, size_out, act, fact), 1:(num_layers+1))

build_model(::Type{Convolutional}, layers::Vector; kwargs...) =
    convolutional_neural_network(layers::Vector{MLOPF.ConvolutionalLayer}; kwargs...)

