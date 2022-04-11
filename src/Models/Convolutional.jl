using Flux
using LinearAlgebra
using MLOPF

struct Convolutional <: NeuralNetwork end

"""
    convolutional_neural_network(
        size_in::Tuple{Int},
        size_out::Int,
        num_layers::Int;
        drop_out::Float64 = 0.0,
        act::Function = Flux.relu,
        fact::Function = Flux.sigmoid,
        kernel::Tuple{Int} = (3, 3),
        pad::Tuple{Int} = (1, 1),
        pool::Tuple{Int} = (2, 2),
        kwargs...,
    )

This function builds a convolutional neural network graph as a Flux.jl chain type.
    
# Arguments:
    - `size_in::Int` -- Network input size (number of channels).
    - `size_out::Int` -- Network output size.
    - `num_layers::Int` -- Number of hidden layers.

# Keywords:
    - `drop_out::Float64` -- Probability assigned to drop out layer. Defaults to 0.
    - `act::Function` -- Activation function (on hidden layer). Defaults to ReLU.
    - `fact::Function` -- Final activation function (on output layer). Defaults to Sigmoid.
    - `kernel::Tuple{Int}` -- Size of convolutional filter. Defaults to (3, 3).
    - `pad::Tuple{Int}` -- Specifies the number of elements added around the image borders. Defaults to (1, 1).
    - `pool::Tuple{Int}` -- Size of pooling layers used to reduce the dimensions of the feature maps. Defaults to (2, 2).

# Outputs
    - `Flux.Chain`: Convolutional neural network.
"""
function convolutional_neural_network(
    size_in::Tuple{Int},
    size_out::Int,
    num_layers::Int;
    drop_out::Float64 = 0.0,
    act::Function = Flux.relu,
    fact::Function = Flux.sigmoid,
    kernel::Tuple{Int} = (3, 3),
    pad::Tuple{Int} = (1, 1),
    pool::Tuple{Int} = (2, 2),
    kwargs...,
)
    size(i::Int) = i == 0 ? size_in : ceil(Int, size_in / 4) * 4 * 2^(i - 1)
    chain = []
    for i ∈ 1:(num_layers)
        push!(chain, Flux.Conv(kernel, size(i - 1) => size(i), act; pad = pad, kwargs...))
        push!(chain, x -> Flux.maxpool(x, pool))
        push!(chain, x -> Flux.BatchNorm(size(x))(x))
        push!(chain, Flux.Dropout(drop_out))
    end
    push!(chain, x -> reshape(x, :, size(x, 4)))
    push!(chain, x -> Flux.Dense(size(x), size_out, fact))(x)
    return Flux.Chain(chain...)
end

function model_input(::Type{Convolutional}, data::Vector{MLOPF.ProcessedSample})
    return cat(map(d -> cat(d.adjacency_matrix, diagm(d.pd), diagm(d.qd), dims = 3), data)..., dims = 4)
end

function model_factory(::Type{Convolutional}, size_in::Int, size_out::Int, num_layers::Int; kwargs...)
    return convolutional_neural_network(size_in, size_out, num_layers; kwargs...)
end
