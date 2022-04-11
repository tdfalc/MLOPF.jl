using Flux
using LinearAlgebra
using MLOPF

struct Convolutional <: NeuralNetwork end

"""
    convolutional_neural_network(
        drop_out::Float64;
        size_in::Tuple{int},
        size_out::int,
        num_layers::int;
        kernel::Tuple{Int} = (3, 3),
        pool::Tuple{Int} = (2, 2),
        act::Function = Flux.relu,
        fact::Function = Flux.sigmoid,
        kwargs...,
    ) where {l<:GeometricFlux.AbstractGraphLayer}

This function builds a convolutional neural network graph as a Flux.jl chain type.
    
# Arguments:
    - `Type{l}` -- Type of graph neural network layer from GeometrixFlux.jl package.
    - `size_in::Tuple{int}` -- Network input size (number of channels).
    - `size_out::int}` -- Network output size.
    - `num_layers::int` -- Number of hidden layers.
    - `act::Function` -- Activation function (on hidden layer).
    - `fact::Function` -- Final activation function (on output layer).

# Outputs
    - `Flux.Chain`: Convolutional neural network.
"""
function convolutional_neural_network(
    drop_out::Float64;
    size_in::Int,
    size_out::int,
    num_layers::int;
    kernel::Tuple{Int} = (3, 3),
    pool::Tuple{Int} = (2, 2),
    act::Function = Flux.relu,
    fact::Function = Flux.sigmoid,
    kwargs...,
)
    chain = []
    for i ∈ 1:(num_layers)
        size(i) = i == 0 ? size_in : ceil(Int, size_in / 4) * 4 * 2^(i - 1)
        push!(chain, Flux.Conv(kernel, size(i - 1) => size(i), act; kwargs...))
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
