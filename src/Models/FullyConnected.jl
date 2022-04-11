using Flux
using MLOPF

struct FullyConnected <: NeuralNetwork end

"""
    fully_connected_neural_network(
        size_in::Tuple{Int},
        size_out::Int,
        num_layers::Int;
        drop_out::Float64 = 0.0,
        act::Function = Flux.relu,
        fact::Function = Flux.sigmoid,
        kwargs...,
    )

This function builds a fully-connected neural network graph as a Flux.jl chain type.
    
# Arguments:
    - `size_in::Int` -- Network input size (number of channels).
    - `size_out::Int` -- Network output size.
    - `num_layers::Int` -- Number of hidden layers.

# Keywords:
    - `drop_out::Float64` -- Probability assigned to drop out layer. Defaults to 0.
    - `act::Function` -- Activation function (on hidden layer). Defaults to ReLU.
    - `fact::Function` -- Final activation function (on output layer). Defaults to Sigmoid.

# Outputs
    - `Flux.Chain`: Fully-connected neural network.
"""
function fully_connected_neural_network(
    size_in::Int,
    size_out::Int,
    num_layers::Int;
    drop_out::Float64 = 0.0,
    act::Function = Flux.relu,
    fact::Function = Flux.sigmoid,
    kwargs...,
)
    size(i::Int) = floor(size_in + (i / num_layers) * (size_out - size_in))
    chain = []
    for i ∈ 1:(num_layers)
        push!(chain, Flux.Dense(size(i - 1), size(i), act), kwargs...)
        push!(chain, x -> Flux.BatchNorm(size(x))(x))
        push!(chain, Flux.Dropout(drop_out))
    end
    push!(chain, Flux.Dense(size(num_layers), size(num_layers + 1), fact))
    return Flux.Chain(chain...)
end

function model_input(::Type{FullyConnected}, data::Vector{MLOPF.ProcessedSample})
    return hcat(map(d -> [d.pd..., d.qd...], data)...)
end

function model_factory(::Type{FullyConnected}, size_in::Int, size_out::Int, num_layers::Int; kwargs...)
    return fully_connected_neural_network(size_in, size_out, num_layers; kwargs...)
end
