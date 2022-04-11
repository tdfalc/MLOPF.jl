using Flux
using MLOPF

struct FullyConnected <: NeuralNetwork end

"""
    fully_connected_neural_network(
        drop_out::Float64;
        size_in::Tuple{int},
        size_out::int,
        num_layers::int;
        act::Function = Flux.relu,
        fact::Function = Flux.sigmoid,
        kwargs...,
    ) where {l<:GeometricFlux.AbstractGraphLayer}

This function builds a fully-connected neural network graph as a Flux.jl chain type.
    
# Arguments:
    - `Type{l}` -- Type of graph neural network layer from GeometrixFlux.jl package.
    - `size_in::int` -- Network input size (number of channels).
    - `size_out::int` -- Network output size.
    - `num_layers::int` -- Number of hidden layers.
    - `act::Function` -- Activation function (on hidden layer).
    - `fact::Function` -- Final activation function (on output layer).

# Outputs
    - `Flux.Chain`: Fully-connected neural network.
"""
function fully_connected_neural_network(
    size_in::int,
    size_out::int,
    drop_out::Float64,
    num_layers::int;
    act::Function = Flux.relu,
    fact::Function = Flux.sigmoid,
    kwargs...,
)
    chain = []
    for i ∈ 1:(num_layers)
        size(i) = floor(size_in + (i / num_layers) * (size_out - size_in))
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
