using Flux
using MLOPF

struct FullyConnected <: NeuralNetwork end

"""
    fully_connected_neural_network(
        size_in::Int,
        size_out::Int,
        num_layers::Int;
        drop_out::Float64 = 0.0,
        act::Function = Flux.relu,
        fact::Function = Flux.sigmoid,
        kwargs...,
    )

This function builds a fully-connected graph as a Flux.jl chain type.
    
# Arguments:
    - `size_in::Int` -- Input layer size.
    - `size_out::Int` -- Output layer size.
    - `num_layers::Int` -- Number of hidden layers.

# Keywords:
    - `drop_out::Float64` -- Probability assigned to drop out layer. Defaults to 0.
    - `act::Function` -- Activation function on hidden layers. Defaults to ReLU.
    - `fact::Function` -- Final activation function on output layer. Defaults to Sigmoid.

# Outputs
    - `Flux.Chain`: Fully-connected neural network.
"""
function fully_connected_neural_network(
    size_in::Int,
    size_out::Int,
    num_layers::Int;
    drop_out::Float64=0.3,
    act::Function=Flux.relu,
    fact::Function=Flux.sigmoid,
    kwargs...
)
    size(i::Int) = Int(floor(size_in + (i / (num_layers + 1)) * (size_out - size_in)))
    chain = []
    for i in 1:(num_layers)
        push!(chain, Flux.Dense(size(i - 1), size(i), act))
        push!(chain, Flux.Dropout(drop_out))
    end
    push!(chain, Flux.Dense(size(num_layers), size(num_layers + 1), fact))
    return Flux.Chain(chain...)
end

function model_input(::Type{FullyConnected}, data::Vector{Dict{String,Any}})
    return Float32.(hcat(map(d -> [d["parameters"][pd.key]..., d["parameters"][qd.key]...], data)...))
end

function model_factory(::Type{FullyConnected}, size_in::Union{Int64,Tuple}, size_out::Int; num_layers::Int = 1, kwargs...)
    return fully_connected_neural_network(size_in, size_out, num_layers; kwargs...)
end
