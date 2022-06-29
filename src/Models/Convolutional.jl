using Flux
using LinearAlgebra
using MLOPF

struct Convolutional <: NeuralNetwork end

"""
    convolutional_neural_network(
        size_in::Tuple,
        size_out::Int64,
        num_layers::Int;
        drop_out::Float64 = 0.0,
        act::Function = Flux.relu,
        fact::Function = Flux.sigmoid,
        kernel::Tuple{Int64} = (3, 3),
        pad::Tuple{Int64} = (1, 1),
        pool::Tuple{Int64} = (2, 2),
        kwargs...,
    )

This function builds a convolutional graph as a Flux.jl chain type.
    
# Arguments:
    - `size_in::Tuple` -- Network input size (width, height, channels).
    - `size_out::Int` -- Network output size.
    - `num_layers::Int` -- Number of hidden layers.

# Keywords:
    - `drop_out::Float64` -- Probability assigned to drop out layer. Defaults to 0.
    - `act::Function` -- Activation function on hidden layers. Defaults to ReLU.
    - `fact::Function` -- Final activation function on output layer. Defaults to Sigmoid.
    - `kernel::Tuple{Int64}` -- Size of convolutional filter. Defaults to (3, 3).
    - `pad::Tuple{Int64}` -- Specifies the number of elements added around the image borders. Defaults to (1, 1).
    - `pool::Tuple{Int64}` -- Size of pooling layers used to reduce the dimensions of the feature maps. Defaults to (2, 2).

# Outputs
    - `Flux.Chain`: Convolutional neural network.
"""
function convolutional_neural_network(
    size_in::Tuple,
    size_out::Int64,
    num_layers::Int;
    drop_out::Float64=0.0,
    act=Flux.relu,
    fact=Flux.sigmoid,
    kernel::Int64=3,
    pad::Int64=1,
    pool::Int64=2,
    kwargs...
)
    width, _, channels = size_in
    size(i::Int) = i == 0 ? channels : Int(ceil(Int64, channels / 4) * 4 * 2^(i - 1))
    chain = []
    for i in 1:(num_layers)
        push!(chain, Flux.Conv((kernel, kernel), size(i - 1) => size(i), act; pad=(pad, pad), kwargs...))
        push!(chain, x -> Flux.maxpool(x, (pool, pool)))
        push!(chain, Flux.Dropout(drop_out))
        width = (1 + (width - kernel + 2pad)) / pool
    end
    push!(chain, x -> reshape(x, :, Base.size(x, 4)))
    push!(chain, Flux.Dense(Int(floor(width)^2) * size(num_layers), size_out, fact))
    return Flux.Chain(chain...)
end

function model_input(::Type{Convolutional}, data::Vector{Dict{String,Any}})
    return Float32.(
        cat(map(d -> cat(d["adjacency_matrix"], diagm(d["parameters"][pd.key]), diagm(d["parameters"][qd.key]), dims=3), data)..., dims=4))
end

function model_factory(::Type{Convolutional}, size_in::Union{Int64,Tuple}, size_out::Int64; num_layers::Int = 1, kwargs...)
    return convolutional_neural_network(size_in, size_out, num_layers; kwargs...)
end
