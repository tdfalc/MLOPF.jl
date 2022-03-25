using Flux
using GeometricFlux
using MLOPF

struct Graph <: NeuralNetwork end
#struct ChebConv <: Graph end

struct GraphLayer
    in::Int
    out::Int
    act::Function
end

function model_input(::Type{Graph}, data::Vector{MLOPF.ProcessedSample})
    return map(x -> FeaturedGraph(x.adjacency_matrix, nf = hcat([x.pd, x.qd]...)'), data)
end
