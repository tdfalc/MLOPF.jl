using Flux
using GeometricFlux
using MLOPF

struct Graph <: NeuralNetwork end
struct ChebConv <: Graph end

struct GraphLayer
    in::Int
    out::Int
    act::Function
end

