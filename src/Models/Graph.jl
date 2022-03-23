using Flux
using GeometricFlux
using MLOPF

struct Graph <: NeuralNetwork end
struct 

struct GraphLayer
    in::Int
    out::Int
    act::Function
end

