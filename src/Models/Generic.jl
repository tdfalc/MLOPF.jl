"""Generic functions used to build, train and test each neural network architecture"""

using Base: @kwdef
using Flux
using Flux.Data: DataLoader
using CUDA
using Statistics

abstract type Target end
struct Primals <: Target end
struct NonTrivialConstraints <: Target end

abstract type Encoding end
struct Global <: Encoding end
struct Local <: Encoding end

abstract type NeuralNetwork end

struct NeuralNetworkLayer
    in::Int
    out::Int
    act::Function
end

function model_output(::Type{Global}, ::Type{Primals}, data::Vector{MLOPF.ProcessedSample})
    return hcat(map(d -> [d.pg..., d.vm...], data)...)
end

function model_output(::Type{Local}, ::Type{Primals}, data::Vector{MLOPF.ProcessedSample})
    return hcat(map(d -> hcat(d.pg, d.vm)', data))
end

function model_output(::Type{Global}, ::Type{NonTrivialConstraints}, data::Vector{MLOPF.ProcessedSample})
    # TODO: I need to move the determination of non-trivial constraints back to the training
    # TODO: set to avoid this data leakage.
    congestion_regimes = hcat([sample.regime for sample in data]...)
    # First we count the number of times each constraint is binding.
    activation_count = sum(congestion_regimes, dims = 2)
    # Then we flag constraints that change binding status atleast once.
    non_trivial_constraints = (activation_count .> 0) .& (activation_count .< length(data))
    return congestion_regimes[non_trivial_constraints[:], :]
end

function build_minibatches(data::Tuple, batch_size::Int, shuffle::Bool)
    return map(d -> DataLoader(d; batchsize = batch_size, shuffle = shuffle), data)
end

"Custom bce - weight adjusted binary crossentropy to account for class imbalance."
function weighted_binary_crossentropy(weight::Float64)
    return (y, ŷ; ϵ = eps(ŷ)) -> -y * log(ŷ + ϵ) * weight - (1 - y) * log(1 - ŷ + ϵ) * (1 - weight)
end

"Custom mse - initalised with bit vector mask to remove redunant rows when evaluating loss."
function mean_squared_error(mask::BitVector)
    return (y, ŷ) -> Statistics.mean(
        ((y, ŷ) -> sum((y[mask] - ŷ[mask]) .^ 2) / size(y[mask], 2)).((eachcol.((y, ŷ)))...),
    )
end

function train!(
    model::Flux.Chain{},
    train_set::DataLoader,
    valid_set::DataLoader,
    objective,
    num_epochs::Int,
    learning_rate::Float64,
    use_cuda::Bool,
)

    if CUDA.functional() && use_cuda
        CUDA.allowscaler(false)
        device = gpu
    else
        device = cpu
    end

    model = model |> device
    opt, θ = ADAM(learning_rate), Flux.params(model)
    @info "commencing training procedure on $(device)"

    losses, eval = [], (X, y) -> objective(Matrix(y), model(X))
    callback = () -> push!(losses, [eval(train_set.data...); eval(valid_set.data...)])

    prog = Progress(anum_epochs; showspeed = true)
    elapsed_time = @elapsed begin
        for _ = 1:num_epochs
            for (X, y) in train_set
                X, y = X |> device, y |> device
                gradients = Flux.gradient(θ) do
                    eval(model(X), y)
                end
                Flux.update!(opt, θ, gradients)
            end
            Flux.testmode!(model)
            callback()
            train_loss, valid_loss = losses[end]
            ProgressMeter.next!(prog; showvalues = [(:train_loss, train_loss), (:valid_loss, valid_loss)])
        end
    end
    return elapsed_time, collect.(zip(losses...))
end

function test(model::Flux.Chain, test_set::DataLoader, objective)
    Flux.testmode!(model)
    loss = []
    elapsed_time = @elapsed begin
        for (X, y) in test_set
            ŷ = model(X)
            append!(loss, objective(y, ŷ))
        end
    end
    return elapsed_time, sum(loss) / length(loss)
end

function model_factory()
    return "model"
end
