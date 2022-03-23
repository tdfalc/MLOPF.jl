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

abstract type NeuralNetwork end

function model_output(::Type{Global}, ::Type{Primals}, data::Vector{MLOPF.ProcessedSample})
    return hcat(map(d -> [d.pg..., d.vm...], data)...)
end

function model_output(::Type{Global}, ::Type{NonTrivialConstraints}, data::Vector{MLOPF.ProcessedSample})
    # TODO: We need to move the evaluation of non-trivial constraints back to the training set to avoid
    # data leakage.
    congestion_regimes = hcat([sample.regime for sample in data]...)
    # First we count the number of times each constraint is binding across the whole set.
    activation_count = sum(congestion_regimes, dims = 2)
    # Then we flag constraints that change binding status atleast once across the whole set.
    non_trivial_constraints = (activation_count .> 0) .& (activation_count .< length(data))
    return congestion_regimes[non_trivial_constraints[:], :]
ends

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

@kwdef mutable struct Args
    η::Float64
    num_epochs::Int
    use_cuda::Bool
end

function train!(model::Flux.Chain{}, train_set::DataLoader, valid_set::DataLoader, loss_func; kwargs...)
    args = Args(; kwargs...)

    if CUDA.functional() && args.use_cuda
        CUDA.allowscaler(false)
        device = gpu
    else
        device = cpu
    end

    model = model |> device
    opt, θ = ADAM(args.η), Flux.params(model)
    @info "commencing training procedure on $(device)"

    losses = []
    eval = (X, y) -> loss_func(Matrix(y), model(X))
    callback = () -> push!(losses, [eval(train_set.data...); eval(valid_set.data...)])

    prog = Progress(args.num_epochs; showspeed = true)
    elapsed_time = @elapsed begin
        for _ = 1:args.num_epochs
            for (X, y) in train_set
                X, y = X |> device, y |> device
                gradients = Flux.gradient(θ) do
                    ŷ = model(X)
                    loss_func(y, ŷ)
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

function test(model::Flux.Chain, test_set::DataLoader, loss_func)
    Flux.testmode!(model)
    loss = []
    elapsed_time = @elapsed begin
        for (X, y) in test_set
            ŷ = model(X)
            append!(loss, loss_func(y, ŷ))
        end
    end
    return elapsed_time, sum(loss) / length(loss)
end

function model_factory()
    return "model"
end