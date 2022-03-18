using Base: @kwdef
using Flux
using Flux.Data: DataLoader
using CUDA
using Statistics

abstract type Target end
struct Primals <: Target end

abstract type Encoding end
struct Global <: Encoding end

abstract type NeuralNetwork end

struct Layer
    in::Int
    out::Int
    act::Function
end

function model_output(::Type{Global}, ::Type{Primals}, data)
    return hcat(map(d -> [d.pg..., d.vm...], data)...)
end

function build_minibatches(data::Tuple, batch_size::Int, shuffle::Bool)
    return map(d -> DataLoader(d; batchsize = batch_size, shuffle = shuffle), data)
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
            append!(loss, loss_func(y, model(ŷ)))
        end
    end
    return elapsed_time, sum(loss) / length(loss)
end