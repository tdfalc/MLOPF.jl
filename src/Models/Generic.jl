"""Generic methods used to build, train and test each neural network architecture"""

using Base: @kwdef
using Flux
using Flux.Data: DataLoader
using CUDA
using GraphSignals
using Statistics

abstract type Target end
struct Primals <: Target end
struct NonTrivialConstraints <: Target end

abstract type Encoding end
struct Global <: Encoding end
struct Local <: Encoding end

abstract type NeuralNetwork end

function prepare_minibatches(data::Vector, batch_size::Int=10, shuffle::Bool=true)
    return map(d -> DataLoader(d; batchsize=batch_size, shuffle=shuffle), data)
end

function prepare_input_and_output(
    train_set,
    valid_set,
    test_set,
    target::Type{T},
    arch::Type{A},
    encoding::Type{E},
    scaler::MLOPF.MinMaxScaler
) where {T<:Target,A<:NeuralNetwork,E<:Encoding}
    return [
        (
            MLOPF.model_input(arch, MLOPF.normalise(Vector(set), scaler, (pd, qd))),
            try
                MLOPF.model_output(encoding, target, MLOPF.normalise(Vector(set), scaler, (pg, vm)))
            catch
                nt_constraints = non_trivial_constraints(Vector(train_set))
                MLOPF.model_output(encoding, target, MLOPF.normalise(Vector(set), scaler, (pg, vm)), nt_constraints)
            end
        ) for set in (train_set, valid_set, test_set)
    ]
end

function model_output(::Union{Type{Global},Type{Local}}, ::Type{Primals}, data::Vector{Dict{String,Any}})
    return Float32.(hcat(map(d -> [d["parameters"][pg.key]..., d["parameters"][vm.key]...], data)...))
end

function non_trivial_constraints(data::Vector{Dict{String,Any}})
    congestion_regimes = hcat([d["congestion_regime"] for d in data]...)
    activation_count = sum(congestion_regimes, dims=2)
    return (activation_count .> 0) .& (activation_count .< length(data))
end

function model_output(
    ::Type{Global}, ::Type{NonTrivialConstraints}, data::Vector{Dict{String,Any}}, non_trivial_constraints::BitMatrix)
    congestion_regimes = hcat([d["congestion_regime"] for d in data]...)
    return Float32.(congestion_regimes[non_trivial_constraints[:], :])
end

"Custom bce - weight adjusted binary crossentropy to account for class imbalance."
function bce(; weight::Float64=0.5)
    return (y, ŷ; ϵ=eps(Float64)) -> mean(@. -y * log(ŷ + ϵ) * weight - (1 - y) * log(1 - ŷ + ϵ) * (1 - weight))
end

"Custom mse - initalised with bit vector mask to remove redunant rows when evaluating loss."
function mse(mask::BitVector)
    return (y, ŷ) -> Statistics.mean(
        ((y, ŷ) -> sum((y[mask] - ŷ[mask]) .^ 2) / sum(mask)).((eachcol.((y, ŷ)))...),
    )
end

function instantiate_device(use_cuda::Bool)
    if CUDA.functional() && use_cuda
        CUDA.allowscaler(false)
        return gpu
    end
    return cpu
end

function train!(
    model::Flux.Chain,
    device::Function,
    train_set::DataLoader,
    valid_set::DataLoader,
    objective::Function;
    num_epochs::Int=1,
    learning_rate::Float64=1e-3
)

    model = model |> device
    @info "commencing training procedure on $(device)"

    function eval(X::Array, y::Array)
        return objective(y, model(X))
    end

    function eval(X::Vector{<:FeaturedGraph}, y::Array)
        return objective(y, hcat(model.(X)...))
    end

    callback = () -> push!(losses, [eval(train_set.data...); eval(valid_set.data...)])
    losses = []

    prog = Progress(num_epochs; showspeed=true)
    elapsed_time = @elapsed begin
        opt, θ = ADAM(learning_rate), Flux.params(model)
        for _ = 1:num_epochs
            for (X, y) in train_set
                X, y = X |> device, y |> device
                gradients = Flux.gradient(θ) do
                    eval(X, y)
                end
                Flux.update!(opt, θ, gradients)
            end
            Flux.testmode!(model)
            callback()
            train_loss, valid_loss = losses[end]
            ProgressMeter.next!(prog; showvalues=[(:train_loss, train_loss), (:valid_loss, valid_loss)])
        end
    end
    return elapsed_time, collect.(zip(losses...))
end

function test(model::Flux.Chain, test_set::DataLoader, objective)
    Flux.testmode!(model)
    loss = []
    elapsed_time = @elapsed begin
        for (X, y) in test_set
            ŷ = try
                model(X)
            catch
                hcat(model.(X)...)
            end
            append!(loss, objective(y, ŷ))
        end
    end
    return elapsed_time, sum(loss) / length(loss)
end
