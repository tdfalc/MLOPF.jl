"""Run basic pipeline: generate and process samples, then build, fit and evaluate spcecified architectures.

Notes:
    * Starting with julia -p n provides n worker processes on local machine.
"""

using Distributed

@everywhere begin
    using Pkg
    Pkg.activate(".")
end

@everywhere begin
    using MLOPF
    using PowerModels
    using MLDataUtils
    using Random
    using Pipe
    using Flux
end

PowerModels.silence()

settings = MLOPF.get_settings()
for case in settings.PGLIB_OPF.cases

    @info "running Architectures.jl analysis for cases $(case)"
    network = PowerModels.parse_file(ENV["HOME"] * settings.PGLIB_OPF.path * "$(case).m")
    MLOPF.truncate!(network)
    data = MLOPF.cache(MLOPF.generate_data, "./cache/data/", "$(case).jld2")(
        network,
        settings.DATA.num_samples,
        settings.DATA.alpha,
    )
    scaler = MLOPF.MinMaxScaler(data)

    for architecture in [MLOPF.FullyConnected, MLOPF.Convolutional, MLOPF.Graph]

        if isa(architecture, Type{MLOPF.FullyConnected}) & !settings.PARAMETERS.FULLY_CONNECTED.active
            continue
        end
        if isa(architecture, Type{MLOPF.Convolutional}) & !settings.PARAMETERS.CONVOLUTIONAL.active
            continue
        end
        if isa(architecture, Type{MLOPF.Graph}) & !settings.PARAMETERS.GRAPH.active
            continue
        end

        if settings.PARAMETERS.GLOBAL.local
            encoding = MLOPF.Local
        else
            encoding = MLOPF.Global
        end

        if settings.PARAMETERS.GLOBAL.regression
            target = MLOPF.Primals
        else
            target = MLOPF.NonTrivialConstraints
        end

        @info "building dataset for $(architecture.name.name) architecture with $(encoding.name.name) encoding and $(target.name.name) target"
        let data = @pipe MLDataUtils.splitobs(data, at=Tuple(settings.DATA.splits)) |>
                         MLOPF.prepare_input_and_output(_..., target, architecture, encoding, scaler) |>
                         MLOPF.prepare_minibatches(_, settings.PARAMETERS.GLOBAL.batch_size)

            results = Dict()
            for seed in settings.GENERAL.seeds

                Random.seed!(seed)

                train_set, valid_set, test_set = deepcopy(data)
                X_train, y_train = train_set.data

                parse_kwargs(config) = Dict{Symbol,Any}(pairs(config))

                if isa(architecture, Type{MLOPF.FullyConnected})
                    size_in = size(X_train, 1)
                    kwargs = parse_kwargs(settings.PARAMETERS.FULLY_CONNECTED.CONFIG)
                end
                if isa(architecture, Type{MLOPF.Convolutional})
                    size_in = size(X_train[:, :, :, 1])
                    kwargs = parse_kwargs(settings.PARAMETERS.CONVOLUTIONAL.CONFIG)
                end
                if isa(architecture, Type{MLOPF.Graph})
                    size_in = size(X_train[1].nf)
                    kwargs = parse_kwargs(settings.PARAMETERS.GRAPH.CONFIG)
                    kwargs[:encoding] = encoding
                end

                size_out = size(y_train, 1)

                model = MLOPF.model_factory(architecture, size_in, size_out; kwargs...)
                objective = target == MLOPF.Primals ? MLOPF.mse(@. !isnan(y_train[:, 1])) : MLOPF.bce()
                device = MLOPF.instantiate_device(settings.PARAMETERS.GLOBAL.use_cuda)

                train_time, (train_loss, valid_loss) = MLOPF.train!(
                    model,
                    device,
                    train_set,
                    valid_set,
                    objective;
                    learning_rate=float(settings.PARAMETERS.GLOBAL.learning_rate),
                    num_epochs=settings.PARAMETERS.GLOBAL.num_epochs
                )

                # TODO: We probably want to record the un-normalised test loss as well.
                test_time, test_loss = MLOPF.test(model, test_set, objective)
                results[seed] = Dict(
                    "target" => MLOPF.Primals,
                    "encoding" => MLOPF.Global,
                    "architecture" => MLOPF.FullyConnected,
                    "train_loss" => train_loss,
                    "valid_loss" => valid_loss,
                    "test_loss" => test_loss,
                    "train_time" => train_time,
                    "test_time" => test_time,
                    "trainable_parameters" => sum(length(p) for p in Flux.params(model)),
                )

            end
            MLOPF.cache(
                () -> results,
                "./cache/results/",
                "$(case)_$(architecture.name.name)_$(encoding.name.name)_$(target.name.name).jld2"
            )()
        end
    end
end

