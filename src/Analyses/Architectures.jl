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
    using Flux
end

PowerModels.silence()

settings = MLOPF.get_settings()
for case in settings.PGLIB_OPF.cases

    network = PowerModels.parse_file(ENV["HOME"] * settings.PGLIB_OPF.path * "$(case).m")
    MLOPF.truncate!(network)
    data = MLOPF.cache(MLOPF.generate_samples, "./cache/samples/", "$(case).jld2")(
        network,
        settings.DATA.num_samples,
        settings.DATA.alpha,
    )

    target = settings.GENERAL.regression ? MLOPF.Primals : MLOPF.NonTrivialConstraints
    scaler = MLOPF.MinMaxScaler(data)
    data = @pipe data |> MLDataUtils.splitobs(_, at=Tuple(settings.DATA.splits)) |>
                 MLOPF.prepare_input_and_output(_, target, MLOPF.FullyConnected, MLOPF.Global, scaler) |>
                 MLOPF.prepare_minibatches(_, settings.PARAMETERS.batch_size)

    results = Dict()
    for seed in settings.GENERAL.seeds

        Random.seed!(seed)

        train_set, valid_set, test_set = deepcopy(data)
        X_train, y_train = train_set.data

        size_in, size_out = size(X_train, 1), size(y_train, 1)
        model = MLOPF.model_factory(
            MLOPF.FullyConnected, size_in, size_out, settings.PARAMETERS.num_layers, settings.PARAMETERS.drop_out)
        objective = target == MLOPF.Primals ? MLOPF.mse(@. !isnan(y_train[:, 1])) : MLOPF.bce()

        train_time, (train_loss, valid_loss) = MLOPF.train!(
            model,
            device,
            train_set,
            valid_set,
            objective;
            learning_rate=float(settings.PARAMETERS.learning_rate),
            num_epochs=settings.PARAMETERS.num_epochs
        )

        # TODO: We probably want to record the un-normalised test loss as well.
        test_time, test_loss = MLOPF.test(model, test_set, loss_func)
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
    MLOPF.cache(() -> results, "./cache/results/", "$(case).jld2")()
end

