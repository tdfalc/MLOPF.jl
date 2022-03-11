"""Run basic pipeline: generate and process samples, then build, fit and evaluate spcecified architectures.

Notes:
    - Starting with julia -p n provides n worker processes on local machine.
"""

using Distributed
using PowerModels
using FileIO
using JLD2
using Random

PowerModels.silence()

@everywhere begin
    using Pkg
    Pkg.activate(".")
end

@everywhere begin
    using MLOPF
    using Flux.Data: DataLoader
    using MLDataUtils
end

settings = MLOPF.get_settings()
for case in settings.PGLIB_OPF.cases

    # Parse synthetic grid file (*.m) to Powermodels.jl network.
    network = PowerModels.parse_file(ENV["HOME"] * settings.PGLIB_OPF.path * "$(case).m")
    MLOPF.truncate!(network)

    # Generate samples and process raw data in preparation for modelling.
    samples = MLOPF.cache(MLOPF.generate_samples, "./cache/samples/", "$(case).jld2")(
        network,
        settings.DATA.num_samples,
        settings.DATA.alpha,
    )
    raw_data = MLOPF.process_raw_samples(
        samples,
        network,
        settings.DATA.inequality_constraints;
        normalise = settings.DATA.normalise,
    )

    # Prepare data for modelling: shuffle training set and create mini-batches.
    X = MLOPF.model_input(MLOPF.FullyConnected, raw_data)
    y = MLOPF.model_output(MLOPF.Global, MLOPF.Primals, raw_data)
    train_set, valid_set, test_set = splitobs((X, y), at = Tuple(settings.DATA.splits))
    train_set, valid_set, test_set = MLOPF.build_minibatches(
        (train_set, valid_set, test_set),
        settings.PARAMETERS.batch_size,
        settings.DATA.shuffle,
    )

    #TODO: Can save these all as one file.
    for seed in settings.GENERAL.seeds

        Random.seed!(seed)

        # Build neural network model for specified architecture.
        size_in, size_out = size(X, 1), size(y, 1)
        layers = MLOPF.define_layers(MLOPF.FullyConnected, size_in, size_out, settings.PARAMETERS.num_layers)
        model = MLOPF.build_model(MLOPF.FullyConnected, layers, settings.PARAMETERS.drop_out)

        # Fit model using training and validation sets, recording elapsed time and losses.
        loss_func = MLOPF.mean_squared_error(@. !isnan(y[:, 1]))
        train_time, (train_loss, valid_loss) = MLOPF.train!(
            model,
            train_set,
            valid_set,
            loss_func;
            η = float(settings.PARAMETERS.learning_rate),
            num_epochs = settings.PARAMETERS.num_epochs,
            use_cuda = settings.PARAMETERS.use_cuda,
        )

        # Test model on specified test set, recording elapsed time and loss.
        # TODO: We probably want to record the un-normalised test loss as well.
        test_time, test_loss = MLOPF.test(model, test_set)

        # Cache results.
        MLOPF.cache(
            () -> Dict(
                :target => MLOPF.Primals,
                :encoding => MLOPF.Global,
                :architecture => MLOPF.FullyConnected,
                :train_loss => train_loss,
                :valid_loss => valid_loss,
                :test_loss => test_loss,
                :train_time => train_time,
                :test_time => test_time,
                :trainable_parameters => sum(length(p) for p ∈ params(model)),
            ),
            "./cache/results/",
            "$(case)__seed$(seed).jld2",
        )()
    end

end

