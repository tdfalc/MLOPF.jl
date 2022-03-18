using Distributed
using PowerModels
using MLDataUtils
using Random
using Flux

PowerModels.silence()

@everywhere begin
    using Pkg
    Pkg.activate(".")
end

@everywhere begin
    using MLOPF
end

settings = MLOPF.get_settings()
case = settings.PGLIB_OPF.cases[1]

network = PowerModels.parse_file(ENV["HOME"] * settings.PGLIB_OPF.path * "$(case).m")
MLOPF.truncate!(network)

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
