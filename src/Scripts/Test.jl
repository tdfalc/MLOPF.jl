"""Run basic pipeline: generate and process samples, then build, fit and evaluate spcecified architectures.

Notes:
    - Starting with julia -p n provides n worker processes on local machine.
"""

@everywhere begin
    using Pkg
    Pkg.activate(".")
end

@everywhere begin
    using MLOPF
end

# settings = MLOPF.get_settings()
# case = settings.PGLIB_OPF.cases[1]
# # Parse synthetic grid file (*.m) to Powermodels.jl network.
# network = PowerModels.parse_file(ENV["HOME"] * settings.PGLIB_OPF.path * "$(case).m")
# MLOPF.truncate!(network)

# # Generate samples and process raw data in preparation for modelling.
# samples = MLOPF.cache(MLOPF.generate_samples, "./cache/samples/", "$(case).jld2")(
#     network,
#     settings.DATA.num_samples,
#     settings.DATA.alpha,
# )
# raw_data = MLOPF.process_raw_samples(
#     samples,
#     network,
#     settings.DATA.inequality_constraints;
#     normalise = settings.DATA.normalise,
# )

