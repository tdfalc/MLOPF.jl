using SHA
using JLD2
using FileIO

cache(f::Function, savedir::String, filename::String) = file_cache(f, savedir, filename)
cache(f::Function, savedir::String; use_mem_cache::Bool=true) = hash_cache(f, savedir, use_mem_cache=use_mem_cache)

"""
    hash_cache(f::function, savedir::String; use_mem_cache::bool=true)

This function applies disk (and optional memory) memoization - the function output is saved 
    using JLD2 by combining the function name with the hash of args and kwargs. 
"""
function hash_cache(f::Function, savedir::String; use_mem_cache::Bool=true)
    mem_cache = Dict()
    (args...; kwargs...) -> begin

        input_hash = bytes2hex(sha256(string(args...; kwargs...)))
        cache_path = joinpath(savedir, "$(f)_$(input_hash).jld2")

        if use_mem_cache & (cache_path in keys(mem_cache))
            return mem_cache[cache_path]
        end

        if isfile(cache_path)
            output = load(cache_path, "output")
            mem_cache[cache_path] = output
            return output
        end

        output = f(args...; kwargs...)
        mem_cache[cache_path] = output

        if !isdir(savedir)
            mkdir(savedir)
        end

        FileIO.save(cache_path, Dict("output" => output))
        return output
    end
end

"""
    file_cache(f::function, savedir::String, filename::String)

This function applies disk memoization by saving the function output to a specified path.  
"""
function file_cache(f::Function, savedir::String, filename::String)
    (args...; kwargs...) -> begin
        cache_path = joinpath(savedir, filename)
        if isfile(cache_path)
            output = load(cache_path, "output")
            return output
        else
            if !ispath(savedir)
                mkpath(savedir)
            end
            output = f(args...; kwargs...)
            FileIO.save(cache_path, Dict("output" => output))
            return output
        end
    end
end