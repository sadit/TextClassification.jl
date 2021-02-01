# This file is a part of TextClassification.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export AbstractConfigSpace, AbstractConfig, search_models, random_configuration, combine_configurations, evaluate_model
using Distributed, Random, StatsBase
abstract type AbstractConfigSpace end
abstract type AbstractConfig end

Base.hash(a::AbstractConfig) = hash(repr(a))
Base.isequal(a::AbstractConfig, b::AbstractConfig) = isequal(repr(a), repr(b))

#function random_configuration(space::AbstractConfigSpace) end
#function combine_configurations(space::AbstractConfigSpace, config_list::AbstractVector) end
#function evaluate_model(config::AbstractConfig) end

function random_configuration end
function combine_configurations end
function evaluate_model end

function search_models(
        configspace::AbstractConfigSpace, corpus, y::CategoricalArray, m=8;
        configurations=Dict{AbstractConfig,Float64}(),
        bsize=16,
        mutbsize=4,
        crossbsize=4,
        folds=0.7,
        searchmaxiters=8,
        score=:macrorecall,
        tol=0.01,
        verbose=true,
        distributed=false
    )
    
    for i in 1:m
        configurations[random_configuration(configspace)] = -1.0
    end

    n = length(y)
    if folds isa Integer
        indexes = shuffle!(collect(1:n))
        folds = kfolds(indexes, folds)
    elseif folds isa AbstractFloat
        !(0.0 < folds < 1.0) && error("ModelSelection> the folds parameter should follow 0.0 < folds < 1.0")
        indexes = shuffle!(collect(1:n))
        m = ceil(Int, folds * n)
        folds = [(indexes[1:m], indexes[m+1:end])]
    end
    
    if score isa Symbol
        scorefun = (perf) -> perf.scores[score]
    else
        scorefun = score::Function
    end
    
    prev = 0.0
    iter = 0
    while iter <= searchmaxiters
        iter += 1
        C = AbstractConfig[]
        S = []

        verbose && println(stderr, "ModelSelection> ==== search params iter=$iter, tol=$tol, m=$m, bsize=$bsize, mutbsize=$mutbsize, crossbsize=$crossbsize, prev=$prev, $(length(configurations))")

        for (config, score_) in configurations
            score_ >= 0.0 && continue
            push!(S, [])

            if distributed
                for (itrain, itest) in folds
                    perf = @spawn evaluate_model(config, corpus[itrain], y[itrain], corpus[itest], y[itest])
                    push!(S[end], perf)
                end
            else
                for (itrain, itest) in folds
                    perf = evaluate_model(config, corpus[itrain], y[itrain], corpus[itest], y[itest])
                    push!(S[end], perf)
                end
            end
        
            push!(C, config)
        end

        for (c, perf_list) in zip(C, S)
            configurations[c] = mean([scorefun(fetch(p)) for p in perf_list])
        end
        
        verbose && println(stderr, "ModelSelection> *** iteration $iter finished; starting combinations.")

        if iter <= searchmaxiters
            L = sort!(collect(configurations), by=x->x[2], rev=true)
            curr = L[1][2]
            if abs(curr - prev) <= tol                
                verbose && println(stderr, "ModelSelection> *** stopping on iteration $iter due to a possible convergence ($curr ≃ $prev, tol: $tol)")
                break
            end

            prev = curr
            if verbose
                println(stderr, "ModelSelection> *** adding more items to the population: bsize=$bsize; #configurations=$(length(configurations)))")
                println(stderr, "ModelSelection> *** scores: ", [l[end] for l in L])
				config__, score__ = L[1]
                println(stderr, "ModelSelection> *** best config with score $score__: ", [(k => getfield(config__, k)) for k in fieldnames(typeof(config__))])
            end

            L =  AbstractConfig[L[i][1] for i in 1:min(bsize, length(L))]
            for i in 1:mutbsize
                conf = combine_configurations(configspace, [rand(L), random_configuration(configspace)])
                if !haskey(configurations, conf)
                    configurations[conf] = -1.0
                end
            end

            for i in 1:crossbsize
                conf = combine_configurations(configspace, L)
                if !haskey(configurations, conf)
                    configurations[conf] = -1.0
                end
            end

            verbose && println(stderr, "ModelSelection> *** finished with $(length(configurations)) configurations")
        end
    end

    sort!(collect(configurations), by=x->x[2], rev=true)
end
