# This file is a part of TextClassification.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using SimilaritySearch, KCenters, KNearestCenters, TextSearch, MLDataUtils, LinearAlgebra
using Distributed, Random, StatsBase
import KNearestCenters: transform, search_params, combine_configurations, random_configuration
import TextSearch: vectorize
import StatsBase: fit, predict
import Base: hash, isequal
export search_params, random_configuration, combine_configurations,
    filtered_power_set, fit, predict, vectorize, transform,
    MicroTC, AngleDistance, CosineDistance, NormalizedAngleDistance, NormalizedCosineDistance
import Base: hash, isequal

struct MicroTC{C_<:MicroTC_Config, AKNC_<:AKNC, TextModel_<:TextModel}
    config::C_
    aknc::AKNC_
    textmodel::TextModel_
end

Base.copy(c::MicroTC;
        config=c.config,
        aknc::AKNC=c.nc,
        textmodel=c.textmodel
) = MicroTC(config, aknc, textmodel)

function broadcastable(tc::MicroTC)
    (tc,)
end

#=
function MicroTC(textconfig::TextConfig, textmodel::TextModel, config::MicroTC_Config, train_corpus::AbstractVector{BOW}, train_y::CategoricalArray; verbose=true) 
    textmodel = create_textmodel(config, train_corpus, train_y)
    MicroTC(config, textmodel, train_corpus, train_y; verbose=verbose)
end
=#
function create_textmodel(config::MicroTC_Config, train_X::AbstractVector{BOW}, train_y::CategoricalArray)
    model = if config.textmodel === EntModel
        EntModel(config.weighting, train_X, train_y, smooth=config.smooth, minocc=config.minocc, weights=config.classweights)
    else
        VectorModel(config.weighting, sum(train_X), minocc=config.minocc)
    end

    if config.p < 1.0
        model = prune_select_top(model, config.p)
    end

    model
end

function MicroTC(config::MicroTC_Config, train_corpus::AbstractVector{S}, train_y::CategoricalArray; verbose=true) where {S<:AbstractString}
    verbose && println("MicroTC> creating bag of words for corpus")
    train_corpus_bow = [compute_bow(config.textconfig, text) for text in train_corpus]
    verbose && println("MicroTC> creating textmodel $(config.textmodel)")
    textmodel = create_textmodel(config, train_corpus_bow, train_y)
    MicroTC(config, textmodel, [vectorize(textmodel, bow) for bow in train_corpus_bow], train_y)
end

function MicroTC(config::MicroTC_Config, textmodel::TextModel, train_X::AbstractVector{S}, train_y::CategoricalArray; verbose=true) where {S<:SVEC}
    verbose && println("MicroTC> creating AKNC classifier")
    aknc = AKNC(config.akncconfig, train_X, train_y; verbose=verbose)
    verbose && println("MicroTC> done")
    MicroTC(config, aknc, textmodel)
end

vectorize(tc::MicroTC, text)::SVEC = vectorize(tc.textmodel, compute_bow(tc.config.textconfig, text))
vectorize(tc::MicroTC, bow::BOW)::SVEC = vectorize(tc.textmodel, bow)

predict(tc::MicroTC, text::S) where {S<:Union{AbstractString,BOW}} = predict(tc.aknc, vectorize(tc, text))
predict(tc::MicroTC, vec::SVEC) = predict(tc.aknc, vec)

function transform(tc::MicroTC, vec::SVEC)
    X = transform(tc.nc.centers, tc.nc.dmax, tc.kernel, vec)
    M = labelmap(tc.nc.class_map)
    L = zeros(Float64, tc.nc.nclasses)

    for i in 1:tc.nc.nclasses
        lst = get(M, i, nothing)
        if lst !== nothing
           L[i] = maximum(X[lst])
        end
    end

    L
end

transform(tc::MicroTC, text::AbstractString) = transform(tc, vectorize(tc, text))

function evaluate_model(
        config::MicroTC_Config,
        train_corpus::AbstractVector{String},
        train_y::CategoricalArray,
        test_corpus::AbstractVector{String},
        test_y::CategoricalArray;
        verbose=true
    )
    tc = MicroTC(config, train_corpus, train_y; verbose=verbose)
    test_X = [vectorize(tc, text) for text in test_corpus]
    ypred = [predict(tc, x) for x in test_X]
    s = classification_scores(test_y.refs, ypred)
    if verbose
        println(stderr, "MicroTC> gold:", typeof(test_y), ", ypred:", typeof(ypred), "-- scores:", s)
    end
    (scores=s, voc=length(tc.textmodel.tokens))
end


function search_params(configspace::MicroTC_ConfigSpace, corpus, y::CategoricalArray, m=8;
        configurations=Dict{MicroTC_Config,Float64}(),
        bsize=4,
        mutationbsize=1,
        ssize=8,
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
        !(0.0 < folds < 1.0) && error("MicroTC> the folds parameter should follow 0.0 < folds < 1.0")
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
        C = MicroTC_Config[]
        S = []

        verbose && println(stderr, "MicroTC> ==== search params iter=$iter, tol=$tol, m=$m, bsize=$bsize, mutatioonbsize=$mutationbsize, ssize=$ssize, prev=$prev, $(length(configurations))")

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
        
        verbose && println(stderr, "MicroTC> *** iteration $iter finished; starting combinations.")

        if iter <= searchmaxiters
            L = sort!(collect(configurations), by=x->x[2], rev=true)
            curr = L[1][2]
            if abs(curr - prev) <= tol                
                verbose && println(stderr, "MicroTC> *** stopping on iteration $iter due to a possible convergence ($curr ≃ $prev, tol: $tol)")
                break
            end

            prev = curr
            if verbose
                println(stderr, "MicroTC> *** generating $ssize configurations using top $bsize configurations, starting with $(length(configurations)))")
                println(stderr, "MicroTC> *** scores: ", [l[end] for l in L])
				config__, score__ = L[1]
                println(stderr, "MicroTC> *** best config with score $score__: ", [(k => getfield(config__, k)) for k in fieldnames(typeof(config__))])
            end

            L =  MicroTC_Config[L[i][1] for i in 1:min(bsize, length(L))]
            for i in 1:mutationbsize
                push!(L, random_configuration(configspace))
            end

            for i in 1:ssize
                conf = combine_configurations(L)
                if !haskey(configurations, conf)
                    configurations[conf] = -1.0
                end
            end
            verbose && println(stderr, "MicroTC> *** finished with $(length(configurations)) configurations")
        end
    end

    sort!(collect(configurations), by=x->x[2], rev=true)
end
