# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt
using SimilaritySearch, KCenters, TextSearch, MLDataUtils
using Distributed, IterTools, Random
import TextSearch: vectorize
import StatsBase: fit, predict
export microtc_search_params, microtc_random_configurations, microtc_combine_configurations, filtered_power_set, fit, predict, vectorize
import Base: hash, isequal

struct μTC_Configuration
    p::Float64
    qlist::Vector{Int}
    nlist::Vector{Int}
    slist::Vector{Tuple{Int,Int}}

    kind::Type
    vkind::Type
    kernel::Function
    dist::Function
    k::Int

    smooth::Float64
    ncenters::Int
    maxiters::Int
    weights
    initial_clusters

    split_entropy::Float64
end 

hash(a::μTC_Configuration) = hash(repr(a))
isequal(a::μTC_Configuration, b::μTC_Configuration) = isequal(repr(a), repr(b))

mutable struct μTC
    nc::NearestCentroid
    model::TextSearch.Model
    config::μTC_Configuration
    kernel::Function
end

function filtered_power_set(set, lowersize=0, uppersize=5)
    lst = collect(subsets(set))
    filter(x -> lowersize <= length(x) <= uppersize, lst)
end

function fit(::Type{μTC}, config::μTC_Configuration, train_corpus, train_y; verbose=true)
    textconfig = TextConfig(qlist=config.qlist, nlist=config.nlist, slist=config.slist)

    model = fit(config.kind, textconfig, train_corpus, train_y, smooth=config.smooth, weights=config.weights)
    if config.p < 1.0
        model = prune_select_top(model, config.p)
    end

    train_X = [vectorize(model, config.vkind, text) for text in train_corpus]
    
    if config.ncenters == 0
        C = kcenters(config.dist, train_X, train_y, TextSearch.centroid)
        cls = fit(NearestCentroid, C)
    else
        C = kcenters(config.dist, train_X, config.ncenters, TextSearch.centroid, initial=config.initial_clusters, recall=1.0, verbose=verbose, maxiters=config.maxiters)
        cls = fit(NearestCentroid, cosine_distance, C, train_X, train_y, TextSearch.centroid, split_entropy=config.split_entropy, verbose=verbose)
    end

    μTC(cls, model, config, config.kernel(config.dist))
end

function predict(tc::μTC, X)
    ypred = predict(tc.nc, tc.kernel, X, tc.config.k)
end

function vectorize(tc::μTC, text)
    vectorize(tc.model, tc.config.vkind, text)
end

function evaluate_model(config, train_corpus, train_y, test_corpus, test_y; verbose=true)
    mtc = fit(μTC, config, train_corpus, train_y)
    test_X = [vectorize(mtc, text) for text in test_corpus]
    ypred = predict(mtc, test_X)
    (scores=scores(test_y, ypred), voc=length(mtc.model.tokens))
end

const QLIST = filtered_power_set([2, 3, 4, 5, 6], 1, 3)
const NLIST = filtered_power_set([1, 2, 3], 0, 2)
const SLIST = filtered_power_set([(2, 1), (2, 2)], 0, 1)

function microtc_random_configurations(H, ssize;
        qlist=QLIST,
        nlist=NLIST,
        slist=SLIST,
        kernel=[relu_kernel], # [gaussian_kernel, laplacian_kernel, sigmoid_kernel, relu_kernel]
        dist=[cosine_distance],
        k=[1, 3],
        smooth=[0, 1, 3],
        p=[1.0],
        maxiters=[1, 3, 10],
        kind=[EntModel],
        vkind=[EntModel, EntTpModel],
        ncenters=[0, 10],
        weights=[:balance],
        initial_clusters=[:fft, :dnet, :rand],
        split_entropy=[0.3, 0.7],
        verbose=true
    )

    _rand_list(lst) = length(lst) == 0 ? [] : rand(lst)

    H = H === nothing ? Dict{μTC_Configuration,Float64}() : H
    iter = 0
    for i in 1:ssize
        iter += 1
        ncenters_ = rand(ncenters)
        if ncenters_ == 0
            maxiters_ = 0
            split_entropy_ = 0.0
            initial_clusters_ = :rand # nothing in fact
            k_ = 1
        else
            maxiters_ = rand(maxiters)
            split_entropy_ = rand(split_entropy)
            initial_clusters_ = rand(initial_clusters)
            k_ = rand(k)
        end

        config = μTC_Configuration(
            rand(p), _rand_list(qlist), _rand_list(nlist), _rand_list(slist),
            rand(kind), rand(vkind), rand(kernel), rand(dist), k_,
            rand(smooth), ncenters_, maxiters_,
            rand(weights), initial_clusters_, split_entropy_
        )
        haskey(H, config) && continue
        H[config] = -1
    end

    H
end

function microtc_combine_configurations(config_list, ssize, H)
    function _sel()
        rand(config_list)
    end  

    for i in 1:ssize
        b = _sel()
        qlist_, nlist_, slist_ = _sel().qlist, _sel().nlist, _sel().slist
        length(qlist_) + length(nlist_) + length(slist_) == 0 && continue
        
        config = μTC_Configuration(
            _sel().p,
            qlist_, nlist_, slist_,
            _sel().kind, _sel().vkind, _sel().kernel, _sel().dist, b.k,
            _sel().smooth, b.ncenters, b.maxiters,
            _sel().weights, b.initial_clusters, b.split_entropy
        )
        haskey(H, config) && continue
        H[config] = -1
    end

    H
end

function microtc_search_params(corpus, y, configurations;
        bsize=4,
        mutation_bsize=1,
        ssize=8,
        folds=0.7,
        maxiters=8,
        score=:macro_recall,
        tol=0.01,
        verbose=true,
        config_kwargs...
    )
    if configurations isa Integer
       configurations = microtc_random_configurations(nothing, configurations; config_kwargs...)
    end

    n = length(y)
    if folds isa Integer
        indexes = shuffle!(collect(1:n))
        folds = kfolds(indexes, folds)
    elseif folds isa AbstractFloat
        !(0.0 < folds < 1.0) && error("the folds parameter should follow 0.0 < folds < 1.0")
        indexes = shuffle!(collect(1:n))
        m = ceil(Int, folds * n)
        folds = [(indexes[1:m], indexes[m+1:end])]
    end
    
    prev = 0.0
    iter = 0
    while iter <= maxiters
        iter += 1
        C = μTC_Configuration[]
        S = []

        for (config, score_) in configurations
            score_ >= 0.0 && continue
            score_ = @spawn begin
                s = 0.0
                local perf = nothing
                for (itrain, itest) in folds
                    perf = evaluate_model(config, corpus[itrain], y[itrain], corpus[itest], y[itest])
                    s += perf.scores[score] * 1/length(folds)
                end
                (score=s, perf...)
            end
            push!(C, config)
            push!(S, score_)
        end
        verbose && println(stderr, "iteration $iter finished")

        for (c, p) in zip(C, S)
            p = fetch(p)
            configurations[c] = p.score
        end

        if iter <= maxiters
            L = sort!(collect(configurations), by=x->x[2], rev=true)
            curr = L[1][2]
            if abs(curr - prev) <= tol                
                verbose && println(stderr, "stopping on iteration $iter due to a possible convergence ($curr ≃ $prev, tol: $tol)")
                break
            end

            prev = curr
            if verbose
                println(stderr, "generating $ssize configurations using top $bsize configurations, starting with $(length(configurations)))")
                println(stderr, [l[end] for l in L])
                println(stderr, L[1])
            end

            L = [L[i][1] for i in 1:min(bsize, length(L))]
            if mutation_bsize > 0
                for p in keys(microtc_random_configurations(nothing, mutation_bsize; config_kwargs...))
                    push!(L, p)
                end
            end

            microtc_combine_configurations(L, ssize, configurations)
            verbose && println(stderr, "finished with $(length(configurations))")
        end
    end

    sort!(collect(configurations), by=x->x[2], rev=true)
end
