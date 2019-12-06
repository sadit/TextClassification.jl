# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt
using SimilaritySearch, KCenters, TextSearch, MLDataUtils
using Distributed, IterTools, Random, StatsBase
import TextSearch: vectorize
import StatsBase: fit, predict
import Base: hash, isequal
export microtc_search_params, microtc_random_configurations, microtc_combine_configurations, filtered_power_set, fit, predict, vectorize, μTC_Configuration
import Base: hash, isequal

struct μTC_Configuration{Kind,VKind}
    p::Float64

    del_diac::Bool
    del_dup::Bool
    del_punc::Bool
    group_num::Bool
    group_url::Bool
    group_usr::Bool
    group_emo::Bool
 
    qlist::Vector{Int}
    nlist::Vector{Int}
    slist::Vector{Tuple{Int,Int}}

    kind::Type{Kind}
    vkind::Type{VKind}
    kernel::Function
    dist::Function
    
    k::Int
    smooth::Float64
    ncenters::Int
    maxiters::Int
    
    recall::Float64
    weights
    initial_clusters
    split_entropy::Float64
end

function μTC_Configuration(;
        p::Real=1.0,

        del_diac::Bool=true,
        del_dup::Bool=false,
        del_punc::Bool=false,
        group_num::Bool=true,
        group_url::Bool=true,
        group_usr::Bool=false,
        group_emo::Bool=false,
 
        qlist::AbstractVector=[5],
        nlist::AbstractVector=[],
        slist::AbstractVector=[],
        
        kind::Type=EntModel,
        vkind::Type=EntModel,
        kernel::Function=relu_kernel, # [gaussian_kernel, laplacian_kernel, sigmoid_kernel, relu_kernel]
        dist::Function=cosine_distance,
        
        k::Int=1,
        smooth::AbstractFloat=3.0,
        ncenters::Integer=0,
        maxiters::Integer=1,
        
        recall::AbstractFloat=1.0,
        weights=:balance,
        initial_clusters=:rand,
        split_entropy::AbstractFloat=0.9)
    
    μTC_Configuration(
        p,
        del_diac, del_dup, del_punc,
        group_num, group_url, group_usr, group_emo,
        qlist, nlist, convert(Vector{Tuple{Int,Int}}, slist),
        kind, vkind, kernel, dist,
        k, smooth, ncenters, maxiters,
        recall, weights, initial_clusters, split_entropy)
end

hash(a::μTC_Configuration) = hash(repr(a))
isequal(a::μTC_Configuration, b::μTC_Configuration) = isequal(repr(a), repr(b))

mutable struct μTC{Kind,VKind,T}
    nc::NearestCentroid{T}
    model::Kind
    config::μTC_Configuration{Kind,VKind}
    kernel::Function
end

function filtered_power_set(set, lowersize=0, uppersize=5)
    lst = collect(subsets(set))
    filter(x -> lowersize <= length(x) <= uppersize, lst)
end

function create_textconfig(config::μTC_Configuration)
    TextConfig(
        del_diac = config.del_diac,
        del_dup = config.del_dup,
        del_punc = config.del_punc,
        group_num = config.group_num,
        group_url = config.group_url,
        group_usr = config.group_usr,
        group_emo = config.group_emo,
        qlist = config.qlist,
        nlist = config.nlist,
        slist = config.slist
    )
end

function create_textmodel(config::μTC_Configuration{EntModel,VKind}, train_corpus, train_y) where VKind
    model = fit(EntModel, create_textconfig(config), train_corpus, train_y, smooth=config.smooth, weights=config.weights)
    if config.p < 1.0
        model = prune_select_top(model, config.p)
    end

    model
end

function create_textmodel(config::μTC_Configuration{VectorModel,VKind}, train_corpus, train_y) where VKind
    model = fit(VectorModel, create_textconfig(config), train_corpus)
    if config.p < 1.0
        model = prune_select_top(model, config.p)
    end

    model
end

function fit(::Type{μTC}, config::μTC_Configuration{Kind,VKind}, train_corpus, train_y; verbose=true) where {Kind,VKind}
    model = create_textmodel(config, train_corpus, train_y)
    train_X = [vectorize(model, config.vkind, text) for text in train_corpus]
    
    if config.ncenters == 0
        C = kcenters(config.dist, train_X, train_y, TextSearch.centroid)
        cls = fit(NearestCentroid, C)
    else
        C = kcenters(config.dist, train_X, config.ncenters, TextSearch.centroid, initial=config.initial_clusters, recall=config.recall, verbose=verbose, maxiters=config.maxiters)
        cls = fit(NearestCentroid, cosine_distance, C, train_X, train_y, TextSearch.centroid, split_entropy=config.split_entropy, verbose=verbose)
    end

    μTC(cls, model, config, config.kernel(config.dist))
end

fit(config::μTC_Configuration, train_corpus, train_y; verbose=true) = fit(μTC, config, train_corpus, train_y; verbose=verbose)

function predict(tc::μTC, X)
    ypred = predict(tc.nc, tc.kernel, X, tc.config.k)
end

function vectorize(tc::μTC{Kind,VKind}, text) where {Kind,VKind}
    vectorize(tc.model, VKind, text)
end

function transform(tc::μTC{Kind,VKind}, text) where {Kind,VKind}

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
        del_diac::AbstractVector=[true],
        del_dup::AbstractVector=[false],
        del_punc::AbstractVector=[false],
        group_num::AbstractVector=[true],
        group_url::AbstractVector=[true],
        group_usr::AbstractVector=[false],
        group_emo::AbstractVector=[false],
        qlist::AbstractVector=QLIST,
        nlist::AbstractVector=NLIST,
        slist::AbstractVector=SLIST,
        kernel::AbstractVector=[relu_kernel], # [gaussian_kernel, laplacian_kernel, sigmoid_kernel, relu_kernel]
        dist::AbstractVector=[cosine_distance],
        k::AbstractVector=[1],
        smooth::AbstractVector=[0, 1, 3],
        p::AbstractVector=[1.0],
        maxiters::AbstractVector=[1, 3, 10],
        recall::AbstractVector=[1.0],
        kind::AbstractVector=[EntModel, VectorModel],
        vkind=Dict(EntModel => [EntModel, EntTpModel, EntTpModel], VectorModel => [TfModel, IdfModel, TfidfModel, FreqModel]),
        ncenters::AbstractVector=[0, 10],
        weights::AbstractVector=[:balance, nothing],
        initial_clusters::AbstractVector=[:fft, :dnet, :rand],
        split_entropy::AbstractVector=[0.3, 0.6, 0.9],
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
            weights_ = rand(weights)
        else
            maxiters_ = rand(maxiters)
            split_entropy_ = rand(split_entropy)
            initial_clusters_ = rand(initial_clusters)
            k_ = rand(k)
            weights_ = nothing
        end

        kind_ = rand(kind)
        vkind_ = vkind isa Dict ? rand(vkind[kind_]) : rand(vkind)
        smooth_ = kind_ == EntModel ? Float64(rand(smooth)) : 0.0

        config = μTC_Configuration(
            p = rand(p),
            del_diac = rand(del_diac),
            del_dup = rand(del_dup),
            del_punc = rand(del_punc),
            group_num = rand(group_num),
            group_url = rand(group_url),
            group_usr = rand(group_usr),
            group_emo = rand(group_emo),
            qlist = _rand_list(qlist),
            nlist = _rand_list(nlist),
            slist = _rand_list(slist),
            kind = kind_,
            vkind = vkind_,
            kernel = rand(kernel),
            dist = rand(dist),
            k = k_,
            smooth = smooth_,
            ncenters = ncenters_,
            maxiters = maxiters_,
            recall = rand(recall),
            weights = weights_,
            initial_clusters = initial_clusters_,
            split_entropy = split_entropy_
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
        a = _sel()
        kind_ = a.kind
        #vkind_ = a.vkind
        a2 = _sel()
        vkind_ = kind_ == a2.kind ? a2.vkind : a.vkind

        b = _sel()
        qlist_, nlist_, slist_ = _sel().qlist, _sel().nlist, _sel().slist
        length(qlist_) + length(nlist_) + length(slist_) == 0 && continue
        
        config = μTC_Configuration(
            p = _sel().p,
            del_diac = _sel().del_diac,
            del_dup = _sel().del_dup,
            del_punc = _sel().del_punc,
            group_num = _sel().group_num,
            group_url = _sel().group_url,
            group_usr = _sel().group_usr,
            group_emo = _sel().group_emo,
            qlist = qlist_,
            nlist = nlist_,
            slist = slist_,
            kind = kind_,
            vkind = vkind_,
            kernel = _sel().kernel,
            dist = _sel().dist,
            k = b.k,
            smooth = _sel().smooth,
            ncenters = b.ncenters,
            maxiters = b.maxiters,
            recall = _sel().recall,
            weights = b.weights,
            initial_clusters = b.initial_clusters,
            split_entropy = b.split_entropy
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
        search_maxiters=8,
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
    
    if score isa Symbol
        scorefun = (perf) -> perf.scores[score]
    else
        scorefun = score::Function
    end
    
    prev = 0.0
    iter = 0
    while iter <= search_maxiters
        iter += 1
        C = μTC_Configuration[]
        S = []

        for (config, score_) in configurations
            score_ >= 0.0 && continue
            push!(S, [])
            
            for (itrain, itest) in folds
                #perf = @spawn evaluate_model(config, corpus[itrain], y[itrain], corpus[itest], y[itest])
                perf = evaluate_model(config, corpus[itrain], y[itrain], corpus[itest], y[itest])
                push!(S[end], perf)
            end
        
            push!(C, config)
        end
        
        verbose && println(stderr, "iteration $iter finished")

        for (c, perf_list) in zip(C, S)
            configurations[c] = mean([scorefun(fetch(p)) for p in perf_list])
        end

        if iter <= search_maxiters
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

            L =  μTC_Configuration[L[i][1] for i in 1:min(bsize, length(L))]
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
