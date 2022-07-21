# This file is part of TextClassification.jl
using KNearestCenters, StatsBase
export microtc, microtc_kfolds

function microtc_kfolds(
        corpus,
        labels;
        slist = [],
        nlist = [[1], [1, 2], []],
        qlist = [[4], [3], [5]],
        space = MicroTC_ConfigSpace(
            textconfig=TextConfigSpace(
                qlist=qlist,
                nlist=nlist,
                slist=slist,
            ),

        ),
        initialpopulation = 32,
        k = 3,
        score = (gold, pred) -> recall_score(gold, pred, weight=:macro),
        maxpopulation=8, bsize=2, mutbsize=8, crossbsize=8, tol=0.0, maxiters=16, verbose=true,
        params = SearchParams(; maxpopulation, bsize, mutbsize, crossbsize, tol, maxiters, verbose),
        parallel = :threads
    )

    Folds = kfolds(shuffleobs((corpus, labels)), k)

    best_list = search_models(space, initialpopulation, params; parallel) do config
        S = Float64[]
        for ((traincorpus, trainlabels), (testcorpus, testlabels)) in Folds
            tc = MicroTC(config, traincorpus, trainlabels; verbose=true)
            ypred = predict_corpus(tc, testcorpus) |> categorical
            push!(S, score(testlabels, ypred))
        end
   
        1.0 - mean(S)
    end
   
    MicroTC(best_list[1][1], corpus, labels), best_list
end

function microtc(
        corpus,    
        labels;
        slist = [],
        nlist = [[1], [1, 2], []],
        qlist = [[4], [3], [5]],
        space = MicroTC_ConfigSpace(
            textconfig=TextConfigSpace(
                qlist=qlist,
                nlist=nlist,
                slist=slist,
            ),

        ),
        initialpopulation = 32,
        score = (gold, pred) -> recall_score(gold, pred, weight=:macro),
        at = 0.7,
        maxpopulation=8, bsize=2, mutbsize=8, crossbsize=8, tol=0.0, maxiters=16, verbose=true,
        params = SearchParams(; maxpopulation, bsize, mutbsize, crossbsize, tol, maxiters, verbose),
        sample = 1.0,
        parallel = :none #:threads
    )

    obs = if sample < 1.0
        randobs((corpus, labels), ceil(Int, length(labels) * sample))
    else
        shuffleobs((corpus, labels))
    end

    (traincorpus, trainlabels), (testcorpus, testlabels) = splitobs(obs; at)

    best_list = search_models(space, initialpopulation, params; parallel) do config
        tc = MicroTC(config, traincorpus, trainlabels; verbose=true)
        ypred = predict_corpus(tc, testcorpus) |> categorical
        1.0 - score(testlabels, ypred)
    end
   
    MicroTC(best_list[1][1], corpus, labels), best_list
end
