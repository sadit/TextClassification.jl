# This file is part of TextClassification.jl
using StatsBase
export microtc, microtc_kfolds, scores, recall_score, accuracy_score, f1_score, SearchParams

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
        maxpopulation=8, bsize=2, mutbsize=8, crossbsize=8, maxiters=16, verbose=true,
        params = SearchParams(; maxpopulation, bsize, mutbsize, crossbsize, maxiters, verbose),
        parallel = :threads
    )

    Folds = kfolds(shuffleobs((corpus, labels)), k)

    best_list = search_models(space, initialpopulation, params; parallel) do config
        S = Float64[]
        for ((traincorpus, trainlabels), (testcorpus, testlabels)) in Folds
            tc = MicroTC(config, traincorpus, trainlabels; verbose=true)
            ypred = predict_corpus(tc, testcorpus)
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
        maxpopulation=8, bsize=2, mutbsize=8, crossbsize=8, maxiters=16, verbose=true,
        params = SearchParams(; maxpopulation, bsize, mutbsize, crossbsize, maxiters, verbose),
        sample = 1.0,
        parallel = :none #:threads
    )

    (traincorpus, trainlabels), (testcorpus, testlabels) = if sample < 1.0
        obs = randobs((corpus, labels), ceil(Int, length(labels) * sample))
        splitobs(obs; at)
    else
        splitobs((corpus, labels); at, shuffle=true)
    end


    best_list = search_models(space, initialpopulation, params; parallel) do config
        tc = MicroTC(config, traincorpus, trainlabels; verbose=true)
        ypred = predict_corpus(tc, testcorpus)
        1.0 - score(testlabels, ypred)
    end
   
    MicroTC(best_list[1][1], corpus, labels), best_list
end
