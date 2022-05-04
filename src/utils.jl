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
        params = SearchParams(maxpopulation=8, bsize=2, mutbsize=8, crossbsize=8, tol=0.0, maxiters=16, verbose=true)
    )

    Folds = kfolds(shuffleobs((corpus, labels)), k)

    best_list = search_models(space, initialpopulation, params) do config
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
        params = SearchParams(maxpopulation=8, bsize=2, mutbsize=8, crossbsize=8, tol=0.0, maxiters=16, verbose=true)
    )

    (traincorpus, trainlabels), (testcorpus, testlabels) = stratifiedobs(shuffleobs((corpus, labels)), at)

    best_list = search_models(space, initialpopulation, params) do config
        tc = MicroTC(config, traincorpus, trainlabels; verbose=true)
        ypred = predict_corpus(tc, testcorpus) |> categorical
        1.0 - score(testlabels, ypred)
    end
   
    MicroTC(best_list[1][1], corpus, labels), best_list
end
