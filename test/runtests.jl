using Test, StatsBase, SearchModels, TextClassification, CategoricalArrays
using Downloads, Random, MLDataUtils, JSON, CodecZlib, KNearestCenters

Random.seed!(1)
function train_test_split(corpus, labels, at=0.7)
    n = length(corpus)
    iX = shuffle!(collect(1:n))
    n_ = floor(Int, n * at)
    itrain, itest = iX[1:n_], iX[n_+1:end]
    corpus[itrain], labels[itrain], corpus[itest], labels[itest]
end

function folds_split(corpus, labels, folds=3)
    n = length(corpus)
    indexes = shuffle!(collect(1:n))
    folds = kfolds(indexes, folds)

    [(corpus[itrain], labels[itrain], corpus[itest], labels[itest]) for (itrain, itest) in folds]
end

@testset "microtc" begin
    !isfile("emo50k.json.gz") && Downloads.download("https://github.com/sadit/TextClassificationTutorial/raw/main/data/emo50k.json.gz", "emo50k.json.gz")
    labels = []
    corpus = []
    targets = ("♡", "💔")
    open("emo50k.json.gz") do f
        gz = GzipDecompressorStream(f)
        for line in eachline(gz)
            tweet = JSON.parse(line)
            label = tweet["klass"]
            if label in targets
                push!(labels, tweet["klass"])
                push!(corpus, tweet["text"])
            end
        end
        close(gz)
    end
    labels = categorical(labels)
    traincorpus, trainlabels, testcorpus, testlabels = train_test_split(corpus, labels, 0.7)
    folds = folds_split(traincorpus, trainlabels)

    for t in traincorpus[1:10]
        @show t
    end

    @show countmap(trainlabels)
    @show countmap(testlabels)

    space = MicroTC_ConfigSpace(
       textconfig=TextConfigSpace(
            qlist=[[4], [3], [5]],
            nlist=[[1], [1, 2], []],
            slist=[]
        )
    )

    params = SearchParams(maxpopulation=8, bsize=2, mutbsize=8, crossbsize=8, tol=0.0, maxiters=30, verbose=true)
    best_list = search_models(space, 32, params) do config
            S = Float64[]
            for (_traincorpus, _trainlabels, _testcorpus, _testlabels) in folds
                tc = MicroTC(config, _traincorpus, _trainlabels; verbose=true)
                ypred = predict_corpus(tc, _testcorpus) |> categorical
                push!(S, recall_score(_testlabels, ypred, weight=:macro))
            end
    
            1.0 - mean(S)
    end

    for (i, b) in enumerate(best_list)
        @info "-- perf best_lists[$i]:", i, b[1], b[2]
    end

    cls = MicroTC(best_list[1][1], traincorpus, trainlabels)
    sc = classification_scores(testlabels, predict_corpus(cls, testcorpus))
    @info "*** Performance on test: " sc
    @test sc.accuracy > 0.6
end
