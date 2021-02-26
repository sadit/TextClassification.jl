using Test, StatsBase, SearchModels, KCenters, KNearestCenters, TextSearch, TextClassification, CategoricalArrays
using CSV, Random, MLDataUtils, JSON3

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
    !isfile("emotions.csv") && download("http://ingeotec.mx/~sadit/emotions.csv", "emotions.csv")
    
    X = CSV.read("emotions.csv", NamedTuple)
    targets = ["♡", "💔"]
    I = [(l in targets) for l in X.klass]
    labels = categorical(X.klass[I])
    corpus = X.text[I]

    traincorpus, trainlabels, testcorpus, testlabels = train_test_split(corpus, labels, 0.7)
    folds = folds_split(traincorpus, trainlabels)

    function error_function(config::MicroTC_Config)
        S = Float64[]
        for (_traincorpus, _trainlabels, _testcorpus, _testlabels) in folds
            tc = MicroTC(config, _traincorpus, _trainlabels; verbose=true)
            valX = vectorize.(tc, _testcorpus)
            ypred = predict.(tc, valX)
            push!(S, recall_score(_testlabels.refs, ypred, weight=:macro))
        end

        1.0 - mean(S)
    end

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

    best_list = search_models(space, error_function, 16;
        # search hyper-parameters
        bsize=8, mutbsize=8, crossbsize=8,
        tol=0.0, maxiters=8, verbose=true)

    for (i, b) in enumerate(best_list)
        @info "-- perf best_lists[$i]:", i, b[1], b[2]
    end

    cls = MicroTC(best_list[1][1], traincorpus, trainlabels)
    sc = classification_scores(testlabels.refs, [predict(cls, t) for t in testcorpus])
    @info "*** Performance on test: " sc
    @test sc.accuracy > 0.6

    cls_ = JSON3.read(JSON3.write(cls), typeof(cls))
    sc = classification_scores(testlabels.refs, [predict(cls_, t) for t in testcorpus])
    @info "*** Performance on test: " sc
    @test sc.accuracy > 0.6

end


# flush(stdout); flush(stderr)
# sort!(P, :score, rev=true)
# plot(P.score, label="", xlabel="rank", ylabel="score", markershape=:auto, title="RS performance on validation set") |> display
