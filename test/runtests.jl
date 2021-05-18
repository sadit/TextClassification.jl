using Test, StatsBase, SearchModels, KCenters, KNearestCenters, TextSearch, TextClassification, CategoricalArrays
using Downloads, Random, MLDataUtils, JSON3, GZip

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
    !isfile("emo50k.json.gz") && download("https://github.com/sadit/TextClassificationTutorial/raw/main/data/emo50k.json.gz", "emo50k.json.gz")
    labels = []
    corpus = []
    targets = ("♡", "💔")
    GZip.open("emo50k.json.gz") do f
        for line in eachline(f)
            tweet = JSON3.read(line)
            label = tweet["klass"]
            if label in targets
                push!(labels, tweet["klass"])
                push!(corpus, tweet["text"])
            end
        end
    end
    labels = categorical(labels)
    traincorpus, trainlabels, testcorpus, testlabels = train_test_split(corpus, labels, 0.7)
    folds = folds_split(traincorpus, trainlabels)

    function error_function(config::MicroTC_Config)
        S = Float64[]
        for (_traincorpus, _trainlabels, _testcorpus, _testlabels) in folds
            tc = MicroTC(config, _traincorpus, _trainlabels; verbose=true)
            #valX = vectorize_corpus(tc, _testcorpus)
            #ypred = predict.(tc, valX)
            ypred = predict_corpus(tc, _testcorpus)
            push!(S, recall_score(_testlabels, ypred, weight=:macro))
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
        ),
        # cls=KncConfigSpace(
        #     centerselection=[TextCentroidSelection()],
        #     kernel=[k_(AngleDistance()) for k_ in [DirectKernel]]
        # )
        #cls = KnnClassifierConfigSpace()
        #cls=LiblinearConfigSpace()
    )

    best_list = search_models(space, error_function, 32;
        maxpopulation=8,
        # search hyper-parameters
        bsize=2, mutbsize=8, crossbsize=8,
        tol=0.0, maxiters=30, verbose=true)

    for (i, b) in enumerate(best_list)
        @info "-- perf best_lists[$i]:", i, b[1], b[2]
    end

    cls = MicroTC(best_list[1][1], traincorpus, trainlabels)
    sc = classification_scores(testlabels, predict_corpus(cls, testcorpus))
    @info "*** Performance on test: " sc
    @test sc.accuracy > 0.6

    cls_ = JSON3.read(JSON3.write(cls), typeof(cls))
    sc = classification_scores(testlabels, predict_corpus(cls, testcorpus))
    @info "*** Performance on test: " sc
    @test sc.accuracy > 0.6

end

# flush(stdout); flush(stderr)
# sort!(P, :score, rev=true)
# plot(P.score, label="", xlabel="rank", ylabel="score", markershape=:auto, title="RS performance on validation set") |> display
