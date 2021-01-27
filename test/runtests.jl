using Test, StatsBase, KNearestCenters, TextSearch, TextClassification, CategoricalArrays

function train_test_split(corpus, labels, at=0.7)
    n = length(corpus)
    iX = shuffle!(collect(1:n))
    n_ = floor(Int, n * at)
    itrain, itest = iX[1:n_], iX[n_+1:end]
    corpus[itrain], labels[itrain], corpus[itest], labels[itest]
end

@testset "microtc" begin
    !isfile("emotions.csv") && download("http://ingeotec.mx/~sadit/emotions.csv", "emotions.csv")
    using CSV, Random, MLDataUtils
    X = CSV.read("emotions.csv", NamedTuple)
    targets = ["♡", "💔"]
    I = [(l in targets) for l in X.klass]
    labels = categorical(X.klass[I])
    corpus = X.text[I]

    traincorpus, trainlabels, testcorpus, testlabels = train_test_split(corpus, labels, 0.7)
    @show traincorpus[1:10]
    @show countmap(trainlabels)
    @show countmap(testlabels)

    space = MicroTC_ConfigSpace(
        minocc=[1],
        textconfig=TextConfigSpace(
            qlist=[[4]],
            nlist=[[1], []],
            slist=[]
        ),
        akncconfig=AKNC_ConfigSpace(
            ncenters=[0],
            dist=[CosineDistance]
        )
    )

    best_list = search_params(space, traincorpus, trainlabels, 8;
        # search hyper-parameters
        tol=0.01, searchmaxiters=3, folds=3, verbose=true, distributed=false)

    for (i, b) in enumerate(best_list)
        @info "-- perf best_lists[$i]:", i, b[1], b[2]
    end

    cls = MicroTC(best_list[1][1], traincorpus, trainlabels)
    sc = classification_scores(testlabels.refs, [predict(cls, t) for t in testcorpus])
    @info "*** Performance on test: " sc
    @test sc.accuracy > 0.7
end


# flush(stdout); flush(stderr)
# sort!(P, :score, rev=true)
# plot(P.score, label="", xlabel="rank", ylabel="score", markershape=:auto, title="RS performance on validation set") |> display
