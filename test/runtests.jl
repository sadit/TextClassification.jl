using Test, StatsBase, KCenters, TextSearch, TextClassification

@testset "microtc" begin
    !isfile("emotions.csv") && download("http://ingeotec.mx/~sadit/emotions.csv", "emotions.csv")
    using CSV, Random, MLDataUtils
    X = CSV.read("emotions.csv")
    L = ["♡", "💔"]
    X = X[[l in L for l in X.klass], :]
    labels = X.klass
    le = labelenc(labels)
    y = label2ind.(labels, le)
    corpus = X.text

    (Xtrain, ytrain), (Xtest, ytest) = splitobs(shuffleobs((corpus, y)), at=0.7)
    best_list = microtc_search_params(
        Xtrain, ytrain, 8;
        # search hyper-parameters
        tol=0.01, search_maxiters=3, folds=3, verbose=true,
        # configuration space
        ncenters=[0],
        qlist=filtered_power_set([3, 5], 1, 2),
        nlist=filtered_power_set([1, 2], 0, 1),
        slist=[],
        kind = [EntModel]
     )

    for (i, b) in enumerate(best_list)
        @info i, b[1], b[2]
    end

    cls = fit(μTC, best_list[1][1], Xtrain, ytrain)
    sc = scores(predict(cls, Xtest), ytest)
    @info "*** Performance on test: " sc
    @test sc.accuracy > 0.7

    cls = bagging(best_list[1][1], Xtrain, ytrain; b=15, ratio=0.85)
    sc = scores(predict(cls, Xtest), ytest)
    @info "*** Bagging performance on test: " sc
    @test sc.accuracy > 0.7

    sc = scores(predict(cls, Xtest, 3), ytest)
    @info "*** Bagging performance on test (k=3): " sc
    @test sc.accuracy > 0.7

    B = optimize!(cls, Xtest, ytest; verbose=false)
    @info "best config for optimize!" B[1]
    sc = scores(predict(cls, Xtest), ytest)
    @info "*** Bagging performance on test (after calling optimize!): " sc
    @test sc.accuracy > 0.7
end


# flush(stdout); flush(stderr)
# sort!(P, :score, rev=true)
# plot(P.score, label="", xlabel="rank", ylabel="score", markershape=:auto, title="RS performance on validation set") |> display
