using Test, TextClassification, StatsBase

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
    best_list = microtc_search_params(
        corpus, y,
        microtc_random_configurations(
            8,
            qlist=filtered_power_set([3, 5], 1, 2),
            nlist=filtered_power_set([1, 2], 0, 1),
            slist=[],
            ncenters=[0]
        ),
        bsize=3, ssize=8, tol=0.001, maxiters=5, folds=3, verbose=true)
    for (i, b) in enumerate(best_list)
        @info i, b[1], b[2]
    end
end


# flush(stdout); flush(stderr)
# sort!(P, :score, rev=true)
# plot(P.score, label="", xlabel="rank", ylabel="score", markershape=:auto, title="RS performance on validation set") |> display
