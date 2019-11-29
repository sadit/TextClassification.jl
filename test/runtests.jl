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
    best_list = microtc_search_params(corpus, y, microtc_random_configurations(16), bsize=4, ssize=16, maxiters=3, folds=3, verbose=true)
    for (i, b) in enumerate(best_list)
        @info i, b[1], b[2]
    end
    
end


# flush(stdout); flush(stderr)
# sort!(P, :score, rev=true)
# plot(P.score, label="", xlabel="rank", ylabel="score", markershape=:auto, title="RS performance on validation set") |> display
