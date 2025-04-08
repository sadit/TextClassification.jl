using Test, StatsBase, SearchModels, TextClassification
using Downloads, Random, MLUtils, JSON, CodecZlib

Random.seed!(1)

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

    (traincorpus, trainlabels), (testcorpus, testlabels) = splitobs(shuffleobs((corpus, labels)); at=0.7)
    for t in traincorpus[1:10]
        @show t
    end

    @show countmap(trainlabels)
    @show countmap(testlabels)

    cls, best_list = microtc(traincorpus, trainlabels)

    for (i, b) in enumerate(best_list)
        @info "-- perf best_lists[$i]:", i, b[1], b[2]
    end

    sc = classification_scores(testlabels, predict_corpus(cls, testcorpus))
    @info "*** Performance on test: " sc
    @test sc.accuracy > 0.6


    cls, best_list = microtc_kfolds(traincorpus, trainlabels)

    for (i, b) in enumerate(best_list)
        @info "-- microtc_kfolds - perf best_lists[$i]:", i, b[1], b[2]
    end

    sc = classification_scores(testlabels, predict_corpus(cls, testcorpus))
    @info "*** Performance microtc_kfolds on test: " sc
    @test sc.accuracy > 0.6
end
