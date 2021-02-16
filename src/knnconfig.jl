# This file is a part of TextClassification.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export KnnClassifierConfig, KnnClassifierConfigSpace

struct KnnClassifierConfig <: AbstractConfig
    k::Int
    keeptop::Int
end

StructTypes.StructType(::Type{<:KnnClassifierConfig}) = StructTypes.Struct()

struct KnnClassifierConfigSpace <: AbstractConfigSpace
    k::Vector{Int}
    keeptop::Vector{Int}
end

Base.eltype(::KnnClassifierConfigSpace) = KnnClassifierConfig
KnnClassifierConfigSpace(; k=[1], keeptop=[30, 100, typemax(Int)]) = KnnClassifierConfigSpace(k, keeptop)

function random_configuration(space::KnnClassifierConfigSpace)
    KnnClassifierConfig(rand(space.k), rand(space.keeptop))
end

function combine_configurations(a::KnnClassifierConfig, b::KnnClassifierConfig)
    k = div(a.k + b.k, 2)
    KnnClassifierConfig(k, a.keeptop)
end

struct KnnClassifier
    config::KnnClassifierConfig
    index::InvIndex
    labels::CategoricalArray
end

function KnnClassifier(config::KnnClassifierConfig, X, y::CategoricalArray)
    invindex = InvIndex(X)
    invindex = config.keeptop < typemax(Int) ? prune(invindex, config.keeptop) : invindex
    KnnClassifier(config, invindex, y)
end

StructTypes.StructType(::Type{<:KnnClassifier}) = StructTypes.Struct()

using StatsBase: counts
import StatsBase: predict

function predict(cls::KnnClassifier, vec::SVEC)
    res = search(cls.index, vec, cls.config.k)
    knn_most_frequent_label(cls, res)
end

function knn_most_frequent_label(knn::KnnClassifier, res::KnnResult)
    c = counts([knn.labels.refs[p.id] for p in res], 1:length(levels(knn.labels)))
    findmax(c)[end]
end