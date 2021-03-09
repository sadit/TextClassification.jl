# This file is a part of TextClassification.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export KnnClassifierConfig, KnnClassifierConfigSpace, KnnClassifier

using StatsBase: counts
import StatsBase: predict

@with_kw struct KnnClassifierConfig
    k::Int32 = 1
    keeptop::Float32 = 1.0
end

StructTypes.StructType(::Type{<:KnnClassifierConfig}) = StructTypes.Struct()

create(config::KnnClassifierConfig, train_X, train_y, dim) = KnnClassifier(config, train_X, train_y)

@with_kw struct KnnClassifierConfigSpace <: AbstractSolutionSpace
    k=[1, 5]
    keeptop=[1.0]
    scale_k = (lower=1, s=1.5, upper=100)
    scale_keeptop = (lower=1.0, s=1.5, upper=1.0)
end

Base.eltype(::KnnClassifierConfigSpace) = KnnClassifierConfig

function Base.rand(space::KnnClassifierConfigSpace)
    KnnClassifierConfig(rand(space.k), rand(space.keeptop))
end

function combine(a::KnnClassifierConfig, b::KnnClassifierConfig)
    k = div(a.k + b.k, 2)
    KnnClassifierConfig(k, a.keeptop)
end

function mutate(space::AbstractSolutionSpace, a::KnnClassifierConfig, iter)
    KnnClassifierConfig(
        SearchModels.scale(a.k; space.scale_k...),
        SearchModels.scale(a.keeptop; space.scale_keeptop...)
    )
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


function predict(cls::KnnClassifier, vec::SVEC)
    res = search(cls.index, vec, cls.config.k)
    knn_most_frequent_label(cls, res)
end

function knn_most_frequent_label(knn::KnnClassifier, res::KnnResult)
    c = counts([knn.labels.refs[p.id] for p in res], 1:length(levels(knn.labels)))
    findmax(c)[end]
end

