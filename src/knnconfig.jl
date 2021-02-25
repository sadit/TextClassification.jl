# This file is a part of TextClassification.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export KnnClassifierConfig, KnnClassifierConfigSpace

@with_kw struct KnnClassifierConfig
    k::Int32 = 1
    keeptop::Float32 = 1.0
end

StructTypes.StructType(::Type{<:KnnClassifierConfig}) = StructTypes.Struct()

@with_kw struct KnnClassifierConfigSpace <: AbstractSolutionSpace
    k=1:2:5 # rand(k)  -> integer
    keeptop=0.1:0.1:0.3 # rand(keeptop) -> float
    scale_k = (lower=1, s=1.5, upper=100)
    scale_keeptop = (lower=0.001, s=1.5, upper=1.0)
end

Base.eltype(::KnnClassifierConfigSpace) = KnnClassifierConfig

function random_configuration(space::KnnClassifierConfigSpace)
    KnnClassifierConfig(rand(space.k), rand(space.keeptop))
end

function combine_configurations(a::KnnClassifierConfig, b::KnnClassifierConfig)
    k = div(a.k + b.k, 2)
    KnnClassifierConfig(k, a.keeptop)
end

function mutate_configuration(space::AbstractSolutionSpace, a::KnnClassifierConfig, iter)
    KnnClassifierConfig(
        SearchModels.scale(a.k; space.scale_k...),
        SearchModels.scale(a.keeptop; space.scale_keeptop...)
    )
end
