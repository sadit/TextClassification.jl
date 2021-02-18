# This file is a part of TextClassification.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export KnnClassifierConfig, KnnClassifierConfigSpace

struct KnnClassifierConfig
    k::Int
    keeptop::Int
end

StructTypes.StructType(::Type{<:KnnClassifierConfig}) = StructTypes.Struct()

struct KnnClassifierConfigSpace <: AbstractSolutionSpace
    k::Vector{Int}
    keeptop::Vector{Int}
end

Base.eltype(::KnnClassifierConfigSpace) = KnnClassifierConfig

KnnClassifierConfigSpace(; k=[1], keeptop=[30, 100, typemax(Int)]) =
    KnnClassifierConfigSpace(k, keeptop)

function random_configuration(space::KnnClassifierConfigSpace)
    KnnClassifierConfig(rand(space.k), rand(space.keeptop))
end

function combine_configurations(a::KnnClassifierConfig, b::KnnClassifierConfig)
    k = div(a.k + b.k, 2)
    KnnClassifierConfig(k, a.keeptop)
end

function mutate_configuration(space::AbstractSolutionSpace, a::KnnClassifierConfig, iter)
    KnnClassifierConfig(SearchModels.translate(a.k, 2, lower=1, upper=33), SearchModels.scale(a.keeptop, lower=0.1, upper=1.0))
end
