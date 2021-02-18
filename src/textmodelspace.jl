# This file is a part of TextClassification.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export EntModelConfigSpace, VectorModelConfigSpace
struct EntModelConfig{W_<:WeightingType}
    weighting::W_
    minocc::Int
    smooth::Float64
    keeptop::Float64
    classweights
end

StructTypes.StructType(::Type{<:EntModelConfig}) = StructTypes.Struct()

struct EntModelConfigSpace <: AbstractConfigSpace
    weighting::Vector{WeightingType}
    minocc::Vector{Int}
    keeptop::Vector{Float64}
    smooth::Vector{Float64}
    classweights::Vector
end

Base.eltype(::EntModelConfigSpace) = EntModelConfig

EntModelConfigSpace(;
    weighting::Vector{WeightingType}=[EntWeighting(), EntTpWeighting(), EntTpWeighting()],
    minocc::Vector{Int}=[1, 3, 7],
    keeptop::Vector{Float64}=[1.0],
    smooth::Vector{Float64}=[0.0, 1.0, 3.0],
    classweights::Vector=[:balance, :none]
) = EntModelConfigSpace(weighting, minocc, keeptop, smooth, classweights)

function random_configuration(space::EntModelConfigSpace)
    EntModelConfig(
        rand(space.weighting),
        rand(space.minocc),
        rand(space.smooth),
        rand(space.keeptop),
        rand(space.classweights)
    )
end

function combine_configurations(a::EntModelConfig, b::EntModelConfig)
    L = [a, b]

    EntModelConfig(
        rand(L).weighting,
        rand(L).minocc,
        rand(L).smooth,
        rand(L).keeptop,
        rand(L).classweights
    )
end

function mutate_configuration(::AbstractConfigSpace, c::EntModelConfig, iter)
    minocc = SearchModels.translate(c.minocc, 3, lower=0)
    smooth = SearchModels.translate(c.smooth, 2, lower=0)
    keeptop = SearchModels.scale(c.keeptop, lower=0.0, upper=1.0)

    EntModelConfig(
        c.weighting,
        minocc,
        smooth,
        keeptop,
        c.classweights
    )
end



struct VectorModelConfig{W_<:WeightingType}
    weighting::W_
    minocc::Int
    keeptop::Float64
end

StructTypes.StructType(::Type{<:VectorModelConfig}) = StructTypes.Struct()

struct VectorModelConfigSpace <: AbstractConfigSpace
    weighting::Vector{WeightingType}
    minocc::Vector{Int}
    keeptop::Vector{Float64}
end

Base.eltype(::VectorModelConfigSpace) = VectorModelConfig

VectorModelConfigSpace(;
    weighting::Vector{WeightingType}=[TfWeighting(), IdfWeighting(), TfidfWeighting(), FreqWeighting()],
    minocc::Vector{Int}=[1, 3, 7],
    keeptop::Vector{Float64}=[1.0],
) = VectorModelConfigSpace(weighting, minocc, keeptop)

function random_configuration(space::VectorModelConfigSpace)
    VectorModelConfig(
        rand(space.weighting),
        rand(space.minocc),
        rand(space.keeptop)
    )
end

function combine_configurations(a::VectorModelConfig, b::VectorModelConfig)
    L = [a, b]

    VectorModelConfig(
        rand(L).weighting,
        rand(L).minocc,
        rand(L).keeptop
    )
end

function mutate_configuration(::AbstractConfigSpace, c::VectorModelConfig, iter)
    minocc = SearchModels.translate(c.minocc, 3, lower=0)
    keeptop = SearchModels.scale(c.keeptop, lower=0.0, upper=1.0)
    VectorModelConfig(c.weighting, minocc, keeptop)
end
