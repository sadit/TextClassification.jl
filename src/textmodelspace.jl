# This file is a part of TextClassification.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export EntModelConfigSpace, VectorModelConfigSpace

@with_kw struct EntModelConfig{W_<:WeightingType}
    weighting::W_ = EntWeighting()
    minocc::Int = 1
    smooth::Float64 = 1.0
    keeptop::Float64 = 1.0
    classweights = :balance
end

StructTypes.StructType(::Type{<:EntModelConfig}) = StructTypes.Struct()

@with_kw struct EntModelConfigSpace <: AbstractSolutionSpace
    weighting = [EntWeighting(), EntTpWeighting(), EntTpWeighting()]
    minocc = 1:3:11
    keeptop = 0.01:0.1:1.0
    smooth = 0.0:0.3:7.0
    classweights = [:balance, :none]
    scale_minocc = (lower=1, s=1.3, upper=30)
    scale_keeptop = (lower=0.01, s=1.1, upper=1.0)
    scale_smooth = (lower=0.0, s=1.1, upper=30.0)
end

Base.eltype(::EntModelConfigSpace) = EntModelConfig

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

function mutate_configuration(space::AbstractSolutionSpace, c::EntModelConfig, iter)
    minocc = SearchModels.scale(c.minocc; space.scale_minocc...)
    smooth = SearchModels.scale(c.smooth; space.scale_smooth...)
    keeptop = SearchModels.scale(c.keeptop; space.scale_keeptop...)

    EntModelConfig(
        c.weighting,
        minocc,
        smooth,
        keeptop,
        c.classweights
    )
end

@with_kw struct VectorModelConfig{W_<:WeightingType}
    weighting::W_ = TfidfWeighting()
    minocc::Int = 1
    keeptop::Float64 = 1.0
end

StructTypes.StructType(::Type{<:VectorModelConfig}) = StructTypes.Struct()

@with_kw struct VectorModelConfigSpace <: AbstractSolutionSpace
    weighting = [TfWeighting(), IdfWeighting(), TfidfWeighting(), FreqWeighting()]
    minocc = 1:3:11
    keeptop = 0.01:0.1:1.0
    scale_minocc = (lower=1, s=1.5, upper=30)
    scale_keeptop = (lower=0.01, s=1.5, upper=1.0)
end

Base.eltype(::VectorModelConfigSpace) = VectorModelConfig

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

function mutate_configuration(space::AbstractSolutionSpace, c::VectorModelConfig, iter)
    minocc = SearchModels.scale(c.minocc; space.scale_minocc...)
    keeptop = SearchModels.scale(c.keeptop; space.scale_keeptop...)
    VectorModelConfig(c.weighting, minocc, keeptop)
end
