# This file is a part of TextClassification.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export EntModelConfigSpace, VectorModelConfigSpace


@with_kw struct EntModelConfig{W_<:LocalWeighting}
    local_weighting::W_ = TfWeighting()
    minocc::Int = 1
    smooth::Float64 = 1.0
    keeptop::Float64 = 1.0
    classweights = :balance
end

StructTypes.StructType(::Type{<:EntModelConfig}) = StructTypes.Struct()

function create(c::EntModelConfig, train_X::AbstractVector{BOW}, train_y::CategoricalArray)
    ent = EntropyWeighting(smooth=c.smooth, lowerweight=0.0, weights=c.classweights)
    model = VectorModel(c.local_weighting, ent, train_X, train_y, minocc=c.minocc)
    c.keeptop < 1.0 ? prune_select_top(model, c.keeptop) : model
end

@with_kw struct EntModelConfigSpace <: AbstractSolutionSpace
    local_weighting = [TfWeighting(), TpWeighting(), FreqWeighting(), BinaryLocalWeighting()]
    minocc = 1:2:11
    keeptop = 0.5:0.1:1.0
    smooth = 0.0:1.0:7.0
    classweights = [:balance, :none]
    scale_minocc = (lower=1, s=1.3, upper=30)
    scale_keeptop = (lower=0.01, s=1.1, upper=1.0)
    scale_smooth = (lower=0.0, s=1.1, upper=30.0)
end

Base.eltype(::EntModelConfigSpace) = EntModelConfig

function random_configuration(space::EntModelConfigSpace)
    EntModelConfig(
        rand(space.local_weighting),
        rand(space.minocc),
        rand(space.smooth),
        rand(space.keeptop),
        rand(space.classweights)
    )
end

function combine_configurations(a::EntModelConfig, b::EntModelConfig)
    L = [a, b]

    EntModelConfig(
        rand(L).local_weighting,
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
        c.local_weighting,
        minocc,
        smooth,
        keeptop,
        c.classweights
    )
end

@with_kw struct VectorModelConfig{L_<:LocalWeighting, G_<:GlobalWeighting}
    local_weighting::L_ = TfWeighting()
    global_weighting::G_ = IdfWeighting()
    minocc::Int = 1
    keeptop::Float64 = 1.0
end

StructTypes.StructType(::Type{<:VectorModelConfig}) = StructTypes.Struct()

function create(c::VectorModelConfig, train_X::AbstractVector{BOW}, train_y::CategoricalArray)
    model = VectorModel(c.local_weighting, c.global_weighting, train_X, minocc=c.minocc)
    c.keeptop < 1.0 ? prune_select_top(model, c.keeptop) : model
end

@with_kw struct VectorModelConfigSpace <: AbstractSolutionSpace
    local_weighting = [TfWeighting(), TpWeighting(), FreqWeighting(), BinaryLocalWeighting()]
    global_weighting = [IdfWeighting(), BinaryGlobalWeighting()]
    minocc = 1:5
    keeptop = 0.5:0.1:1.0
    scale_minocc = (lower=1, s=1.3, upper=30)
    scale_keeptop = (lower=0.01, s=1.3, upper=1.0)
end

Base.eltype(::VectorModelConfigSpace) = VectorModelConfig

function random_configuration(space::VectorModelConfigSpace)
    VectorModelConfig(
        rand(space.local_weighting),
        rand(space.global_weighting),
        rand(space.minocc),
        rand(space.keeptop)
    )
end

function combine_configurations(a::VectorModelConfig, b::VectorModelConfig)
    L = [a, b]

    VectorModelConfig(
        rand(L).local_weighting,
        rand(L).global_weighting,
        rand(L).minocc,
        rand(L).keeptop
    )
end

function mutate_configuration(space::AbstractSolutionSpace, c::VectorModelConfig, iter)
    minocc = SearchModels.scale(c.minocc; space.scale_minocc...)
    keeptop = SearchModels.scale(c.keeptop; space.scale_keeptop...)
    VectorModelConfig(c.local_weighting, c.global_weighting, minocc, keeptop)
end
