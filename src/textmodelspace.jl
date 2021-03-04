# This file is a part of TextClassification.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export EntModelConfigSpace, VectorModelConfigSpace


@with_kw struct EntModelConfig{W_<:LocalWeighting}
    local_weighting::W_ = TfWeighting()
    smooth::Float64 = 1.0
    keeptop::Float64 = 1.0
    weights::Union{Symbol,Vector{Float64}} = :balance
end

StructTypes.StructType(::Type{<:EntModelConfig}) = StructTypes.Struct()

function create(c::EntModelConfig, train_X::AbstractVector{BOW}, train_y::CategoricalArray)
    model = VectorModel(EntropyWeighting(), c.local_weighting, train_X, train_y, smooth=c.smooth, weights=c.weights)
    c.keeptop < 1.0 ? prune_select_top(model, c.keeptop) : model
end

@with_kw struct EntModelConfigSpace <: AbstractSolutionSpace
    local_weighting = [TfWeighting(), TpWeighting(), FreqWeighting(), BinaryLocalWeighting()]
    keeptop = [0.5, 1.0]
    smooth = [1.0, 3.0]
    weights = [:balance, :none]
    scale_keeptop = (lower=0.01, s=1.1, upper=1.0)
    scale_smooth = (lower=0.0, s=1.1, upper=30.0)
end

Base.eltype(::EntModelConfigSpace) = EntModelConfig

function random_configuration(space::EntModelConfigSpace)
    EntModelConfig(
        rand(space.local_weighting),
        rand(space.smooth),
        rand(space.keeptop),
        rand(space.weights)
    )
end

function combine_configurations(a::EntModelConfig, b::EntModelConfig)

    EntModelConfig(
        a.local_weighting,
        (a.smooth + b.smooth) / 2,
        (a.keeptop + b.keeptop) / 2,
        b.weights
    )
end

function mutate_configuration(space::AbstractSolutionSpace, c::EntModelConfig, iter)
    smooth = SearchModels.scale(c.smooth; space.scale_smooth...)
    keeptop = SearchModels.scale(c.keeptop; space.scale_keeptop...)
    lw = SearchModels.change(c.local_weighting, space.local_weighting, p1=0.3)

    EntModelConfig(
        lw,
        smooth,
        keeptop,
        c.weights
    )
end

@with_kw struct VectorModelConfig{L_<:LocalWeighting, G_<:GlobalWeighting}
    global_weighting::G_ = IdfWeighting()
    local_weighting::L_ = TfWeighting()
    keeptop::Float64 = 1.0
end

StructTypes.StructType(::Type{<:VectorModelConfig}) = StructTypes.Struct()

function create(c::VectorModelConfig, train_X::AbstractVector{BOW}, train_y::CategoricalArray)
    model = VectorModel(c.global_weighting, c.local_weighting, train_X)
    c.keeptop < 1.0 ? prune_select_top(model, c.keeptop) : model
end

@with_kw struct VectorModelConfigSpace <: AbstractSolutionSpace
    global_weighting = [IdfWeighting(), BinaryGlobalWeighting()]
    local_weighting = [TfWeighting(), TpWeighting(), FreqWeighting(), BinaryLocalWeighting()]
    keeptop = [0.5, 1.0]
    scale_keeptop = (lower=0.01, s=1.3, upper=1.0)
end

Base.eltype(::VectorModelConfigSpace) = VectorModelConfig

function random_configuration(space::VectorModelConfigSpace)
    VectorModelConfig(
        rand(space.global_weighting),
        rand(space.local_weighting),
        rand(space.keeptop)
    )
end

function combine_configurations(a::VectorModelConfig, b::VectorModelConfig)
    VectorModelConfig(
        b.global_weighting,
        a.local_weighting,
        (a.keeptop + b.keeptop) / 2
    )
end

function mutate_configuration(space::VectorModelConfigSpace, c::VectorModelConfig, iter)
    keeptop = SearchModels.scale(c.keeptop; space.scale_keeptop...)
    lw = SearchModels.change(c.local_weighting, space.local_weighting, p1=0.3)
    gw = SearchModels.change(c.global_weighting, space.global_weighting, p1=0.3)

    VectorModelConfig(gw, lw, keeptop)
end
