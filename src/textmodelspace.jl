# This file is a part of TextClassification.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export EntModelConfigSpace, VectorModelConfigSpace


@with_kw struct EntModelConfig{W_<:LocalWeighting}
    local_weighting::W_ = TfWeighting()
    minocc::Int = 1
    smooth::Float64 = 1.0
    keeptop::Float64 = 1.0
    weights::Union{Symbol,Vector{Float64}} = :balance
end

StructTypes.StructType(::Type{<:EntModelConfig}) = StructTypes.Struct()

function create(c::EntModelConfig, train_X::AbstractVector{BOW}, train_y::CategoricalArray)
    model = VectorModel(c.local_weighting, EntropyWeighting(), train_X, train_y, minocc=c.minocc, smooth=c.smooth, lowerweight=0.0, weights=c.weights)
    c.keeptop < 1.0 ? prune_select_top(model, c.keeptop) : model
end

@with_kw struct EntModelConfigSpace <: AbstractSolutionSpace
    local_weighting = [TfWeighting(), TpWeighting(), FreqWeighting(), BinaryLocalWeighting()]
    minocc = [1, 3, 5]
    keeptop = [0.5, 1.0]
    smooth = [1.0, 3.0]
    weights = [:balance, :none]
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
        rand(space.weights)
    )
end

function combine_configurations(a::EntModelConfig, b::EntModelConfig)
    L = [a, b]

    EntModelConfig(
        rand(L).local_weighting,
        rand(L).minocc,
        rand(L).smooth,
        rand(L).keeptop,
        rand(L).weights
    )
end

function mutate_configuration(space::AbstractSolutionSpace, c::EntModelConfig, iter)
    minocc = SearchModels.scale(c.minocc; space.scale_minocc...)
    smooth = SearchModels.scale(c.smooth; space.scale_smooth...)
    keeptop = SearchModels.scale(c.keeptop; space.scale_keeptop...)
    lw = rand() < 0.3 ? rand(space.local_weighting) : c.local_weighting

    EntModelConfig(
        lw,
        minocc,
        smooth,
        keeptop,
        c.weights
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
    minocc = [1, 3, 5]
    keeptop = [0.5, 1.0]
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
        a.local_weighting,
        b.global_weighting,
        div(a.minocc + b.minocc, 2),
        (a.keeptop + b.keeptop) / 2
    )
end

function mutate_configuration(space::VectorModelConfigSpace, c::VectorModelConfig, iter)
    minocc = SearchModels.scale(c.minocc; space.scale_minocc...)
    keeptop = SearchModels.scale(c.keeptop; space.scale_keeptop...)
    lw = SearchModels.change(c.local_weighting, space.local_weighting, p1=0.3)
    gw = SearchModels.change(c.global_weighting, space.global_weighting, p1=0.3)

    VectorModelConfig(lw, gw, minocc, keeptop)
end
