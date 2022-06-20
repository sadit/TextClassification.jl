# This file is part of TextClassification.jl

export EntModelConfigSpace, VectorModelConfigSpace


@with_kw struct EntModelConfig{W_<:LocalWeighting}
    local_weighting::W_ = TfWeighting()
    mindocs::Int = 1.0
    smooth::Float64 = 0.0
    keeptop::Float64 = 1.0
    weights::Union{Symbol,Vector{Float64}} = :balance
end

function create(c::EntModelConfig, textconfig::TextConfig, corpus, train_y; minbatch=0)
    model = VectorModel(EntropyWeighting(), c.local_weighting, textconfig, corpus, train_y; mindocs=c.mindocs, smooth=c.smooth, weights=c.weights, minbatch)
    c.keeptop < 1.0 ? prune_select_top(model, c.keeptop) : model
end

@with_kw struct EntModelConfigSpace <: AbstractSolutionSpace
    #local_weighting = [TfWeighting(), TpWeighting(), FreqWeighting(), BinaryLocalWeighting()]
    local_weighting = [TfWeighting(), TpWeighting(), FreqWeighting(), BinaryLocalWeighting()]
    mindocs = 1:5
    smooth = 0:3 # by default=1 in favor of mindocs
    keeptop = [0.5, 1.0]
    weights = [:balance, :none]
    scale_mindocs = (lower=1, s=1.3, upper=11)
    scale_smooth = (lower=0.0, s=1.3, upper=11) 
    scale_keeptop = (lower=0.01, s=1.1, upper=1.0)
end

Base.eltype(::EntModelConfigSpace) = EntModelConfig

function Base.rand(space::EntModelConfigSpace)
    EntModelConfig(
        rand(space.local_weighting),
        rand(space.mindocs),
        rand(space.smooth),
        rand(space.keeptop),
        rand(space.weights)
    )
end

function combine(a::EntModelConfig, b::EntModelConfig)
    EntModelConfig(
        a.local_weighting,
        div(a.mindocs + b.mindocs, 2),
        (a.smooth + b.smooth) / 2,
        (a.keeptop + b.keeptop) / 2,
        b.weights
    )
end

function mutate(space::EntModelConfigSpace, c::EntModelConfig, iter)
    mindocs = space.scale_mindocs === nothing ? c.mindocs : SearchModels.scale(c.mindocs; space.scale_mindocs...)
    smooth = space.scale_smooth === nothing ? c.smooth : SearchModels.scale(c.smooth; space.scale_smooth...)
    keeptop = space.scale_keeptop === nothing ? c.keeptop : SearchModels.scale(c.keeptop; space.scale_keeptop...)
    lw = SearchModels.change(c.local_weighting, space.local_weighting, p1=0.3)
    
    EntModelConfig(
        lw,
        mindocs,
        smooth,
        keeptop,
        c.weights
    )
end

@with_kw struct VectorModelConfig{L_<:LocalWeighting, G_<:GlobalWeighting}
    global_weighting::G_ = IdfWeighting()
    local_weighting::L_ = TfWeighting()
    mindocs = 1
    keeptop::Float64 = 1.0
end

function create(c::VectorModelConfig, textconfig::TextConfig, train_corpus::AbstractVector, train_y; minbatch=0)
    model = VectorModel(c.global_weighting, c.local_weighting, textconfig, train_corpus; minbatch)
    c.keeptop < 1.0 ? prune_select_top(model, c.keeptop) : model
end

@with_kw struct VectorModelConfigSpace <: AbstractSolutionSpace
    global_weighting = [IdfWeighting(), BinaryGlobalWeighting()]
    local_weighting = [TfWeighting(), TpWeighting(), FreqWeighting(), BinaryLocalWeighting()]
    mindocs = [1, 4, 7]
    keeptop = [0.5, 1.0]
    scale_mindocs = (lower=1, s=1.3, upper=30)
    scale_keeptop = (lower=0.01, s=1.3, upper=1.0)
end

Base.eltype(::VectorModelConfigSpace) = VectorModelConfig

function Base.rand(space::VectorModelConfigSpace)
    VectorModelConfig(
        rand(space.global_weighting),
        rand(space.local_weighting),
        rand(space.mindocs),
        rand(space.keeptop)
    )
end

function combine(a::VectorModelConfig, b::VectorModelConfig)
    VectorModelConfig(
        b.global_weighting,
        a.local_weighting,
        div(a.mindocs + b.mindocs, 2),
        (a.keeptop + b.keeptop) / 2
    )
end

function mutate(space::VectorModelConfigSpace, c::VectorModelConfig, iter)
    mindocs = space.scale_mindocs === nothing ? c.mindocs : SearchModels.scale(c.mindocs; space.scale_mindocs...)
    keeptop = space.scale_keeptop === nothing ? c.keeptop : SearchModels.scale(c.keeptop; space.scale_keeptop...)
    lw = SearchModels.change(c.local_weighting, space.local_weighting, p1=0.3)
    gw = SearchModels.change(c.global_weighting, space.global_weighting, p1=0.3)

    VectorModelConfig(gw, lw, mindocs, keeptop)
end
