# This file is part of TextClassification.jl

export EntModelConfigSpace, VectorModelConfigSpace


@with_kw struct EntModelConfig{W_<:LocalWeighting}
    local_weighting::W_ = TfWeighting()
    smooth::Float64 = 0.0
    min_token_ndocs::Int = 5
    max_token_pdocs::Float64 = 0.7
    weights::Union{Symbol,Vector{Float64}} = :balance
end

function create(c::EntModelConfig, textconfig::TextConfig, corpus, train_y; minbatch=0)
    model = VectorModel(EntropyWeighting(), c.local_weighting, textconfig, corpus, train_y; mindocs=c.min_token_ndocs, smooth=c.smooth, weights=c.weights, minbatch)
    maxf = ceil(Int, c.max_token_pdocs * length(corpus))
    model = filter_tokens(model) do t
        c.min_token_ndocs <= t.ndocs <= maxf
    end
end

@with_kw struct EntModelConfigSpace <: AbstractSolutionSpace
    #local_weighting = [TfWeighting(), TpWeighting(), FreqWeighting(), BinaryLocalWeighting()]
    local_weighting = [TfWeighting(), TpWeighting(), FreqWeighting(), BinaryLocalWeighting()]
    smooth = 0:3 # by default=1 in favor of mindocs
    min_token_ndocs = [3, 7, 11]
    max_token_pdocs = [0.5, 0.7]
    weights = [:balance, :none]
    scale_smooth = (lower=0.0, s=1.3, upper=11)
    scale_min_token_ndocs = (lower=1, s=1.3, upper=31)
    scale_max_token_pdocs = (lower=0.5, s=1.2, upper=1.0)
end

Base.eltype(::EntModelConfigSpace) = EntModelConfig

function Base.rand(space::EntModelConfigSpace)
    EntModelConfig(
        rand(space.local_weighting),
        rand(space.smooth),
        rand(space.min_token_ndocs),
        rand(space.max_token_pdocs),
        rand(space.weights)
    )
end

function combine(a::EntModelConfig, b::EntModelConfig)
    EntModelConfig(
        a.local_weighting,
        (a.smooth + b.smooth) / 2,
        (a.min_token_ndocs + b.min_token_ndocs) ÷ 2,
        (a.max_token_pdocs + b.max_token_pdocs) ÷ 2,
        b.weights
    )
end

function mutate(space::EntModelConfigSpace, c::EntModelConfig, iter)
    smooth = space.scale_smooth === nothing ? c.smooth : SearchModels.scale(c.smooth; space.scale_smooth...)
    min_token_ndocs = space.scale_min_token_ndocs === nothing ? c.min_token_ndocs : SearchModels.scale(c.min_token_ndocs; space.scale_min_token_ndocs...)
    max_token_pdocs = space.scale_max_token_pdocs === nothing ? c.max_token_pdocs : SearchModels.scale(c.max_token_pdocs; space.scale_max_token_pdocs...)

    lw = SearchModels.change(c.local_weighting, space.local_weighting, p1=0.3)
    
    EntModelConfig(
        lw,
        smooth,
        min_token_ndocs,
        max_token_pdocs,
        c.weights
    )
end

@with_kw struct VectorModelConfig{L_<:LocalWeighting, G_<:GlobalWeighting}
    global_weighting::G_ = IdfWeighting()
    local_weighting::L_ = TfWeighting()
    min_token_ndocs::Int = 5
    max_token_pdocs::Float64 = 0.7
end

function create(c::VectorModelConfig, textconfig::TextConfig, corpus::AbstractVector, train_y; minbatch=0)
    model = VectorModel(c.global_weighting, c.local_weighting, textconfig, corpus; minbatch)
    maxf = ceil(Int, c.max_token_pdocs * length(corpus))
    filter_tokens(model) do t
        c.min_token_ndocs <= t.ndocs <= maxf
    end
end

@with_kw struct VectorModelConfigSpace <: AbstractSolutionSpace
    global_weighting = [IdfWeighting(), BinaryGlobalWeighting()]
    local_weighting = [TfWeighting(), TpWeighting(), FreqWeighting(), BinaryLocalWeighting()]
    scale_mindocs = (lower=1, s=1.3, upper=30)
    min_token_ndocs = [3, 7, 11]
    max_token_pdocs = [0.5, 0.7]
    scale_min_token_ndocs = (lower=1, s=1.3, upper=31)
    scale_max_token_pdocs = (lower=0.5, s=1.3, upper=1.0)
end

Base.eltype(::VectorModelConfigSpace) = VectorModelConfig

function Base.rand(space::VectorModelConfigSpace)
    VectorModelConfig(
        rand(space.global_weighting),
        rand(space.local_weighting),
        rand(space.min_token_ndocs),
        rand(space.max_token_pdocs),
    )
end

function combine(a::VectorModelConfig, b::VectorModelConfig)
    VectorModelConfig(
        b.global_weighting,
        a.local_weighting,
        (a.min_token_ndocs + b.min_token_ndocs) ÷ 2,
        (a.max_token_pdocs + b.max_token_pdocs) ÷ 2
    )
end

function mutate(space::VectorModelConfigSpace, c::VectorModelConfig, iter)
    min_token_ndocs = space.scale_min_token_ndocs === nothing ? c.min_token_ndocs : SearchModels.scale(c.min_token_ndocs; space.scale_min_token_ndocs...)
    max_token_pdocs = space.scale_max_token_pdocs === nothing ? c.max_token_pdocs : SearchModels.scale(c.max_token_pdocs; space.scale_max_token_pdocs...)
    lw = SearchModels.change(c.local_weighting, space.local_weighting, p1=0.3)
    gw = SearchModels.change(c.global_weighting, space.global_weighting, p1=0.3)

    VectorModelConfig(gw, lw, min_token_ndocs, max_token_pdocs)
end
