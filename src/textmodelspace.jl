export EntModelConfigSpace, VectorModelConfigSpace
struct EntModelConfig{W_<:WeightingType} <: AbstractConfig
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


struct VectorModelConfig{W_<:WeightingType} <: AbstractConfig
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

function random_configuration(space::EntModelConfigSpace)
    EntModelConfig(
        rand(space.weighting),
        rand(space.minocc),
        rand(space.smooth),
        rand(space.keeptop),
        rand(space.classweights)
    )
end

function combine_configurations(::Type{T}, configlist::AbstractVector) where {T<:EntModelConfig}
    _sel() = rand(configlist)

    EntModelConfig(
        _sel().weighting,
        _sel().minocc,
        _sel().smooth,
        _sel().keeptop,
        _sel().classweights
    )
end


function random_configuration(space::VectorModelConfigSpace)
    VectorModelConfig(
        rand(space.weighting),
        rand(space.minocc),
        rand(space.keeptop)
    )
end

function combine_configurations(::Type{<:VectorModelConfig}, configlist::AbstractVector)
    _sel() = rand(configlist)

    VectorModelConfig(
        _sel().weighting,
        _sel().minocc,
        _sel().keeptop
    )
end
