# This file is a part of TextClassification.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt
export MicroTC_Config, MicroTC_ConfigSpace, AKNC_ConfigSpace, TextConfigSpace


struct MicroTC_Config{TextConfig_<:TextConfig, TextModel_<:TextModel, WeightingType_<:WeightingType}
    textconfig::TextConfig_
    textmodel::Type{TextModel_}
    weighting::Type{WeightingType_}
    akncconfig::AKNC_Config

    p::Float64
    smooth::Float64
    minocc::Int        
    classweights::Symbol
end

function MicroTC_Config(;
        textconfig::TextConfig=TextConfig(),
        textmodel::Type=EntModel,
        weighting::Type=EntWeighting,
        akncconfig::AKNC_Config=AKNC_Config(),
        p::Real=1.0,
        smooth::AbstractFloat=3.0,
		minocc::Integer=1,
        classweights=:balance
    )
    
    MicroTC_Config(textconfig, textmodel, weighting, akncconfig, p, smooth, minocc, classweights)
end

Base.copy(c::MicroTC_Config;
        textconfig=c.textconfig,
        textmodel=c.textmodel,
        weighting=c.weighting,
        akncconfig=c.akncconfig,
        p=c.p,
        smooth=c.smooth,
        minocc=c.minocc,
        recall=c.recall,
        classweights=c.classweights
    ) = MicroTC_Config(textconfig, textmodel, weighting, akncconfig, p, smooth, minocc, classweights)

Base.hash(a::MicroTC_Config) = hash(repr(a))
Base.isequal(a::MicroTC_Config, b::MicroTC_Config) = isequal(repr(a), repr(b))

struct MicroTC_ConfigSpace
    textmodel::Vector
    weighting::Dict
    classweights::Vector
    textconfig::TextConfigSpace
    akncconfig::AKNC_ConfigSpace
    p::Vector
    smooth::Vector
    minocc::Vector
end

Base.copy(s::MicroTC_ConfigSpace;
        textmodel=s.textmodel,
        weigthing=s.weigthing,
        classweights=s.classweights,
        textconfig=s.textconfig,
        akncconfig=s.akncconfig,
        p=s.p,
        smooth=s.smooth,
        minocc=s.minocc
    ) = MicroTC_ConfigSpace(textmodel, weigthing, classweights, textconfig, akncconfig, p, smooth, minocc)

function MicroTC_ConfigSpace(;
        textmodel::Vector = [EntModel, VectorModel],
        weighting::Dict = Dict(
            EntModel => [EntWeighting, EntTpWeighting, EntTpWeighting],
            VectorModel => [TfWeighting, IdfWeighting, TfidfWeighting, FreqWeighting]),
        classweights::Vector = [:balance, :none],
        textconfig::TextConfigSpace = TextConfigSpace(),
        akncconfig::AKNC_ConfigSpace = AKNC_ConfigSpace(),
        p::Vector{Float64} = [1.0],
        smooth::Vector = [0, 1, 3],
        minocc::Vector = [1, 3, 7]
    )

    MicroTC_ConfigSpace(textmodel, weighting, classweights, textconfig, akncconfig, p, smooth, minocc)
end

function random_configuration(space::MicroTC_ConfigSpace)
    textmodel = rand(space.textmodel)
    weighting = rand(space.weighting[textmodel])
    smooth = textmodel == EntModel ? Float64(rand(space.smooth)) : 0.0
    classweights = textmodel == EntModel ? rand(space.classweights) : :balance

    MicroTC_Config(
        textconfig=random_configuration(space.textconfig),
        textmodel=textmodel,
        weighting=weighting,
        classweights=classweights,
        akncconfig=random_configuration(space.akncconfig),
        p=rand(space.p),
        smooth=smooth,
		minocc=rand(space.minocc)
    )
end

function combine_configurations(config_list::AbstractVector{MicroTC_Config})
    _sel() = rand(config_list)
    a = _sel()
    MicroTC_Config(
        textconfig=combine_configurations(TextConfig[c.textconfig for c in config_list]),
        textmodel=a.textmodel,
        weighting=a.weighting,
        classweights=_sel().classweights,
        akncconfig=combine_configurations(AKNC_Config[c.akncconfig for c in config_list]),
        p=_sel().p,
        smooth=_sel().smooth,
		minocc=_sel().minocc
    )
end