# This file is a part of TextClassification.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt
export MicroTC_Config, MicroTC_ConfigSpace, TextConfigSpace

struct MicroTC_Config{WeightingType_<:WeightingType, Config_<:Any} <: AbstractConfig
    textconfig::TextConfig
    textmodel::Symbol
    weighting::WeightingType_
    cls::Config_
    p::Float64
    smooth::Float64
    minocc::Int        
    classweights::Symbol
end

StructTypes.StructType(::Type{<:MicroTC_Config}) = StructTypes.Struct()

function MicroTC_Config(;
        textconfig::TextConfig=TextConfig(),
        textmodel::Symbol=:EntModel,
        weighting::WeightingType=EntTpWeighting(),
        cls::AbstractConfig=KncConfig(),
        #cls=LiblinearConfig(1.0, 0.1),
        p::Real=1.0,
        smooth::AbstractFloat=3.0,
		minocc::Integer=1,
        classweights=:balance
    )
    
    MicroTC_Config(textconfig, textmodel, weighting, cls, p, smooth, minocc, classweights)
end

Base.copy(c::MicroTC_Config;
        textconfig=c.textconfig,
        textmodel=c.textmodel,
        weighting=c.weighting,
        cls=c.cls,
        p=c.p,
        smooth=c.smooth,
        minocc=c.minocc,
        recall=c.recall,
        classweights=c.classweights
    ) = MicroTC_Config(textconfig, textmodel, weighting, cls, p, smooth, minocc, classweights)

struct MicroTC_ConfigSpace{ConfigSpace_<:AbstractConfigSpace} <: AbstractConfigSpace
    textmodel::Vector{Symbol}
    weighting::Dict
    classweights::Vector{Symbol}
    textconfig::TextConfigSpace
    cls::ConfigSpace_
    p::Vector
    smooth::Vector
    minocc::Vector
end

Base.copy(s::MicroTC_ConfigSpace;
        textmodel=s.textmodel,
        weigthing=s.weigthing,
        classweights=s.classweights,
        textconfig=s.textconfig,
        cls=s.cls,
        p=s.p,
        smooth=s.smooth,
        minocc=s.minocc
    ) = MicroTC_ConfigSpace(textmodel, weigthing, classweights, textconfig, cls, p, smooth, minocc)

function MicroTC_ConfigSpace(;
        textmodel::Vector = [:EntModel, :VectorModel],
        weighting::Dict = Dict(
            :EntModel => [EntWeighting(), EntTpWeighting(), EntTpWeighting()],
            :VectorModel => [TfWeighting(), IdfWeighting(), TfidfWeighting(), FreqWeighting()]
        ),
        classweights::Vector = [:balance, :none],
        textconfig::TextConfigSpace = TextConfigSpace(),

        centerselection=[
            CentroidSelection(),
            MedoidSelection(dist=CosineDistance()),
            KnnCentroidSelection(sel=CentroidSelection(), dist=CosineDistance())
        ],
        kernel=[k_(CosineDistance()) for k_ in [DirectKernel, ReluKernel]],
        k::Vector=[1],
        maxiters::Vector=[1, 3, 10],
        recall::Vector=[1.0],
        ncenters::Vector=[0, 7],
        initial_clusters::Vector=[:fft, :dnet, :rand],
        split_entropy::Vector=[0.3, 0.6, 0.9],
        minimum_elements_per_region::Vector=[1, 3, 5],
        #=cls::AbstractConfigSpace = KncConfigSpace(
            centerselection=centerselection,
            kernel=kernel,
            k=k,
            maxiters=maxiters,
            recall=recall,
            ncenters=ncenters,
            initial_clusters=initial_clusters,
            split_entropy=split_entropy,
            minimum_elements_per_region=minimum_elements_per_region
        ),=#
        cls::LiblinearConfigSpace=LiblinearConfigSpace([100.0, 10.0, 1.0, 0.1, 0.01], [0.1, 0.01, 0.001]),
        p::Vector{Float64} = [1.0],
        smooth::Vector = [0, 1, 3],
        minocc::Vector = [1, 3, 7]
    )

    MicroTC_ConfigSpace(textmodel, weighting, classweights, textconfig, cls, p, smooth, minocc)
end

function random_configuration(space::MicroTC_ConfigSpace)
    textmodel = rand(space.textmodel)
    weighting = rand(space.weighting[textmodel])    
    smooth, classweights = if textmodel == :EntModel
        Float64(rand(space.smooth)), rand(space.classweights)
    else
        0.0, :balance
    end

    MicroTC_Config(;
        textconfig=random_configuration(space.textconfig),
        textmodel=textmodel,
        weighting=weighting,
        classweights=classweights,
        cls=random_configuration(space.cls),
        p=rand(space.p),
        smooth=smooth,
		minocc=rand(space.minocc)
    )
end

function combine_configurations(space::MicroTC_ConfigSpace, config_list)
    _sel() = rand(config_list)
    a = _sel()
    MicroTC_Config(
        textconfig=combine_configurations(space.textconfig, TextConfig[c.textconfig for c in config_list]),
        textmodel=a.textmodel,
        weighting=a.weighting,
        classweights=_sel().classweights,
        cls=combine_configurations(space.cls, [c.cls for c in config_list]),
        p=_sel().p,
        smooth=_sel().smooth,
		minocc=_sel().minocc
    )
end
