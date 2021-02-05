# This file is a part of TextClassification.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt
export MicroTC_Config, MicroTC_ConfigSpace, TextConfigSpace

struct MicroTC_Config{T_<:AbstractConfig, C_<:AbstractConfig} <: AbstractConfig
    textconfig::TextConfig
    textmodel::T_
    cls::C_
end

StructTypes.StructType(::Type{<:MicroTC_Config}) = StructTypes.Struct()

function MicroTC_Config(;
        textconfig=TextConfig(),
        textmodel=EntModelConfig(),
        cls=LiblinearConfig(1.0, 0.1)
    )
    
    MicroTC_Config(textconfig, textmodel, cls)
end

struct MicroTC_ConfigSpace <: AbstractConfigSpace
    textconfig::TextConfigSpace
    textmodel #Union{Vector,AbstractConfigSpace}
    cls
end

Base.eltype(::MicroTC_ConfigSpace) = MicroTC_Config

function MicroTC_ConfigSpace(;
        kernel=[k_(CosineDistance()) for k_ in [DirectKernel, ReluKernel]],
        k::Vector=[1],
        maxiters::Vector=[1, 3, 10],
        recall::Vector=[1.0],
        ncenters::Vector=[0, 7],
        initial_clusters::Vector=[:fft, :dnet, :rand],
        split_entropy::Vector=[0.3, 0.6, 0.9],
        minimum_elements_per_region::Vector=[1, 3, 5],
        centerselection=[
            CentroidSelection(),
            MedoidSelection(dist=CosineDistance()),
            KnnCentroidSelection(sel=CentroidSelection(), dist=CosineDistance())
        ],
        textmodel::Vector=[EntModelConfigSpace(), VectorModelConfigSpace()],
        cls::Vector=[
            KncConfigSpace(
                centerselection=centerselection,
                kernel=kernel,
                k=k,
                maxiters=maxiters,
                recall=recall,
                ncenters=ncenters,
                initial_clusters=initial_clusters,
                split_entropy=split_entropy,
                minimum_elements_per_region=minimum_elements_per_region
            ),
            LiblinearConfigSpace(
                [100.0, 10.0, 1.0, 0.1, 0.01],
                [0.1, 0.01, 0.001]
            )
        ],
        textconfig::TextConfigSpace = TextConfigSpace()
    )

    MicroTC_ConfigSpace(textconfig, textmodel, cls)
end

function random_configuration(space::MicroTC_ConfigSpace)
    T = space.textmodel isa AbstractArray ? rand(space.textmodel) : space.textmodel
    C = space.cls isa AbstractArray ? rand(space.cls) : space.cls

    MicroTC_Config(;
        textconfig=random_configuration(space.textconfig),
        textmodel=random_configuration(T),
        cls=random_configuration(C)
    )
end

function combine_configurations(::Type{<:MicroTC_Config}, config_list)
    _sel() = rand(config_list)
    a = _sel()
    typeT = Base.typename(typeof(a.textmodel)).wrapper
    typeC = Base.typename(typeof(a.cls)).wrapper
    MicroTC_Config(
        textconfig=combine_configurations(TextConfig, [c.textconfig for c in config_list]),
        textmodel=combine_configurations(a.textmodel, [c.textmodel for c in config_list if c.textmodel isa typeT]),
        cls=combine_configurations(a.cls, [c.cls for c in config_list if c.cls isa typeC])
    )
end
