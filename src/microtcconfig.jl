# This file is a part of TextClassification.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt
export MicroTC_Config, MicroTC_ConfigSpace, TextConfigSpace

struct MicroTC_Config{T_, C_}
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

struct MicroTC_ConfigSpace <: AbstractSolutionSpace
    textconfig::TextConfigSpace
    textmodel #Union{Vector,AbstractSolutionSpace}
    cls
end

Base.eltype(::MicroTC_ConfigSpace) = MicroTC_Config

function MicroTC_ConfigSpace(;
        kernel=[k_(CosineDistance()) for k_ in [DirectKernel, ReluKernel]],
        centerselection=[
            CentroidSelection(),
            MedoidSelection(dist=CosineDistance()),
            KnnCentroidSelection(sel1=CentroidSelection(), sel2=CentroidSelection(), dist=CosineDistance())
        ],
        ncenters=[3, 7],
        textmodel=[EntModelConfigSpace(), VectorModelConfigSpace()],
        cls=[
            KncConfigSpace(centerselection=centerselection, kernel=kernel),
            # KncPerClassConfigSpace{0.3}(centerselection=centerselection, kernel=kernel, ncenters=ncenters),
            LiblinearConfigSpace(),
            KnnClassifierConfigSpace(k=1:2:11, keeptop=0.5:0.1:1.0)
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

function mutate_configuration(space::MicroTC_ConfigSpace, c::MicroTC_Config, iter)
    textconfig = mutate_configuration(space.textconfig, c.textconfig, iter)
    textmodel = mutate_configuration(space.textmodel, c.textmodel, iter)
    cls = mutate_configuration(space.cls, c.cls, iter)
    MicroTC_Config(textconfig=textconfig, textmodel=textmodel, cls=cls)
end

function combine_configurations(a::MicroTC_Config, L::AbstractVector)
    textconfig = combine_configurations(a.textconfig, [b.first.textconfig => b.second for b in L])
    textmodel = combine_configurations(a.textmodel, [b.first.textmodel => b.second for b in L])
    cls = combine_configurations(a.cls, [b.first.cls => b.second for b in L])

    MicroTC_Config(textconfig=textconfig, textmodel=textmodel, cls=cls)
end
