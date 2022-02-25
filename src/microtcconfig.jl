# This file is a part of TextClassification.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt
export MicroTC_Config, MicroTC_ConfigSpace, TextConfigSpace

struct MicroTC_Config{T_, C_}
    textconfig::TextConfig
    textmodel::T_
    cls::C_
end

function MicroTC_Config(;
        textconfig=TextConfig(),
        textmodel=EntModelConfig(),
        cls=LiblinearConfig(1.0, 0.1)
    )
    
    MicroTC_Config(textconfig, textmodel, cls)
end

function Base.show(io::IO, config::MicroTC_Config) 
    print(io, "{MicroTC_Config ")
    show(io, config.textconfig)
    print(io, " ")
    show(io, config.textmodel)
    print(io, " ")
    show(io, typeof(config.cls))
    print(io, " }")
end

struct MicroTC_ConfigSpace <: AbstractSolutionSpace
    textconfig::TextConfigSpace
    textmodel
    cls
end

Base.eltype(::MicroTC_ConfigSpace) = MicroTC_Config
Base.hash(c::MicroTC_Config) = hash(repr(c))
Base.isequal(a::MicroTC_Config, b::MicroTC_Config) = repr(a) == repr(b)

function MicroTC_ConfigSpace(;
        kernel=[k_(CosineDistance()) for k_ in [DirectKernel, ReluKernel]],
        centerselection=[
            TextCentroidSelection(),
            #MedoidSelection(dist=CosineDistance()),
            #KnnCentroidSelection(sel1=CentroidSelection(), sel2=CentroidSelection(), dist=CosineDistance())
        ],
        ncenters=[3, 7, 11],
        textmodel=[EntModelConfigSpace(), VectorModelConfigSpace()],
        cls=[
            #KncConfigSpace(centerselection=centerselection, kernel=kernel),
            ### KncPerClassConfigSpace{0.3}(centerselection=centerselection, kernel=kernel, ncenters=ncenters),
            LiblinearConfigSpace(),
            #KnnClassifierConfigSpace(k=1:2:11, keeptop=[10, 30, 100, 300, 1000, 3000])
        ],
        textconfig::TextConfigSpace = TextConfigSpace()
    )

    MicroTC_ConfigSpace(textconfig, textmodel, cls)
end

function Base.rand(space::MicroTC_ConfigSpace)
    T = space.textmodel isa AbstractArray ? rand(space.textmodel) : space.textmodel
    C = space.cls isa AbstractArray ? rand(space.cls) : space.cls

    MicroTC_Config(;
        textconfig=rand(space.textconfig),
        textmodel=rand(T),
        cls=rand(C)
    )
end

function mutate(space::MicroTC_ConfigSpace, c::MicroTC_Config, iter)
    textconfig = mutate(space.textconfig, c.textconfig, iter)
    textmodel = mutate(space.textmodel, c.textmodel, iter)
    cls = mutate(space.cls, c.cls, iter)
    MicroTC_Config(textconfig=textconfig, textmodel=textmodel, cls=cls)
end

function combine_select(a::MicroTC_Config, L::AbstractVector)
    textconfig = combine_select(a.textconfig, [b.first.textconfig => b.second for b in L])
    textmodel = combine_select(a.textmodel, [b.first.textmodel => b.second for b in L])
    cls = combine_select(a.cls, [b.first.cls => b.second for b in L])
    MicroTC_Config(textconfig=textconfig, textmodel=textmodel, cls=cls)
end
