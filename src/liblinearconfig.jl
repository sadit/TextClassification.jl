# This file is a part of TextClassification.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export LiblinearConfig, LiblinearConfigSpace

struct LiblinearConfig <: AbstractConfig
    C::Float64
    eps::Float64
end

StructTypes.StructType(::Type{LiblinearConfig}) = StructTypes.Struct()

struct LiblinearConfigSpace <: AbstractConfigSpace
    C::Vector{Float64}
    eps::Vector{Float64}
end

LiblinearConfigSpace(;
    C=[100.0, 10.0, 1.0, 0.1, 0.01, 0.001],
    eps=[0.1, 0.01, 0.001]
) = LiblinearConfigSpace(C, eps)

Base.eltype(::LiblinearConfigSpace) = LiblinearConfig

function random_configuration(space::LiblinearConfigSpace)
    LiblinearConfig(rand(space.C), rand(space.eps))
end

function combine_configurations(::Type{LiblinearConfig}, config_list::AbstractVector)
    _sel() = rand(config_list)

    LiblinearConfig(_sel().C, _sel().eps)
end