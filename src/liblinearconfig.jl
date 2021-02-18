# This file is a part of TextClassification.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export LiblinearConfig, LiblinearConfigSpace

using LIBLINEAR
StructTypes.StructType(::Type{<:LIBLINEAR.LinearModel}) = StructTypes.Struct()

struct LiblinearConfig
    C::Float64
    eps::Float64
end

StructTypes.StructType(::Type{LiblinearConfig}) = StructTypes.Struct()

struct LiblinearConfigSpace <: AbstractConfigSpace
    C::Vector{Float64}
    eps::Vector{Float64}
end

LiblinearConfigSpace(;
    C=[10.0, 1.0, 0.1, 0.01],
    eps=[0.1, 0.01]
) = LiblinearConfigSpace(C, eps)

Base.eltype(::LiblinearConfigSpace) = LiblinearConfig

function random_configuration(space::LiblinearConfigSpace)
    LiblinearConfig(rand(space.C), rand(space.eps))
end

function combine_configurations(a::LiblinearConfig, b::LiblinearConfig)
    LiblinearConfig(a.C, b.eps)
end

function mutate_configuration(space::LiblinearConfigSpace, a::LiblinearConfig, iter)
    LiblinearConfig(SearchModels.scale(a.C, 10.0), SearchModels.scale(a.eps, 10.0))
end

function predict(cls::LIBLINEAR.LinearModel, vec::SVEC)
    ypred = linear_predict(cls, sparse([vec], cls.nr_feature))
    ypred[1][1]
end
