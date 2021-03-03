# This file is a part of TextClassification.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export LiblinearConfig, LiblinearConfigSpace

using LIBLINEAR
StructTypes.StructType(::Type{<:LIBLINEAR.LinearModel}) = StructTypes.Struct()

@with_kw struct LiblinearConfig
    C::Float64 = 1.0
    eps::Float64 = 0.1
end

StructTypes.StructType(::Type{LiblinearConfig}) = StructTypes.Struct()

create(config::LiblinearConfig, train_X, train_y, dim) =
    linear_train(train_y.refs, sparse(train_X, dim); C=config.C, eps=config.eps, bias=1.0)

@with_kw struct LiblinearConfigSpace <: AbstractSolutionSpace
    C = [1.0]
    eps = [0.1]
    scale_C = (lower=0.001, s=3.0, upper=1000.0)
    scale_eps = (lower=0.0001, s=3.0, upper=0.9)
end

Base.eltype(::LiblinearConfigSpace) = LiblinearConfig

function random_configuration(space::LiblinearConfigSpace)
    LiblinearConfig(rand(space.C), rand(space.eps))
end

function combine_configurations(a::LiblinearConfig, b::LiblinearConfig)
    LiblinearConfig(a.C, b.eps)
end

function mutate_configuration(space::LiblinearConfigSpace, a::LiblinearConfig, iter)
    LiblinearConfig(
        SearchModels.scale(a.C; space.scale_C...),
        SearchModels.scale(a.eps; space.scale_eps...)
    )
end

function predict(cls::LIBLINEAR.LinearModel, vec::SVEC)
    ypred = linear_predict(cls, sparse([vec], cls.nr_feature))
    ypred[1][1]
end
