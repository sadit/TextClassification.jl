# This file is a part of TextClassification.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export LiblinearConfig, LiblinearConfigSpace

using LIBLINEAR

@with_kw struct LiblinearConfig
    C::Float64 = 1.0
    eps::Float64 = 0.1
end

struct LibLinearWrapper{LinearModel}
    map::Dict{UInt64,Int}
    cls::LinearModel
end

StructTypes.StructType(::Type{<:LIBLINEAR.LinearModel}) = StructTypes.Struct()
StructTypes.StructType(::Type{<:LibLinearWrapper}) = StructTypes.Struct()
StructTypes.StructType(::Type{LiblinearConfig}) = StructTypes.Struct()

function create(config::LiblinearConfig, train_X, train_y, dim)
    map = Dict{UInt64,Int}()
    train_X_ = [Dict(get!(map, k, length(map)+1) => v for (k, v) in x) for x in train_X]
    cls = linear_train(train_y.refs, sparse(train_X_, dim); C=config.C, eps=config.eps, bias=1.0)
    LibLinearWrapper(map, cls)
end

@with_kw struct LiblinearConfigSpace <: AbstractSolutionSpace
    C = [1.0]
    eps = [0.1]
    scale_C = (lower=0.001, s=3.0, upper=1000.0)
    scale_eps = (lower=0.0001, s=3.0, upper=0.99)
end

Base.eltype(::LiblinearConfigSpace) = LiblinearConfig

function Base.rand(space::LiblinearConfigSpace)
    LiblinearConfig(rand(space.C), rand(space.eps))
end

function combine(a::LiblinearConfig, b::LiblinearConfig)
    LiblinearConfig(a.C, b.eps)
end

function mutate(space::LiblinearConfigSpace, a::LiblinearConfig, iter)
    LiblinearConfig(
        SearchModels.scale(a.C; space.scale_C...),
        SearchModels.scale(a.eps; space.scale_eps...)
    )
end

function predict(w::LibLinearWrapper, vec::SVEC)
    v = Dict(get(w.map, k, rand(typemin(Int32):1)) => v for (k, v) in vec)
    ypred = linear_predict(w.cls, sparse([v], w.cls.nr_feature))
    ypred[1][1]
end
