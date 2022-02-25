# This file is part of TextClassification.jl

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

function create(config::LiblinearConfig, train_X, train_y, dim)
    map = Dict{UInt64,Int}()
    map[0] = 1 # the vectorization procedure use the zero id as an special symbol
    train_X_ = [Dict(get!(map, k, length(map)+1) => v for (k, v) in x) for x in train_X]
    # dim + 1 for out-of-vocabulary tokens
    cls = linear_train(train_y.refs, sparse(train_X_, dim + 1); C=config.C, eps=config.eps, bias=1.0)
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
    C = space.scale_C === nothing ? a.C : SearchModels.scale(a.C; space.scale_C...)
    eps = space.scale_eps === nothing ? a.eps : SearchModels.scale(a.eps; space.scale_eps...)
    LiblinearConfig(C, eps)
end

function predict(w::LibLinearWrapper, vec::SVEC)
    D = Dict(get(w.map, k, 1) => v for (k, v) in vec)
    ypred = linear_predict(w.cls, sparse([D], w.cls.nr_feature))
    ypred[1][1]
end
