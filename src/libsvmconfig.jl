# This file is part of TextClassification.jl

export LIBSVMConfig, LIBSVMConfigSpace

using LIBSVM: svmpredict, svmtrain, Kernel

@with_kw struct LIBSVMConfig
    C::Float64 = 1.0
    weights = :balance
end

struct LIBSVMWrapper{LIBSVMModel}
    dim::Int
    cls::LIBSVMModel
end

function balanced_weights(y)
    C = countmap(y)
    s = sum(values(C))
    nc = length(C)
    Dict{Any,Float64}(label => (s / (nc * count)) for (label, count) in C)
end

function create(config::LIBSVMConfig, train_X, train_y, dim)
    train_X_ = sparse(train_X, dim)
    nt = Threads.nthreads()
    verbose = true
    kernel = Kernel.Linear
    weights = balanced_weights(train_y)
    cls = svmtrain(train_X_, train_y; nt, weights, verbose, kernel, cost=config.C, )
    LIBSVMWrapper(dim, cls)
end

@with_kw struct LIBSVMConfigSpace <: AbstractSolutionSpace
    C = [1.0]
    eps = [0.1]
    weights = [:balance, nothing]
    scale_C = (lower=0.001, s=3.0, upper=1000.0)
    scale_eps = (lower=0.0001, s=3.0, upper=0.99)
end

Base.eltype(::LIBSVMConfigSpace) = LIBSVMConfig

function Base.rand(space::LIBSVMConfigSpace)
    LIBSVMConfig(rand(space.C), rand(space.weights))
end

function combine(a::LIBSVMConfig, b::LIBSVMConfig)
    LIBSVMConfig(a.C, rand([a.weights, b.weights]))
end

function mutate(space::LIBSVMConfigSpace, a::LIBSVMConfig, iter)
    C = space.scale_C === nothing ? a.C : SearchModels.scale(a.C; space.scale_C...)
    weights = rand([a.weights, rand(space.weights)])
    LIBSVMConfig(C, weights)
end

function predict(w::LIBSVMWrapper, vec::SVEC)
    ypred = svmpredict(w.cls, sparse([vec], w.dim); nt=Threads.nthreads())
    ypred[1][1]
end
