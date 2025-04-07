# This file is part of TextClassification.jl

using MLUtils, LinearAlgebra
import TextSearch: vectorize, vectorize_corpus, BOW
import StatsBase: predict
import Base: hash, isequal
export filtered_power_set, predict, predict_corpus, vectorize, vectorize_corpus, 
        MicroTC, AngleDistance, CosineDistance, NormalizedAngleDistance, NormalizedCosineDistance
import Base: hash, isequal
using SparseArrays

struct MicroTC{C_<:MicroTC_Config, CLS_<:Any, TextModel_<:TextModel, LabelType<:Any}
    config::C_
    cls::CLS_
    textmodel::TextModel_
    textconfig::TextConfig
    levels::Vector{LabelType}
end

function Base.show(io::IO, model::MicroTC) 
    print(io, "{MicroTC")
    show(io, model.config)
    print(io, " ")
    show(io, typeof(model.cls))
    print(io, " ")
    show(io, model.textmodel)
    print(io, " ")
    show(io, model.textconfig)
    print(io, " ")
    show(io, model.levels)
    print(io, "}")
end

Base.copy(c::MicroTC; config=c.config, cls=c.cls, textmodel=c.textmodel) = MicroTC(config, cls, textmodel)
Base.broadcastable(tc::MicroTC) = (tc,)

"""
    MicroTC(
        config::MicroTC_Config,
        train_corpus::AbstractVector,
        train_y::CategoricalArray;
        textconfig=config.textconfig,
        verbose=true)
    MicroTC(config::MicroTC_Config, textmodel::TextModel, train_X::AbstractVector{S}, train_y::CategoricalArray; verbose=true) where {S<:SVEC}

Creates a MicroTC model on the given dataset and configuration
"""
function MicroTC(
            config::MicroTC_Config,
            train_corpus::AbstractVector,
            train_y;
            textconfig=config.textconfig,
            verbose=true,
            minbatch=0
        )
    
    ## vectorization and tokenization make heavy use of multithreading, we lock other threads here to allow the running thread can take others inside the block without resource competition
    textmodel = create(config.textmodel, textconfig, train_corpus, train_y; minbatch)    
    X = vectorize_corpus(textmodel, train_corpus; minbatch)
    mask = BitVector(undef, length(train_corpus))
    for (i, x) in enumerate(X)
        mask[i] = !(length(x) == 1 && haskey(x, 0))
    end

    m = sum(mask)
    if m != length(mask)
        @info "WARNING using $(m) of $(length(mask)) examples after vectorization using $(config)"
        MicroTC(config, textmodel, X[mask], train_y[mask]; textconfig)
    else
        MicroTC(config, textmodel, X, train_y; textconfig)
    end
end

function MicroTC(
        config::MicroTC_Config,
        textmodel::TextModel,
        train_X::AbstractVector{S},
        train_y;
        textconfig=config.textconfig,
        verbose=true) where {S<:SVEC}
    cls = create(config.cls, train_X, train_y, vocsize(textmodel))
    MicroTC(config, cls, textmodel, textconfig, copy(levels(train_y)))
end

"""
    vectorize(tc::MicroTC, text; bow=BOW(), textconfig=tc.textconfig, normalize=true)
    vectorize(tc::MicroTC, bow::BOW; normalize=true)

Creates a weighted vector using the model. The input `text` can be a string or an array of strings;
it also can be an already computed bag of words.

"""
function vectorize(tc::MicroTC, text; textconfig=tc.textconfig, normalize=true)::SVEC
    vectorize(tc.textmodel, textconfig, text; normalize)
end

function vectorize(tc::MicroTC, bow::BOW; normalize=true)::SVEC
    vectorize(tc.textmodel, bow; normalize=normalize)
end

function vectorize_corpus(tc::MicroTC, corpus;
        textconfig=tc.textconfig,
        minbatch=0,
        normalize=true
    )
    vectorize_corpus(tc.textmodel, textconfig, corpus; normalize, minbatch)
end

"""
    predict(tc::MicroTC, text)
    predict(tc::MicroTC, vec::SVEC)

Predicts the label of the given input
"""
predict(tc::MicroTC, text) = predict(tc.cls, vectorize(tc, text))
predict(tc::MicroTC, vec::SVEC) = tc.levels[predict(tc.cls, vec)]

function predict_corpus(tc::MicroTC, corpus;
    textconfig=tc.textconfig,
    minbatch=0,
    normalize=true)

    V = vectorize_corpus(tc.textmodel, corpus; normalize, minbatch)
    #don't know if liblinear prediction is multithreading
    n = length(V)
    # minbatch = getminbatch(minbatch, n)
    P = Vector{eltype(tc.levels)}(undef, n)
    #Threads.@threads
    for i in 1:n
        P[i] = predict(tc, V[i])
    end
    #[predict(tc, v) for v in V]
    P
end
