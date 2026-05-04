# This file is part of TextClassification.jl

using MLUtils, LinearAlgebra
import TextSearch: vectorize, vectorize_corpus, BOW
import StatsBase: predict
import Base: hash, isequal
export filtered_power_set, predict, predict_corpus, vectorize, vectorize_corpus, MicroTC
import Base: hash, isequal
using SparseArrays

struct MicroTC{C_<:MicroTC_Config,CLS_<:Any,TextModel_<:TextModel}
    config::C_
    cls::CLS_
    textmodel::TextModel_
end

function Base.show(io::IO, model::MicroTC)
    print(io, "{MicroTC")
    show(io, model.config)
    print(io, " ")
    show(io, typeof(model.cls))
    print(io, " ")
    show(io, model.textmodel)
    print(io, "}")
end

Base.copy(c::MicroTC; config=c.config, cls=c.cls, textmodel=c.textmodel) = MicroTC(config, cls, textmodel)
Base.broadcastable(tc::MicroTC) = (tc,)

"""
    MicroTC(
        config::MicroTC_Config,
        train_corpus::AbstractVector,
        train_y;
        textconfig=config.textconfig,
        verbose=true)
    MicroTC(config::MicroTC_Config, textmodel::TextModel, train_X::AbstractVector{S}, train_y; verbose=true) where {S<:SVEC}

Creates a MicroTC model on the given dataset and configuration
"""
function MicroTC(
    config::MicroTC_Config,
    train_corpus,
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
        @warn "using $(m) of $(length(mask)) examples after vectorization using $(config)"
        MicroTC(config, textmodel, X[mask], train_y[mask])
    else
        MicroTC(config, textmodel, X, train_y)
    end
end

function MicroTC(
    config::MicroTC_Config,
    textmodel::TextModel,
    train_X::AbstractVector,
    train_y::AbstractVector;
    verbose=true)
    cls = create(config.cls, train_X, train_y, vocsize(textmodel))
    MicroTC(config, cls, textmodel)
end

"""
    vectorize(tc::MicroTC, text; bow=BOW(), textconfig=tc.textconfig, normalize=true)
    vectorize(tc::MicroTC, bow::BOW; normalize=true)

Creates a weighted vector using the model. The input `text` can be a string or an array of strings;
it also can be an already computed bag of words.

"""
function vectorize(tc::MicroTC, text; normalize=true)::SVEC
    vectorize(tc.textmodel, text; normalize)
end

function vectorize(tc::MicroTC, bow::BOW; normalize=true)::SVEC
    vectorize(tc.textmodel, bow; normalize=normalize)
end

function vectorize_corpus(tc::MicroTC, corpus;
    minbatch=0,
    normalize=true
)
    V = vectorize_corpus(tc.textmodel, corpus; normalize, minbatch)
    for v in V
        if length(v) == 0 # empty vector
            v[rand(1:vocsize(tc.textmodel))] = 1f0
        end
    end

    V
end

"""
    predict(tc::MicroTC, text)
    predict(tc::MicroTC, vec::SVEC)

Predicts the label of the given input
"""
predict(tc::MicroTC, text) = predict(tc.cls, vectorize(tc, text))
predict(tc::MicroTC, vec::SVEC) = predict(tc.cls, vec)

function predict_corpus(tc::MicroTC, corpus;
    minbatch=0,
    normalize=true)

    V = vectorize_corpus(tc.textmodel, corpus; normalize, minbatch)

    #I don't know if liblinear prediction is already multithreading per call
    [predict(tc, v) for v in V]
end
