# This file is a part of TextClassification.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using MLDataUtils, LinearAlgebra
import TextSearch: vectorize, vectorize_corpus
import StatsBase: predict
import Base: hash, isequal
export filtered_power_set, predict, predict_corpus, vectorize, vectorize_corpus, 
        MicroTC, AngleDistance, CosineDistance, NormalizedAngleDistance, NormalizedCosineDistance
import Base: hash, isequal
using SparseArrays

struct MicroTC{C_<:MicroTC_Config, CLS_<:Any, TextModel_<:TextModel}
    config::C_
    cls::CLS_
    textmodel::TextModel_
end

StructTypes.StructType(::Type{<:MicroTC}) = StructTypes.Struct()
function Base.show(io::IO, model::MicroTC) 
    print(io, "{MicroTC ")
    show(io, model.config)
    show(io, model.cls)
    show(io, model.textmodel)
    print(io, "}")
end

Base.copy(c::MicroTC; config=c.config, cls=c.cls, textmodel=c.textmodel) = MicroTC(config, cls, textmodel)
Base.broadcastable(tc::MicroTC) = (tc,)

"""
    create(config, train_X, train_y) # config describes a text model
    create(config, train_X, train_y, dim) # config describes a classifier

Creates a new object from a configuration and a train / test datasets.

- train_X is an array of BOW objects for text models
- train_X is an array of SVEC objects for classifiers
"""
create(config::KncConfig, train_X, train_y, dim) = Knc(config, train_X, train_y)
create(config::KncProtoConfig, train_X, train_y, dim) = KncProto(config, train_X, train_y)

"""
    MicroTC(
        config::MicroTC_Config,
        train_corpus::AbstractVector,
        train_y::CategoricalArray;
        tok=Tokenizer(config.textconfig),
        verbose=true)
    MicroTC(config::MicroTC_Config, textmodel::TextModel, train_X::AbstractVector{S}, train_y::CategoricalArray; verbose=true) where {S<:SVEC}

Creates a MicroTC model on the given dataset and configuration
"""
function MicroTC(
        config::MicroTC_Config,
        train_corpus::AbstractVector,
        train_y::CategoricalArray;
        tok=Tokenizer(config.textconfig, invmap=nothing),
        verbose=true)
    @show :train_corpus_bow
    @time train_corpus_bow = compute_bow_corpus(tok, train_corpus)
    # @show :textmodel
    textmodel = create(config.textmodel, train_corpus_bow, train_y)
    MicroTC(config, textmodel, [vectorize(textmodel, bow) for bow in train_corpus_bow], train_y, tok=tok)
end

function MicroTC(
        config::MicroTC_Config,
        textmodel::TextModel,
        train_X::AbstractVector{S},
        train_y::CategoricalArray;
        tok=Tokenizer(config.textconfig, invmap=nothing),
        verbose=true) where {S<:SVEC}
    cls = create(config.cls, train_X, train_y, textmodel.m)
    MicroTC(config, cls, textmodel)
end

"""
    vectorize(tc::MicroTC, text;
        bow=BOW(),
        tok=Tokenizer(tc.config.textconfig, invmap=nothing),
        normalize=true)
    vectorize(tc::MicroTC, bow::BOW)

Creates a weighted vector using the model. The input `text` can be a string or an array of strings;
it also can be an already computed bag of words.

"""
function vectorize(
        tc::MicroTC,
        text;
        bow=BOW(),
        tok=Tokenizer(tc.config.textconfig, invmap=nothing),
        normalize=true
    )::SVEC
    vectorize(tc.textmodel, compute_bow(tok, text, bow); normalize)
end

function vectorize(
        tc::MicroTC,
        bow::BOW;
        normalize=true
    )::SVEC
    vectorize(tc.textmodel, bow; normalize=normalize)
end

function vectorize_corpus(
        tc::MicroTC,
        corpus;
        bow=BOW(),
        tok=Tokenizer(tc.config.textconfig, isconstruction=false, invmap=nothing),
        normalize=true
    )
    V = Vector{SVEC}(undef, length(corpus))

    for i in eachindex(corpus)
        empty!(bow)
        V[i] = vectorize(tc, corpus[i]; bow, tok, normalize)
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
    bow=BOW(),
    tok=Tokenizer(tc.config.textconfig, isconstruction=false, invmap=nothing),
    normalize=true)
    V = Vector{UInt32}(undef, length(corpus))

    for i in eachindex(corpus)
        empty!(bow)
        empty!(tok)
        v = vectorize(tc, corpus[i]; bow, tok, normalize)
        V[i] = predict(tc, v)
    end

    V
end
