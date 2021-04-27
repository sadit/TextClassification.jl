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

struct MicroTC{C_<:MicroTC_Config, CLS_<:Any, TextModel_<:TextModel, LabelType<:Any}
    config::C_
    cls::CLS_
    textmodel::TextModel_
    tok::Tokenizer
    levels::Vector{LabelType}
end

StructTypes.StructType(::Type{<:MicroTC}) = StructTypes.Struct()
function Base.show(io::IO, model::MicroTC) 
    print(io, "{MicroTC")
    show(io, ' ', model.config)
    show(io, ' ', model.cls)
    show(io, ' ', model.textmodel)
    show(io, ' ', model.tok)
    show(io, ' ', model.levels)
    print(io, "}")
end

Base.copy(c::MicroTC; config=c.config, cls=c.cls, textmodel=c.textmodel) = MicroTC(config, cls, textmodel)
Base.broadcastable(tc::MicroTC) = (tc,)

"""
    create(config, train_X, train_y) # config describes a text model
    create(config, train_X, train_y, tok, dim) # config describes a classifier

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
    train_corpus_bow = compute_bow_corpus(tok, train_corpus)
    textmodel = create(config.textmodel, train_corpus_bow, train_y)
    X = SVEC[]
    mask = Bool[]
    for bow in train_corpus_bow
        x = vectorize(textmodel, bow)
        push!(mask, false)
        length(x) == 1 && haskey(x, 0) && continue
        push!(X, x)
        mask[end] = true
    end

    m = sum(mask)
    if m != length(mask)
        @info "WARNING considering $(m) of $(length(mask)) examples after vectorization using $(config)"
    end
    MicroTC(config, textmodel, X, train_y[mask], tok=Tokenizer(tok))  # tok is a copy with isconstruction=false
end

function MicroTC(
        config::MicroTC_Config,
        textmodel::TextModel,
        train_X::AbstractVector{S},
        train_y::CategoricalArray;
        tok=Tokenizer(config.textconfig, invmap=nothing),
        verbose=true) where {S<:SVEC}
    cls = create(config.cls, train_X, train_y, textmodel.m)
    MicroTC(config, cls, textmodel, tok, copy(levels(train_y)))
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
        tok=tc.tok,
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
        tok=tok=tc.tok,
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
predict(tc::MicroTC, vec::SVEC) = tc.levels[predict(tc.cls, vec)]

function predict_corpus(tc::MicroTC, corpus;
    bow=BOW(),
    tok=tc.tok,
    normalize=true)
    V = Vector{eltype(tc.levels)}(undef, length(corpus))

    for i in eachindex(corpus)
        empty!(bow)
        empty!(tok)
        v = vectorize(tc, corpus[i]; bow, tok, normalize)
        V[i] = predict(tc, v)
    end

    V
end
