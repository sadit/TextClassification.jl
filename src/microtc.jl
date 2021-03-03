# This file is a part of TextClassification.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using MLDataUtils, LinearAlgebra
import KNearestCenters: combine_configurations, random_configuration
import TextSearch: vectorize
import StatsBase: predict
import Base: hash, isequal
export filtered_power_set, predict, vectorize,
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
    MicroTC(config::MicroTC_Config, train_corpus::AbstractVector, train_y::CategoricalArray; verbose=true)
    MicroTC(config::MicroTC_Config, textmodel::TextModel, train_X::AbstractVector{S}, train_y::CategoricalArray; verbose=true) where {S<:SVEC}

Creates a MicroTC model on the given dataset and configuration
"""
function MicroTC(config::MicroTC_Config, train_corpus::AbstractVector, train_y::CategoricalArray; verbose=true)
    train_corpus_bow = [compute_bow(config.textconfig, text) for text in train_corpus]
    textmodel = create(config.textmodel, train_corpus_bow, train_y)
    MicroTC(config, textmodel, [vectorize(textmodel, bow) for bow in train_corpus_bow], train_y)
end

function MicroTC(config::MicroTC_Config, textmodel::TextModel, train_X::AbstractVector{S}, train_y::CategoricalArray; verbose=true) where {S<:SVEC}
    cls = create(config.cls, train_X, train_y, textmodel.m)
    MicroTC(config, cls, textmodel)
end

"""
    vectorize(tc::MicroTC, text)
    vectorize(tc::MicroTC, bow::BOW)

Creates a weighted vector using the model. The input `text` can be a string or an array of strings;
it also can be an already computed bag of words.

"""
vectorize(tc::MicroTC, text)::SVEC = vectorize(tc.textmodel, compute_bow(tc.config.textconfig, text))
vectorize(tc::MicroTC, bow::BOW)::SVEC = vectorize(tc.textmodel, bow)

"""
    predict(tc::MicroTC, text)
    predict(tc::MicroTC, vec::SVEC)

Predicts the label of the given input
"""
predict(tc::MicroTC, text) = predict(tc.cls, vectorize(tc, text))
predict(tc::MicroTC, vec::SVEC) = predict(tc.cls, vec)
