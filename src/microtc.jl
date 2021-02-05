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

Base.copy(c::MicroTC;
        config=c.config,
        cls=c.cls,
        textmodel=c.textmodel
) = MicroTC(config, cls, textmodel)

Base.broadcastable(tc::MicroTC) = (tc,)

#=
function MicroTC(textconfig::TextConfig, textmodel::TextModel, config::MicroTC_Config, train_corpus::AbstractVector{BOW}, train_y::CategoricalArray; verbose=true) 
    textmodel = create_textmodel(config, train_corpus, train_y)
    MicroTC(config, textmodel, train_corpus, train_y; verbose=verbose)
end
=#
function create_textmodel(config::MicroTC_Config, train_X::AbstractVector{BOW}, train_y::CategoricalArray)
    model = if config.textmodel == :EntModel
        EntModel(config.weighting, train_X, train_y, smooth=config.smooth, minocc=config.minocc, weights=config.classweights)
    else
        VectorModel(config.weighting, sum(train_X), minocc=config.minocc)
    end

    if config.p < 1.0
        model = prune_select_top(model, config.p)
    end

    model
end

function MicroTC(config::MicroTC_Config, train_corpus::AbstractVector{S}, train_y::CategoricalArray; verbose=true) where {S<:AbstractString}
    train_corpus_bow = [compute_bow(config.textconfig, text) for text in train_corpus]
    textmodel = create_textmodel(config, train_corpus_bow, train_y)
    MicroTC(config, textmodel, [vectorize(textmodel, bow) for bow in train_corpus_bow], train_y)
end

using LIBLINEAR
StructTypes.StructType(::Type{<:LIBLINEAR.LinearModel}) = StructTypes.Struct()

function MicroTC(config::MicroTC_Config, textmodel::TextModel, train_X::AbstractVector{S}, train_y::CategoricalArray; verbose=true) where {S<:SVEC}
    cls = if config.cls isa KncConfig
        Knc(config.cls, train_X, train_y)
    else
        linear_train(train_y.refs, sparse(train_X); C=config.cls.C, eps=config.cls.eps)
    end
    MicroTC(config, cls, textmodel)
end

vectorize(tc::MicroTC, text)::SVEC = vectorize(tc.textmodel, compute_bow(tc.config.textconfig, text))
vectorize(tc::MicroTC, bow::BOW)::SVEC = vectorize(tc.textmodel, bow)

predict(tc::MicroTC, text::S) where {S<:Union{AbstractString,BOW}} = predict(tc.cls, vectorize(tc, text))
predict(tc::MicroTC, vec::SVEC) = predict(tc.cls, vec)

function predict(cls::LIBLINEAR.LinearModel, vec::SVEC)
    ypred = linear_predict(cls, sparse([vec], cls.nr_feature))
    ypred[1][1]
end
