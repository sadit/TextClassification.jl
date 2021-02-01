# This file is a part of TextClassification.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using MLDataUtils, LinearAlgebra
import KNearestCenters: transform, search_params, combine_configurations, random_configuration
import TextSearch: vectorize
import StatsBase: predict
import Base: hash, isequal
export filtered_power_set, predict, vectorize, transform,
        MicroTC, AngleDistance, CosineDistance, NormalizedAngleDistance, NormalizedCosineDistance
import Base: hash, isequal

struct MicroTC{C_<:MicroTC_Config, AKNC_<:AKNC, TextModel_<:TextModel}
    config::C_
    aknc::AKNC_
    textmodel::TextModel_
end

StructTypes.StructType(::Type{<:MicroTC}) = StructTypes.Struct()

Base.copy(c::MicroTC;
        config=c.config,
        aknc::AKNC=c.nc,
        textmodel=c.textmodel
) = MicroTC(config, aknc, textmodel)

function broadcastable(tc::MicroTC)
    (tc,)
end

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
    verbose && println("MicroTC> creating bag of words for corpus")
    train_corpus_bow = [compute_bow(config.textconfig, text) for text in train_corpus]
    verbose && println("MicroTC> creating textmodel $(config.textmodel)")
    textmodel = create_textmodel(config, train_corpus_bow, train_y)
    MicroTC(config, textmodel, [vectorize(textmodel, bow) for bow in train_corpus_bow], train_y)
end

function MicroTC(config::MicroTC_Config, textmodel::TextModel, train_X::AbstractVector{S}, train_y::CategoricalArray; verbose=true) where {S<:SVEC}
    verbose && println("MicroTC> creating AKNC classifier")
    @info config.akncconfig, typeof(train_X)
    aknc = AKNC(config.akncconfig, train_X, train_y; verbose=verbose)
    verbose && println("MicroTC> done")
    MicroTC(config, aknc, textmodel)
end

vectorize(tc::MicroTC, text)::SVEC = vectorize(tc.textmodel, compute_bow(tc.config.textconfig, text))
vectorize(tc::MicroTC, bow::BOW)::SVEC = vectorize(tc.textmodel, bow)

predict(tc::MicroTC, text::S) where {S<:Union{AbstractString,BOW}} = predict(tc.aknc, vectorize(tc, text))
predict(tc::MicroTC, vec::SVEC) = predict(tc.aknc, vec)

function transform(tc::MicroTC, vec::SVEC)
    X = transform(tc.nc.centers, tc.nc.dmax, tc.kernel, vec)
    M = labelmap(tc.nc.class_map)
    L = zeros(Float64, tc.nc.nclasses)

    for i in 1:tc.nc.nclasses
        lst = get(M, i, nothing)
        if lst !== nothing
           L[i] = maximum(X[lst])
        end
    end

    L
end

transform(tc::MicroTC, text::AbstractString) = transform(tc, vectorize(tc, text))