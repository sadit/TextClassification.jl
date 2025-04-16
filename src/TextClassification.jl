# This file is part of TextClassification.jl

module TextClassification
import StatsAPI: predict, fit
using SimilaritySearch, TextSearch, SearchModels
import SearchModels: combine, combine_select, mutate, config_type, InvalidSetupError
using Parameters, InvertedFiles

include("scores.jl")
include("textconfigspace.jl")
include("textmodelspace.jl")
include("libsvmconfig.jl")
include("microtcconfig.jl")
include("microtc.jl")
include("utils.jl")
end # module
