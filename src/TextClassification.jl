# This file is part of TextClassification.jl

module TextClassification
using SimilaritySearch, TextSearch, SearchModels
import SearchModels: combine, combine_select, mutate, config_type
using Parameters, CategoricalArrays, InvertedFiles

include("textconfigspace.jl")
include("textmodelspace.jl")
include("liblinearconfig.jl")
include("microtcconfig.jl")
include("microtc.jl")
include("utils.jl")
end # module
