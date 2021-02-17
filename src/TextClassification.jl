# This file is a part of TextClassification.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

module TextClassification
using SimilaritySearch
using KCenters
using StructTypes
using KNearestCenters
using TextSearch

using CategoricalArrays
using SearchModels
import SearchModels: random_configuration, combine_configurations, mutate_configuration, config_type

include("centerselection.jl")
include("textconfigspace.jl")
include("textmodelspace.jl")
include("knnconfig.jl")
include("knnclassifier.jl")
include("liblinearconfig.jl")
include("microtcconfig.jl")
include("microtc.jl")
end # module
