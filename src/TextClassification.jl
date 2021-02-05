# This file is a part of TextClassification.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

module TextClassification
using SimilaritySearch
using KCenters
using StructTypes
using KNearestCenters
using TextSearch

using CategoricalArrays
import KNearestCenters: random_configuration, combine_configurations

include("centerselection.jl")
include("textconfigspace.jl")
include("textmodelspace.jl")
include("liblinearconfig.jl")
include("microtcconfig.jl")
include("microtc.jl")
end # module
