# This file is a part of TextClassification.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

import KCenters: center
export TextCentroidSelection

struct TextCentroidSelection <: AbstractCenterSelection end
StructTypes.StructType(::Type{<:TextCentroidSelection}) = StructTypes.Struct()

function center(::TextCentroidSelection, lst::AbstractVector{<:DVEC})
    u = zero(eltype(lst))

    for v in lst
        TextSearch.add!(u, v)
    end
    
    TextSearch.normalize!(u)
end
