# This file is a part of TextClassification.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

import KCenters: center
export TextCentroidSelection

struct TextCentroidSelection <: AbstractCenterSelection end

function center(::TextCentroidSelection, lst::AbstractDatabase)
    u = zero(eltype(lst))

    for v in lst
        InvertedFiles.add!(u, v)
    end
    
    InvertedFiles.normalize!(u)
end
