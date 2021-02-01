# This file is a part of TextClassification.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

import KCenters: center

function center(::CentroidSelection, lst::AbstractVector{DVEC{Ti,Tv}}) where {Ti,Tv<:Real}
    u = zero(DVEC{Ti,Tv})

    for v in lst
        TextSearch.add!(u, v)
    end
    
    TextSearch.normalize!(u)
end


function center(sel::KnnCentroidSelection, lst::AbstractVector{DVEC{Ti,Tv}}) where {Ti,Tv<:Real}
    c = center(sel.sel, lst)
    seq = ExhaustiveSearch(sel.dist, lst)
    k = sel.k == 0 ? ceil(Int32, log2(length(lst))) : sel.k
    TextSearch.normalize!(TextSearch.sum(lst[[p.id for p in search(seq, c, k)]]))
end
