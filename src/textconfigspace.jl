# This file is a part of TextClassification.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using IterTools

function filtered_power_set(set, lowersize=0, uppersize=5)
    lst = collect(subsets(set))
    filter(x -> lowersize <= length(x) <= uppersize, lst)
end

const QLIST = filtered_power_set([2, 3, 4, 5, 6], 1, 3)
const NLIST = filtered_power_set([1, 2, 3], 0, 2)
const SLIST = [[Skipgram(2, 1), Skipgram(2, 2)], [Skipgram(2, 1)], [Skipgram(2, 2)], Skipgram[]]

@with_kw struct TextConfigSpace <: AbstractSolutionSpace
    del_diac = [true]
    del_dup = [false]
    del_punc = [false]

    group_num = [true]
    group_url = [true]
    group_usr = [true]
    group_emo = [false]

    lc = [true]
    qlist = QLIST
    nlist = NLIST
    slist = SLIST
end

StructTypes.StructType(::Type{TextConfigSpace}) = StructTypes.Struct()
Base.eltype(::TextConfigSpace) = TextConfig

function random_configuration(space::TextConfigSpace)
    TextConfig(
        rand(space.del_diac),
        rand(space.del_dup),
        rand(space.del_punc),

        rand(space.group_num),
        rand(space.group_url),
        rand(space.group_usr),
        rand(space.group_emo),
        
        rand(space.lc),
        isempty(space.qlist) ? Int[] : rand(space.qlist),
        isempty(space.nlist) ? Int[] : rand(space.nlist),
        isempty(space.slist) ? Skipgram[] : rand(space.slist),
    )
end

function combine_configurations(a::TextConfig, b::TextConfig)
    L = [a, b]

    TextConfig(
        rand(L).del_diac,
        rand(L).del_dup,
        rand(L).del_punc,
        rand(L).group_num,
        rand(L).group_url,
        rand(L).group_usr,
        rand(L).group_emo,
        rand(L).lc,
        rand(L).qlist,
        rand(L).nlist,
        rand(L).slist
    )
end