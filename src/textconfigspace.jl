# This file is part of TextClassification.jl

using Random

function filtered_power_set(set, lowersize=0, uppersize=5)
    lst = collect(subsets(set))
    filter(x -> lowersize <= length(x) <= uppersize, lst)
end

@with_kw struct TextConfigSpace <: AbstractSolutionSpace
    del_diac = [true]
    del_dup = [false]
    del_punc = [false]

    group_num = [true]
    group_url = [true]
    group_usr = [true]
    group_emo = [false]

    lc = [true]
    qlist = filtered_power_set([2, 3, 4, 5], 1, 3)
    nlist = filtered_power_set([1, 2, 3], 0, 2)
    slist = [[Skipgram(2, 1), Skipgram(2, 2)], [Skipgram(2, 1)], [Skipgram(2, 2)], Skipgram[]]

    qlist_space = 1:9
    nlist_space = 1:5
    slist_space = [Skipgram(2, 1), Skipgram(2, 2), Skipgram(3, 1), Skipgram(3, 2)]
end

Base.eltype(::TextConfigSpace) = TextConfig

function Base.rand(space::TextConfigSpace)
    qlist = isempty(space.qlist) ? Int[] : rand(space.qlist)
    nlist = isempty(space.nlist) ? Int[] : rand(space.nlist)
    slist = isempty(space.slist) ? Skipgram[] : rand(space.slist)

    if length(qlist) + length(nlist) + length(slist) == 0
        nlist = [1]
    end

    TextConfig(;
        del_diac = rand(space.del_diac),
        del_dup = rand(space.del_dup),
        del_punc = rand(space.del_punc),

        group_num = rand(space.group_num),
        group_url = rand(space.group_url),
        group_usr = rand(space.group_usr),
        group_emo = rand(space.group_emo),
        
        lc = rand(space.lc),
        qlist,
        nlist,
        slist
    )
end

function combine(a::TextConfig, b::TextConfig)
    L = [a, b]

    qlist = rand() < 0.5 ? union(a.qlist, b.qlist) : intersect(a.qlist, b.qlist)
    nlist = rand() < 0.5 ? union(a.nlist, b.nlist) : intersect(a.nlist, b.nlist)
    slist = rand() < 0.5 ? union(a.slist, b.slist) : intersect(a.slist, b.slist)

    if length(qlist) + length(nlist) + length(slist) == 0
        nlist = [1]
    end
    
    TextConfig(;
        rand(L).del_diac,
        rand(L).del_dup,
        rand(L).del_punc,
        rand(L).group_num,
        rand(L).group_url,
        rand(L).group_usr,
        rand(L).group_emo,
        rand(L).lc,
        qlist=sort!(qlist), 
        nlist=sort!(nlist),
        slist=sort!(slist)
    )
end

function mutate_token_list(lst, L; p1=0.5, p2=0.5)
    if length(L) == 0 || rand() < p1 
        return lst
    end

    if rand() < p2
        if length(lst) > 0
            lst = copy(lst)
            shuffle!(lst)
            pop!(lst)
            sort!(lst)
        end
    else
        lst = union(lst, [rand(L)])
    end

    lst
end

function mutate(space::TextConfigSpace, c::TextConfig, iter)
    qlist = mutate_token_list(c.qlist, space.qlist_space)
    nlist = mutate_token_list(c.nlist, space.nlist_space)
    slist = mutate_token_list(c.slist, space.slist_space)
    if length(qlist) + length(nlist) + length(slist) == 0
        nlist = [1]
    end
    
    TextConfig(;
        del_diac = SearchModels.change(c.del_diac, space.del_diac),
        del_dup = SearchModels.change(c.del_dup, space.del_dup),
        del_punc = SearchModels.change(c.del_punc, space.del_punc),
        group_num = SearchModels.change(c.group_num, space.group_num),
        group_url = SearchModels.change(c.group_url, space.group_url),
        group_usr = SearchModels.change(c.group_usr, space.group_usr),
        group_emo = SearchModels.change(c.group_emo, space.group_emo),
        lc = SearchModels.change(c.lc, space.lc),
        qlist,
        nlist,
        slist
    )
end