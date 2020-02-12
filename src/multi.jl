# This file is a part of TextClassification.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using Random, TextSearch
import KCenters: bagging, glue
import SimilaritySearch: optimize!
export glue, bagging, optimize!

"""
    glue(arr::AbstractVector{μTC})

Joins a list of text classifiers into a single one classifier.
"""
function glue(arr::AbstractVector{μTC})
    centers = []
    class_map = Int[]
    dmax = Float64[]
    for a in arr
        for c in a.nc.centers
            push!(centers, bow(a.model, c))
        end

        append!(class_map, a.nc.class_map)
        append!(dmax, a.nc.dmax)
    end

    item = first(arr)
    model = glue([a.model for a in arr])
    centers_ = [dvec(model, c) for c in centers]
    nc = KNC(centers_, dmax, class_map, item.nc.nclasses)
    config = item.config
    kernel = item.kernel
    μTC(nc, model, config, kernel)
end

"""
    bagging(config::μTC_Configuration, X::AbstractVector, y::AbstractVector{I}; b=13, ratio=0.5) where {I<:Integer}

Creates `b` text classifiers, each trained with a random `ratio` of the dataset;
the resulting classifiers are joint into a single classifier.
"""
function bagging(config::μTC_Configuration, X::AbstractVector, y::AbstractVector{I}; b=13, ratio=0.5) where {I<:Integer}
    indexes = collect(1:length(X))
    m = ceil(Int, ratio * length(X))

    L = Vector{μTC}(undef, b)
    for i in 1:b
        shuffle!(indexes)
        sample = @view indexes[1:m]
        L[i] = fit(μTC, config, X[sample], y[sample]; verbose=true)
    end

    glue(L)
end


"""
    optimize!(model::μTC, X, y; k=[1, 3, 5, 7], kernel=[direct_kernel, relu_kernel, laplacian_kernel, gaussian_kernel])
Selects `k` and `kernel` to adjust better to the given score and the dataset ``(X, y)``.
"""
function optimize!(model::μTC, X, y, score::Function=recall_score; k=[1, 3, 5, 7], kernel=[direct_kernel, relu_kernel, laplacian_kernel, gaussian_kernel], verbose=true)
    L = []
    for k_ in k, kernel_ in kernel
        kernel_fun = kernel_(model.config.dist)
        model.config.k = k_
        model.kernel = kernel_fun
        ypred = predict(model, X)
        s = score(y, ypred)
        push!(L, (score=s, k=k_, kernel=kernel_, kernel_fun=kernel_fun))
        verbose && println(stderr, L[end])
    end

    sort!(L, by=x->x.score, rev=true)
    c = first(L)
    model.config.k = c.k
    model.config.kernel = c.kernel
    model.kernel = c.kernel_fun
    L
end