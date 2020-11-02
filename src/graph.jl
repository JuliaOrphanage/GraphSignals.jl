"""
    adjacency_list(::AbstractFeaturedGraph)

Get adjacency list of graph.
"""
adjacency_list(::NullGraph) = [zeros(0)]
adjacency_list(fg::FeaturedGraph) = adjacency_list(graph(fg))

"""
    adjacency_list(adj)
Transform a adjacency matrix into a adjacency list.
"""
function adjacency_list(adj::AbstractMatrix{T}) where {T}
    n = size(adj,1)
    @assert n == size(adj,2) "adjacency matrix is not a square matrix."
    A = (adj .!= zero(T))
    if !issymmetric(adj)
        A = A .| A'
    end
    indecies = collect(1:n)
    ne = Vector{Int}[indecies[view(A, :, i)] for i = 1:n]
    return ne
end

adjacency_list(adj::AbstractVector{<:AbstractVector{<:Integer}}) = adj

Zygote.@nograd adjacency_list

"""
    nv(::AbstractFeaturedGraph)

Get node number of graph.
"""
nv(::NullGraph) = 0
nv(fg::FeaturedGraph) = nv(graph(fg))
nv(fg::FeaturedGraph{T}) where {T<:AbstractMatrix} = size(graph(fg), 1)
nv(g::AbstractMatrix) = size(g, 1)
nv(g::AbstractVector{T}) where {T<:AbstractVector} = size(g, 1)

Zygote.@nograd nv

"""
    ne(::AbstractFeaturedGraph)

Get edge number of graph.
"""
ne(::NullGraph) = 0
ne(fg::FeaturedGraph) = ne(graph(fg))
ne(fg::FeaturedGraph{T}) where {T<:AbstractMatrix} = ne(graph(fg))
ne(fg::FeaturedGraph{T}) where {T<:AbstractVector} = ne(graph(fg), fg.directed)
function ne(g::AbstractMatrix; self_loop::Bool=false)
    g = Matrix(g .!= 0)

    if issymmetric(g)
        if self_loop
            return div(sum(g .+ diagm(diag(g))), 2)
        else
            return div(sum(g .- diagm(diag(g))), 2)
        end
    else
        if self_loop
            return sum(g)
        else
            return sum(g .- diagm(diag(g)))
        end
    end
end

function ne(g::AbstractVector{T}, directed::Bool) where {T<:AbstractVector}
    for i in 1:length(g)
        filter!(x -> x != i, g[i])
    end
    s = map(x -> count(x .== x), g)
    return directed ? sum(s) : div(sum(s), 2)
end

Zygote.@nograd ne

is_directed(fg::FeaturedGraph) = fg.directed

function is_directed(g::AbstractVector{T}) where {T<:AbstractVector}
    edges = Set{Tuple{Int64,Int64}}()
    for (i, js) in enumerate(g)
        for j in Set(js)
            if i != j
                e = (i,j)
                if e in edges
                    pop!(edges, e)
                else
                    push!(edges, (j,i))
                end
            end
        end
    end
    !isempty(edges)
end

Zygote.@nograd is_directed

"""
    fetch_graph(g1, g2)

Fetch graph from `g1` or `g2`. If there is only one graph available, fetch that one.
Otherwise, fetch the first one.
"""
fetch_graph(::NullGraph, fg::FeaturedGraph) = graph(fg)
fetch_graph(fg::FeaturedGraph, ::NullGraph) = graph(fg)
fetch_graph(fg1::FeaturedGraph, fg2::FeaturedGraph) = has_graph(fg1) ? graph(fg1) : graph(fg2)
