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
nv(fg::FeaturedGraph{T}) where {T<:AbstractVector} = length(graph(fg))
nv(g::AbstractMatrix) = size(g, 1)

"""
    ne(::AbstractFeaturedGraph)

Get edge number of graph.
"""
ne(::NullGraph) = 0
ne(fg::FeaturedGraph) = ne(graph(fg))
ne(fg::FeaturedGraph{T}) where {T<:AbstractMatrix} = sum(graph(fg) .!= zero(eltype(T)))
ne(fg::FeaturedGraph{T}) where {T<:AbstractVector} = sum(map(length, graph(fg)))÷2

"""
    fetch_graph(g1, g2)

Fetch graph from `g1` or `g2`. If there is only one graph available, fetch that one.
Otherwise, fetch the first one.
"""
fetch_graph(::NullGraph, fg::FeaturedGraph) = graph(fg)
fetch_graph(fg::FeaturedGraph, ::NullGraph) = graph(fg)
fetch_graph(fg1::FeaturedGraph, fg2::FeaturedGraph) = has_graph(fg1) ? graph(fg1) : graph(fg2)
