"""
    adjacency_list(::AbstractFeaturedGraph)

Get adjacency list of graph.
"""
adjacency_list(::NullGraph) = [zeros(0)]
adjacency_list(fg::FeaturedGraph) = adjacency_list(fg.graph[])

"""
    nv(::AbstractFeaturedGraph)

Get node number of graph.
"""
nv(::NullGraph) = 0
nv(fg::FeaturedGraph) = nv(fg.graph[])
nv(fg::FeaturedGraph{T}) where {T<:AbstractMatrix} = size(fg.graph[], 1)
nv(fg::FeaturedGraph{T}) where {T<:AbstractVector} = length(fg.graph[])

"""
    ne(::AbstractFeaturedGraph)

Get edge number of graph.
"""
ne(::NullGraph) = 0
ne(fg::FeaturedGraph) = ne(fg.graph[])
ne(fg::FeaturedGraph{T}) where {T<:AbstractVector} = sum(map(length, fg.graph[]))÷2
