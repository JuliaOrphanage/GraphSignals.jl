function adjacency_list(g::AbstractSimpleWeightedGraph)
    N = nv(g)
    Vector{Int}[outneighbors(g, i) for i = 1:N]
end
