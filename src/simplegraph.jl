using LightGraphs: AbstractSimpleGraph, nv, outneighbors

function adjacency_list(g::AbstractSimpleGraph)
    N = nv(g)
    Vector{Int}[outneighbors(g, i) for i = 1:N]
end
