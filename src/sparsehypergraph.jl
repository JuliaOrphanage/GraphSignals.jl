struct SparseHyperGraph{G,M<:AbstractMatrix} <: AbstractSparseGraph
    H::M

    SparseHyperGraph{G}(H::M) where {G,M<:AbstractMatrix} = new{G,M}(H)
end

Graphs.nv(g::SparseHyperGraph) = size(incidence_matrix(g), 1)
Graphs.ne(g::SparseHyperGraph) = size(incidence_matrix(g), 2)

Graphs.is_directed(::SparseHyperGraph{G}) where {G} = G
Graphs.is_directed(::Type{<:SparseHyperGraph{G}}) where {G} = G

Base.eltype(g::SparseHyperGraph) = eltype(incidence_matrix(g))

Graphs.vertices(g::SparseHyperGraph) = 1:nv(g)

Graphs.has_vertex(g::SparseHyperGraph, i::Integer) = 1 <= i <= nv(g)

function Graphs.has_edge(g::SparseHyperGraph, edge...)
    edge = Tuple(sort!(collect(edge)))
    for (_, e) in edges(g)
        edge == e && return true
    end
    return false
end

Base.:(==)(g1::SparseHyperGraph, g2::SparseHyperGraph) =
    incidence_matrix(g1) == incidence_matrix(g2)

Graphs.incidence_matrix(g::SparseHyperGraph) = g.H

Graphs.outneighbors(g::SparseHyperGraph{true}) = map(i -> outneighbors(g, i), vertices(g))
Graphs.outneighbors(g::SparseHyperGraph{true}, i::Integer) = _neighbors(incidence_matrix(g), i, -1)

Graphs.inneighbors(g::SparseHyperGraph{true}) = map(i -> inneighbors(g, i), vertices(g))
Graphs.inneighbors(g::SparseHyperGraph{true}, i::Integer) = _neighbors(incidence_matrix(g), i, 1)

Graphs.neighbors(g::SparseHyperGraph{false}) = map(i -> neighbors(g, i), vertices(g))
Graphs.neighbors(g::SparseHyperGraph{false}, i::Integer) = _neighbors(incidence_matrix(g), i, 1)

function _neighbors(H::AbstractMatrix, i::Integer, identifier)
    nbrs = findall(vec(any(H[:, H[i, :] .!= 0] .== identifier, dims=2)))
    self_idx = nbrs .== i
    any(self_idx) && deleteat!(nbrs, self_idx)
    return nbrs
end

isneighbor(g::SparseHyperGraph, edge...) = any(all(incidence_matrix(g)[collect(edge), :] .!= 0, dims=1))

Graphs.outdegree(g::SparseHyperGraph{true}) = vec(sum(incidence_matrix(g) .== -1, dims=2))
Graphs.outdegree(g::SparseHyperGraph{true}, i::Integer) = sum(incidence_matrix(g)[i, :] .== -1)

Graphs.indegree(g::SparseHyperGraph{true}) = vec(sum(incidence_matrix(g) .== 1, dims=2))
Graphs.indegree(g::SparseHyperGraph{true}, i::Integer) = sum(incidence_matrix(g)[i, :] .== 1)

Graphs.degree(g::SparseHyperGraph) = vec(sum(incidence_matrix(g) .!= 0, dims=2))
Graphs.degree(g::SparseHyperGraph, i::Integer) = sum(incidence_matrix(g)[i, :] .!= 0)

function Graphs.laplacian_matrix(g::SparseHyperGraph, T::DataType=eltype(g))
    H = T.(incidence_matrix(g))
    return H * H'
end

dual(g::SparseHyperGraph) = SparseHyperGraph(incidence_matrix(g))


struct HyperEdgeIter{G,S}
    g::G
    start::S

    function HyperEdgeIter(g::SparseHyperGraph)
        if ne(g) == 0
            start = (0, (0, 0))
        else
            H = incidence_matrix(g)
            e = 1
            vs = findall(H[:, e] .!= 0)
            start = (e, tuple(vs...))
        end
        return new{typeof(g),typeof(start)}(g, start)
    end
end

graph(iter::HyperEdgeIter) = iter.g

Graphs.edges(g::SparseHyperGraph) = HyperEdgeIter(g)

Graphs.ne(iter::HyperEdgeIter) = ne(graph(iter))

Base.length(iter::HyperEdgeIter) = ne(iter)

function Base.iterate(iter::HyperEdgeIter, (el, i)=(iter.start, 1))
    H = incidence_matrix(graph(iter))
    next_i = i + 1
    if next_i <= length(iter)
        vs = findall(H[:, next_i] .!= 0)
        next_el = (next_i, tuple(vs...))
        return (el, (next_el, next_i))
    elseif next_i == length(iter) + 1
        next_el = (0, (0, 0))
        return (el, (next_el, next_i))
    else
        return nothing
    end
end

function Base.collect(iter::HyperEdgeIter)
    H = incidence_matrix(graph(iter))
    vs = [tuple(findall(H[:, e] .!= 0)...) for e in 1:ne(iter)]
    return 1:ne(iter), vs
end
