SparseArrays.getcolptr(S::SparseMatrixCSC, col::Integer) = S.colptr[col]:(S.colptr[col+1]-1)
SparseArrays.getcolptr(S::SparseMatrixCSC, I::UnitRange) = S.colptr[I.start]:(S.colptr[I.stop+1]-1)

SparseArrays.rowvals(S::SparseMatrixCSC, col::Integer) = view(S.rowval, SparseArrays.getcolptr(S, col))
SparseArrays.rowvals(S::SparseMatrixCSC, I::UnitRange) = view(S.rowval, SparseArrays.getcolptr(S, I))

SparseArrays.nonzeros(S::SparseMatrixCSC, col::Integer) = view(S.nzval, SparseArrays.getcolptr(S, col))
SparseArrays.nonzeros(S::SparseMatrixCSC, I::UnitRange) = view(S.nzval, SparseArrays.getcolptr(S, I))


"""
    SparseGraph(A, directed)

A sparse graph structure represents by sparse matrix.
A directed graph is represented by a sparse matrix, of which column index as source node index and row index as sink node index.
"""
struct SparseGraph{D,M<:AbstractSparseMatrixCSC,V<:AbstractVector,T}
    S::M
    edges::V
    E::T
end

function SparseGraph(A::AbstractMatrix{Tv}, edges::AbstractVector{Ti}, directed::Bool) where {Tv,Ti}
    @assert size(A, 1) == size(A, 2) "A must be a square matrix."
    E = length(unique(edges))
    spA = SparseMatrixCSC{Tv,Ti}(A)
    return SparseGraph{directed,typeof(spA),typeof(edges),typeof(E)}(spA, edges, E)
end

SparseGraph(A::SparseMatrixCSC, directed::Bool) = SparseGraph(A, order_edges(A, directed=directed), directed)
SparseGraph(A::AbstractMatrix, directed::Bool) = SparseGraph(sparse(A), directed)

Base.show(io::IO, sg::SparseGraph) = print(io, "SparseGraph(#V=", nv(sg), ", #E=", ne(sg), ")")

nv(sg::SparseGraph) = size(sg.S, 1)
ne(sg::SparseGraph) = sg.E
is_directed(::SparseGraph{D}) where {D} = D

edgevals(sg::SparseGraph, col::Integer) = view(sg.edges, SparseArrays.getcolptr(sg.S, col))
edgevals(sg::SparseGraph, I::UnitRange) = view(sg.edges, SparseArrays.getcolptr(sg.S, I))

"""
    neighbors(sg, i)

Return the neighbors of vertex `i` in sparse graph `sg`.

# Arguments

- `sg::SparseGraph`: sparse graph to query.
- `i`: vertex index.
"""
neighbors(sg::SparseGraph{false}, i; dir::Symbol=:out) = SparseArrays.rowvals(sg.S, i)

function neighbors(sg::SparseGraph{true}, i; dir::Symbol=:out)
    if dir == :out
        return outneighbors(sg, i)
    elseif dir == :in
        return inneighbors(sg, i)
    elseif dir == :both
        return unique!(append!(inneighbors(sg, i), outneighbors(sg, i)))
    else
        throw(ArgumentError("dir must be one of [:out, :in, :both]."))
    end
end

outneighbors(sg::SparseGraph{true}, i) = SparseArrays.rowvals(sg.S, i)

function inneighbors(sg::SparseGraph{true}, i)
    mask = [i in SparseArrays.rowvals(sg.S, j) for j in 1:size(sg.S, 2)]
    return findall(mask)
end

"""
    incident_edges(sg, i)

Return the edges incident to vertex `i` in sparse graph `sg`.

# Arguments

- `sg::SparseGraph`: sparse graph to query.
- `i`: vertex index.
"""
incident_edges(sg::SparseGraph{false}, i) = edgevals(sg, i)

function incident_edges(sg::SparseGraph{true}, i; dir=:out)
    if dir == :out
        return incident_outedges(sg, i)
    elseif dir == :in
        return incident_inedges(sg, i)
    elseif dir == :both
        return append!(incident_inedges(sg, i), incident_outedges(sg, i))
    else
        throw(ArgumentError("dir must be one of [:out, :in, :both]."))
    end
end

incident_outedges(sg::SparseGraph{true}, i) = edgevals(sg, i)

function incident_inedges(sg::SparseGraph{true,M,V}, i) where {M,V}
    inedges = V()
    for j in 1:size(sg.S, 2)
        mask = i in SparseArrays.rowvals(sg.S, j)
        edges = edgevals(sg, j)
        append!(inedges, edges[findall(mask)])
    end
    return inedges
end

Base.getindex(sg::SparseGraph, ind...) = getindex(sg.S, ind...)
edge_index(sg::SparseGraph, i, j) = sg.edges[_to_csc_index(sg.S, i, j)]

"""
Order the edges in a graph by giving a unique integer to each edge.
"""
function order_edges(S::SparseMatrixCSC; directed::Bool=false)
    return order_edges!(similar(S.rowval), S, Val(directed))
end

function order_edges!(edges, S::SparseMatrixCSC, directed::Val{false})
    @assert issymmetric(S) "Matrix of undirected graph must be symmetric."
    k = 1
    for j in axes(S, 2)
        idx1 = SparseArrays.getcolptr(S, j)
        row = SparseArrays.rowvals(S, j)
        for idx2 in 1:length(row)
            idx = idx1[idx2]
            i = row[idx2]
            if i < j  # upper triangle
                edges[idx] = k
                edges[_to_csc_index(S, j, i)] = k
                k += 1
            elseif i == j  # diagonal
                edges[idx] = k
                k += 1
            end
        end
    end
    return edges
end

function order_edges!(edges, S::SparseMatrixCSC, directed::Val{true})
    for i in 1:length(edges)
        edges[i] = i
    end
    edges
end

"""
Transform a regular cartesian index `A[i, j]` into a CSC-compatible index `spA.nzval[idx]`.
"""
function _to_csc_index(S::SparseMatrixCSC, i::Integer, j::Integer)
    idx1 = SparseArrays.getcolptr(S, j)
    row = view(S.rowval, idx1)
    idx2 = findfirst(row .== i)
    return idx1[idx2]
end

"""
    aggregate_index(sg; direction=:undirected, kind=:edge)

Generate index structure for scatter operation.

# Arguments

- `sg::SparseGraph`: The reference graph.
- `direction::Symbol`: The direction of an edge to be choose to aggregate. It must be one of `:inward` and `:outward`.
- `kind::Symbol`: To aggregate feature upon edge or vertex. It must be one of `:edge` and `:vertex`.
"""
function aggregate_index(sg::SparseGraph, kind::Symbol=:edge, direction::Symbol=:outward)
    if !(kind in [:edge, :vertex])
        throw(ArgumentError("kind must be one of :edge or :vertex."))
    end

    if !(direction in [:inward, :outward])
        throw(ArgumentError("direction must be one of :outward or :inward."))
    end
    
    return aggregate_index(sg, Val(kind), Val(direction))
end

function aggregate_index(sg::SparseGraph{true}, ::Val{:edge}, ::Val{:inward})
    return rowvals(sg.S)
end

function aggregate_index(sg::SparseGraph{true}, ::Val{:edge}, ::Val{:outward})
    res = Int[]
    for j in 1:size(sg.S, 2)
        l = length(SparseArrays.getcolptr(sg.S, j))
        append!(res, repeat([j], l))
    end
    return res
end

function aggregate_index(sg::SparseGraph{false}, ::Val{:edge}, ::Val{:inward})
    # for undirected graph, upper traingle of matrix is considered only.
    res = Int[]
    for j in 1:size(sg.S, 2)
        r = rowvals(sg.S, j)
        r = view(r, r .≤ j)
        append!(res, r)
    end
    return res
end

function aggregate_index(sg::SparseGraph{false}, ::Val{:edge}, ::Val{:outward})
    # for undirected graph, upper traingle of matrix is considered only.
    res = Int[]
    for j in 1:size(sg.S, 2)
        l = length(SparseArrays.getcolptr(sg.S, j))
        c = repeat([j], l)
        r = rowvals(sg.S, j)
        c = view(c, c .≥ r)
        append!(res, c)
    end
    return res
end

function aggregate_index(sg::SparseGraph{true}, ::Val{:vertex}, ::Val{:inward})
    return [neighbors(sg, i, dir=:out) for i in 1:nv(sg)]
end

function aggregate_index(sg::SparseGraph{true}, ::Val{:vertex}, ::Val{:outward})
    return [neighbors(sg, i, dir=:in) for i in 1:nv(sg)]
end

function aggregate_index(sg::SparseGraph{false}, ::Val{:vertex}, ::Val{:inward})
    # for undirected graph, upper traingle of matrix is considered only.
    return [neighbors(sg, i, dir=:out) for i in 1:nv(sg)]
end

function aggregate_index(sg::SparseGraph{false}, ::Val{:vertex}, ::Val{:outward})
    # for undirected graph, upper traingle of matrix is considered only.
    return [neighbors(sg, i, dir=:in) for i in 1:nv(sg)]
end


"""
    edge_scatter(aggr, E, sg, direction=:undirected)

Scatter operation for aggregating edge feature into vertex feature.

# Arguments

- `aggr`: aggregating operators, e.g. `+`.
- `E`: Edge features with size of (#feature, #edge).
- `sg::SparseGraph`: The reference graph.
- `direction::Symbol`: The direction of an edge to be choose to aggregate. It must be one of `:undirected`, `:inward` and `:outward`.
"""
function edge_scatter(aggr, E::AbstractArray, sg::SparseGraph{D}; direction::Symbol=:undirected) where {D}
    if direction == :undirected || !D
        idx1 = aggregate_index(sg, :edge, :outward)
        idx2 = aggregate_index(sg, :edge, :inward)
        # idx may be incomplelely cover all index, this may cause some bug which makes dst shorter than expect
        # currently, scatter idx1 first can avoid this condition
        dst = NNlib.scatter(aggr, E, idx1)
        return NNlib.scatter!(aggr, dst, E, idx2)
    else
        idx = aggregate_index(sg, :edge, direction)
        return NNlib.scatter(aggr, E, idx)
    end
end

"""
    neighbor_scatter(aggr, X, sg, direction=:undirected)

Scatter operation for aggregating neighbor vertex feature together.

# Arguments

- `aggr`: aggregating operators, e.g. `+`.
- `X`: Vertex features with size of (#feature, #vertex).
- `sg::SparseGraph`: The reference graph.
- `direction::Symbol`: The direction of an edge to be choose to aggregate. It must be one of `:undirected`, `:inward` and `:outward`.
"""
function neighbor_scatter(aggr, X::AbstractArray, sg::SparseGraph; direction::Symbol=:undirected)
    direction == :undirected && (direction = :outward)
    idx = aggregate_index(sg, :vertex, direction)
    Ys = [neighbor_features(aggr, X, idx[i]) for i = 1:length(idx)]
    return hcat(Ys...)
end

function neighbor_features(aggr, X::AbstractArray{T}, idx) where {T}
    if isempty(idx)
        return fill(NNlib.scatter_empty(aggr, T), size(X, 1))
    else
        return mapreduce(j -> view(X,:,j), aggr, idx)
    end
end
