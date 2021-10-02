const SparseCSC = Union{SparseMatrixCSC,CuSparseMatrixCSC}

sparsecsc(A::AbstractMatrix) = sparse(A)
sparsecsc(A::AnyCuMatrix) = CuSparseMatrixCSC(A)

SparseArrays.getcolptr(S::SparseMatrixCSC, col::Integer) = S.colptr[col]:(S.colptr[col+1]-1)
SparseArrays.getcolptr(S::SparseMatrixCSC, I::UnitRange) = S.colptr[I.start]:(S.colptr[I.stop+1]-1)

SparseArrays.rowvals(S::SparseCSC, col::Integer) = rowvals(S)[SparseArrays.getcolptr(S, col)]
SparseArrays.rowvals(S::SparseCSC, I::UnitRange) = rowvals(S)[SparseArrays.getcolptr(S, I)]
rowvalview(S::SparseCSC, col::Integer) = view(rowvals(S), SparseArrays.getcolptr(S, col))
rowvalview(S::SparseCSC, I::UnitRange) = view(rowvals(S), SparseArrays.getcolptr(S, I))

SparseArrays.nonzeros(S::SparseCSC, col::Integer) = nonzeros(S)[SparseArrays.getcolptr(S, col)]
SparseArrays.nonzeros(S::SparseCSC, I::UnitRange) = nonzeros(S)[SparseArrays.getcolptr(S, I)]
SparseArrays.nzvalview(S::SparseCSC, col::Integer) = view(nonzeros(S), SparseArrays.getcolptr(S, col))
SparseArrays.nzvalview(S::SparseCSC, I::UnitRange) = view(nonzeros(S), SparseArrays.getcolptr(S, I))


"""
    SparseGraph(A, directed)

A sparse graph structure represents by sparse matrix.
A directed graph is represented by a sparse matrix, of which column index as source node index and row index as sink node index.
"""
struct SparseGraph{D,M,V,T} <: AbstractGraph{Int}
    S::M
    edges::V
    E::T
end

function SparseGraph{D}(A::AbstractMatrix{Tv}, edges::AbstractVector{Ti}, E::Integer) where {D,Tv,Ti}
    @assert size(A, 1) == size(A, 2) "A must be a square matrix."
    return SparseGraph{D,typeof(A),typeof(edges),typeof(E)}(A, edges, E)
end

function SparseGraph(A::AbstractMatrix{Tv}, edges::AbstractVector{Ti}, directed::Bool) where {Tv,Ti}
    E = length(unique(edges))
    spA = SparseMatrixCSC{Tv,Ti}(A)
    return SparseGraph{directed,typeof(spA),typeof(edges),typeof(E)}(spA, edges, E)
end

SparseGraph(A::SparseCSC, directed::Bool) = SparseGraph(A, order_edges(A, directed=directed), directed)
SparseGraph(A::AbstractMatrix, directed::Bool) = SparseGraph(sparsecsc(A), directed)

function SparseGraph(adjl::AbstractVector{T}, directed::Bool) where {T<:AbstractVector}
    n = length(adjl)
    colptr, rowval, nzval = to_csc(adjl)
    spA = SparseMatrixCSC(n, n, colptr, rowval, nzval)
    return SparseGraph(spA, directed)
end

SparseGraph(g::G, directed::Bool=is_directed(G)) where {G<:AbstractSimpleGraph} =
    SparseGraph(g.fadjlist, directed)

SparseGraph(g::G, directed::Bool=is_directed(G)) where {G<:AbstractSimpleWeightedGraph} =
    SparseGraph(weights(g)', directed)

function to_csc(adjl::AbstractVector{T}) where {T<:AbstractVector}
    ET = eltype(adjl[1])
    colptr = ET[1, ]
    rowval = ET[]
    for nbs in adjl
        r = sort!(unique(nbs))
        push!(colptr, colptr[end] + length(r))
        append!(rowval, r)
    end
    nzval = ones(ET, length(rowval))

    return colptr, rowval, nzval
end

@functor SparseGraph{true}
@functor SparseGraph{false}

Base.show(io::IO, sg::SparseGraph) = print(io, "SparseGraph(#V=", nv(sg), ", #E=", ne(sg), ")")

LightGraphs.nv(sg::SparseGraph) = size(sg.S, 1)
LightGraphs.ne(sg::SparseGraph) = sg.E
LightGraphs.is_directed(::SparseGraph{D}) where {D} = D
LightGraphs.is_directed(::Type{<:SparseGraph{D}}) where {D} = D

function LightGraphs.has_self_loops(sg::SparseGraph)
    n = nv(sg)
    for i = 1:n
        (i in rowvalview(sg.S, i)) && return true
    end
    return false
end

Base.eltype(sg::SparseGraph) = eltype(sg.S)
LightGraphs.has_vertex(sg::SparseGraph, i::Integer) = 1 <= i <= nv(sg)
LightGraphs.vertices(sg::SparseGraph) = 1:nv(sg)

LightGraphs.edgetype(sg::SparseGraph) = Tuple{Int, Int}
LightGraphs.has_edge(sg::SparseGraph, i::Integer, j::Integer) = i ∈ SparseArrays.rowvals(sg.S, j)

Base.:(==)(sg1::SparseGraph, sg2::SparseGraph) =
    sg1.E == sg2.E && sg1.edges == sg2.edges && sg1.S == sg2.S

edgevals(sg::SparseGraph) = sg.edges
edgevals(sg::SparseGraph, col::Integer) = view(sg.edges, SparseArrays.getcolptr(sg.S, col))
edgevals(sg::SparseGraph, I::UnitRange) = view(sg.edges, SparseArrays.getcolptr(sg.S, I))

"""
    neighbors(sg, i)

Return the neighbors of vertex `i` in sparse graph `sg`.

# Arguments

- `sg::SparseGraph`: sparse graph to query.
- `i`: vertex index.
"""
LightGraphs.neighbors(sg::SparseGraph{false}, i::Integer; dir::Symbol=:out) = rowvalview(sg.S, i)

function LightGraphs.neighbors(sg::SparseGraph{true}, i::Integer; dir::Symbol=:out)
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

LightGraphs.outneighbors(sg::SparseGraph{true}, i::Integer) = rowvalview(sg.S, i)

function LightGraphs.inneighbors(sg::SparseGraph{true}, i::Integer)
    mask = [i in rowvalview(sg.S, j) for j in 1:size(sg.S, 2)]
    return findall(mask)
end

noutneighbors(sg::SparseGraph, col::Integer) = length(SparseArrays.getcolptr(sg.S, col))
noutneighbors(sg::SparseGraph, I::UnitRange) = length(SparseArrays.getcolptr(sg.S, I))

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
        mask = i in rowvalview(sg.S, j)
        edges = edgevals(sg, j)
        append!(inedges, edges[findall(mask)])
    end
    return inedges
end

Base.getindex(sg::SparseGraph, ind...) = getindex(sg.S, ind...)
edge_index(sg::SparseGraph, i, j) = sg.edges[get_csc_index(sg.S, i, j)]

"""
Transform a CSC-based edge index `edges[eidx]` into a regular cartesian index `A[i, j]`.
"""
function get_cartesian_index(sg::SparseGraph, eidx::Int)
    r = rowvals(sg.S)
    idx = findfirst(x -> x == eidx, edgevals(sg))
    i = r[idx]
    j = 1
    while idx > noutneighbors(sg, 1:j)
        j += 1
    end
    return (i, j)
end

"""
Transform a regular cartesian index `A[i, j]` into a CSC-compatible index `spA.nzval[idx]`.
"""
function get_csc_index(S::SparseCSC, i::Integer, j::Integer)
    idx1 = SparseArrays.getcolptr(S, j)
    row = view(rowvals(S), idx1)
    idx2 = findfirst(x -> x == i, row)
    return idx1[idx2]
end

"""
Order the edges in a graph by giving a unique integer to each edge.
"""
order_edges(S::SparseCSC; directed::Bool=false) = order_edges!(similar(rowvals(S)), S, Val(directed))

function order_edges!(edges, S::SparseCSC, directed::Val{false})
    @assert issymmetric(S) "Matrix of undirected graph must be symmetric."
    k = 1
    for j in axes(S, 2)
        idx1 = SparseArrays.getcolptr(S, j)
        row = rowvalview(S, j)
        for idx2 in 1:length(row)
            idx = idx1[idx2]
            i = row[idx2]
            if i < j  # upper triangle
                edges[idx] = k
                edges[get_csc_index(S, j, i)] = k
                k += 1
            elseif i == j  # diagonal
                edges[idx] = k
                k += 1
            end
        end
    end
    return edges
end

function order_edges!(edges::T, S::SparseCSC, directed::Val{true}) where {T}
    edges .= T(1:length(edges))
    edges
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

aggregate_index(sg::SparseGraph{true}, ::Val{:edge}, ::Val{:inward}) = rowvals(sg.S)

function aggregate_index(sg::SparseGraph{true}, ::Val{:edge}, ::Val{:outward})
    cols = size(sg.S, 2)
    colptr = collect(SparseArrays.getcolptr(sg.S))
    ls = view(colptr, 2:(cols+1)) - view(colptr, 1:cols)
    pushfirst!(ls, 1)
    cumsum!(ls, ls)
    res = similar(sg.edges)
    for j in 1:size(sg.S, 2)
        fill!(view(res, ls[j]:(ls[j+1]-1)), j)
    end
    return res
end

function aggregate_index(sg::SparseGraph{false}, ::Val{:edge}, ::Val{:inward})
    # for undirected graph, upper traingle of matrix is considered only.
    res = Int[]
    for j in 1:size(sg.S, 2)
        r = rowvalview(sg.S, j)
        r = view(r, r .≤ j)
        append!(res, r)
    end
    return res
end

function aggregate_index(sg::SparseGraph{false}, ::Val{:edge}, ::Val{:outward})
    # for undirected graph, upper traingle of matrix is considered only.
    ls = [count(rowvalview(sg.S, j) .≤ j) for j in 1:size(sg.S, 2)]
    pushfirst!(ls, 1)
    cumsum!(ls, ls)
    res = similar(sg.edges, ls[end]-1)
    for j in 1:size(sg.S, 2)
        fill!(view(res, ls[j]:(ls[j+1]-1)), j)
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
    edge_scatter(aggr, E, sg, direction=:outward)

Scatter operation for aggregating edge feature into vertex feature.

# Arguments

- `aggr`: aggregating operators, e.g. `+`.
- `E`: Edge features of dimension (#feature, #edge).
- `sg::SparseGraph`: The reference graph.
- `direction::Symbol`: The direction of an edge to be choose to aggregate.
    It must be one of `:undirected`, `:inward` and `:outward`.
"""
function edge_scatter(aggr, E::AbstractArray, sg::SparseGraph{D}; direction::Symbol=:outward) where {D}
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
    neighbor_scatter(aggr, X, sg, direction=:outward)

Scatter operation for aggregating neighbor vertex feature together.

# Arguments

- `aggr`: aggregating operators, e.g. `+`.
- `X`: Vertex features of dimension (#feature, #vertex).
- `sg::SparseGraph`: The reference graph.
- `direction::Symbol`: The direction of an edge to be choose to aggregate.
    It must be one of `:undirected`, `:inward` and `:outward`.
"""
function neighbor_scatter(aggr, X::AbstractArray, sg::SparseGraph; direction::Symbol=:outward)
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


## Graph representations

adjacency_list(sg::SparseGraph) = [SparseArrays.rowvals(sg.S, j) for j in 1:size(sg.S, 2)]
LightGraphs.adjacency_matrix(sg::SparseGraph, T::DataType=eltype(sg)) = T.(sg.S)


## Linear algebra

LightGraphs.degrees(sg::SparseGraph, T::DataType=eltype(sg); dir::Symbol=:out) =
    degrees(sg.S, T; dir=dir)

GraphLaplacians.degree_matrix(sg::SparseGraph, T::DataType=eltype(sg); dir::Symbol=:out) =
    GraphLaplacians.degree_matrix(sg.S, T; dir=dir)

GraphLaplacians.laplacian_matrix(sg::SparseGraph, T::DataType=eltype(sg); dir::Symbol=:out) =
    GraphLaplacians.laplacian_matrix(sg.S, T; dir=dir)

GraphLaplacians.normalized_laplacian(sg::SparseGraph, T::DataType=eltype(sg);
                                     dir::Symbol=:both, selfloop::Bool=false) =
    GraphLaplacians.normalized_laplacian(sg.S, T; selfloop=selfloop)

GraphLaplacians.scaled_laplacian(sg::SparseGraph, T::DataType=eltype(sg)) =
    GraphLaplacians.scaled_laplacian(sg.S, T)


## Edge iterator

struct EdgeIter{G,S}
    sg::G
    start::S

    function EdgeIter(sg::SparseGraph)
        j = 1
        while 1 > noutneighbors(sg, 1:j)
            j += 1
        end
        i = rowvals(sg.S)[1]
        e = edgevals(sg)[1]
        start = (e, (i, j))
        return new{typeof(sg),typeof(start)}(sg, start)
    end
end

LightGraphs.edges(sg::SparseGraph) = EdgeIter(sg)
Base.length(iter::EdgeIter) = iter.sg.E

function Base.iterate(iter::EdgeIter, (el, i)=(iter.start, 1))
    next_i = i + 1
    if next_i <= ne(iter.sg)
        car_idx = get_cartesian_index(iter.sg, next_i)
        next_el = (next_i, car_idx)
        return (el, (next_el, next_i))
    elseif next_i == ne(iter.sg) + 1
        next_el = (0, (0, 0))
        return (el, (next_el, next_i))
    else
        return nothing
    end
end
