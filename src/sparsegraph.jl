abstract type AbstractSparseGraph <: AbstractGraph{Int} end

"""
    SparseGraph(A, directed, [T])

A sparse graph structure represents by sparse matrix.
A directed graph is represented by a sparse matrix, of which column
index as source node index and row index as sink node index.

# Arguments

- `A`: Adjacency matrix.
- `directed`: If this is a directed graph or not.
- `T`: Element type for `SparseGraph`.
"""
struct SparseGraph{D,M,V,T} <: AbstractSparseGraph
    S::M
    edges::V
    E::T
end

function SparseGraph{D}(A::AbstractMatrix{Tv}, edges::AbstractVector{Ti}, E::Integer) where {D,Tv,Ti}
    @assert size(A, 1) == size(A, 2) "A must be a square matrix."
    return SparseGraph{D,typeof(A),typeof(edges),typeof(E)}(A, edges, E)
end

function SparseGraph(
        A::AbstractMatrix{Tv},
        edges::AbstractVector{Ti},
        directed::Bool,
        ::Type{T}=eltype(A)
    ) where {Tv,Ti,T}
    E = length(unique(edges))
    spA = (Tv === T) ? SparseMatrixCSC{Tv,Ti}(A) : SparseMatrixCSC{T,Ti}(A)
    return SparseGraph{directed}(spA, edges, E)
end

SparseGraph(A::SparseCSC, directed::Bool, ::Type{T}=eltype(A)) where {T} =
    SparseGraph(A, order_edges(A, directed=directed), directed, T)
SparseGraph(A::AbstractMatrix, directed::Bool, ::Type{T}=eltype(A)) where {T} =
    SparseGraph(sparsecsc(A), directed, T)

function SparseGraph(
        adjl::AbstractVector{T},
        directed::Bool,
        ::Type{Te}=eltype(eltype(adjl))
    ) where {T<:AbstractVector,Te}
    n = length(adjl)
    colptr, rowval, nzval = to_csc(adjl)
    spA = SparseMatrixCSC(n, n, UInt32.(colptr), UInt32.(rowval), Te.(nzval))
    return SparseGraph(spA, directed)
end

SparseGraph(g::G, directed::Bool=is_directed(G), ::Type{T}=eltype(g)) where {G<:AbstractSimpleGraph,T} =
    SparseGraph(g.fadjlist, directed, T)

SparseGraph(g::G, directed::Bool=is_directed(G), ::Type{T}=eltype(g)) where {G<:AbstractSimpleWeightedGraph,T} =
    SparseGraph(weights(g)', directed, T)

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

struct SparseSubgraph{G<:AbstractSparseGraph,T} <: AbstractSparseGraph
    sg::G
    nodes::T
end

@functor SparseSubgraph

SparseArrays.sparse(sg::SparseGraph) = sg.S
SparseArrays.sparse(ss::SparseSubgraph) = sparse(ss.sg)[ss.nodes, ss.nodes]

Base.collect(sg::AbstractSparseGraph) = collect(sparse(sg))

Base.show(io::IO, sg::SparseGraph) =
    print(io, "SparseGraph{", eltype(sg), "}(#V=", nv(sg), ", #E=", ne(sg), ")")
Base.show(io::IO, ss::SparseSubgraph) =
    print(io, "subgraph of ", ss.sg, " with nodes=$(ss.nodes)")

Graphs.nv(sg::SparseGraph) = size(sparse(sg), 1)
Graphs.nv(ss::SparseSubgraph) = length(ss.nodes)

Graphs.ne(sg::SparseGraph) = sg.E
# Graphs.ne(ss::SparseSubgraph) = 

Graphs.is_directed(::SparseGraph{G}) where {G} = G
Graphs.is_directed(::Type{<:SparseGraph{G}}) where {G} = G
Graphs.is_directed(ss::SparseSubgraph) = is_directed(ss.sg)
Graphs.is_directed(::Type{<:SparseSubgraph{G}}) where {G} = is_directed(G)

function Graphs.has_self_loops(sg::SparseGraph)
    for i in vertices(sg)
        (i in rowvalview(sparse(sg), i)) && return true
    end
    return false
end

function Graphs.has_self_loops(ss::SparseSubgraph)
    for i in vertices(ss)
        (i in rowvalview(sparse(ss.sg), i)) && return true
    end
    return false
end

Base.eltype(sg::SparseGraph) = eltype(sparse(sg))
Base.eltype(ss::SparseSubgraph) = eltype(ss.sg)

Graphs.has_vertex(sg::SparseGraph, i::Integer) = 1 <= i <= nv(sg)
Graphs.has_vertex(ss::SparseSubgraph, i::Integer) = (i in ss.nodes)

Graphs.vertices(sg::SparseGraph) = 1:nv(sg)
Graphs.vertices(ss::SparseSubgraph) = ss.nodes

Graphs.edgetype(::AbstractSparseGraph) = Tuple{Int, Int}

Graphs.has_edge(sg::SparseGraph, i::Integer, j::Integer) = j ∈ SparseArrays.rowvals(sparse(sg), i)
Graphs.has_edge(ss::SparseSubgraph, i::Integer, j::Integer) =
    (i in ss.nodes && j in ss.nodes && has_edge(ss.sg, i, j))

Base.:(==)(sg1::SparseGraph, sg2::SparseGraph) =
    sg1.E == sg2.E && sg1.edges == sg2.edges && sg1.S == sg2.S
Base.:(==)(ss1::SparseSubgraph, ss2::SparseSubgraph) =
    ss1.nodes == ss2.nodes && ss1.sg == ss2.sg

graph(sg::SparseGraph) = sg

subgraph(sg::AbstractSparseGraph, nodes::AbstractVector) = SparseSubgraph(sg, nodes)
subgraph(ss::SparseSubgraph, nodes::AbstractVector) = SparseSubgraph(ss.sg, nodes)

edgevals(sg::SparseGraph) = sg.edges
edgevals(sg::SparseGraph, col::Integer) = view(sg.edges, SparseArrays.getcolptr(sparse(sg), col))
edgevals(sg::SparseGraph, I::UnitRange) = view(sg.edges, SparseArrays.getcolptr(sparse(sg), I))

"""
    neighbors(sg, i)

Return the neighbors of vertex `i` in sparse graph `sg`.

# Arguments

- `sg::SparseGraph`: sparse graph to query.
- `i`: vertex index.
"""
Graphs.neighbors(sg::SparseGraph{false}; dir::Symbol=:out) = rowvals(sparse(sg))

Graphs.neighbors(sg::SparseGraph{false}, i::Integer; dir::Symbol=:out) = rowvalview(sparse(sg), i)

function Graphs.neighbors(sg::SparseGraph{true}, i::Integer; dir::Symbol=:out)
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

Graphs.outneighbors(sg::SparseGraph{true}, i::Integer) = rowvalview(sparse(sg), i)

function Graphs.inneighbors(sg::SparseGraph{true}, i::Integer)
    S = sparse(sg)
    mask = [i in rowvalview(S, j) for j in 1:size(S, 2)]
    return findall(mask)
end

noutneighbors(sg::SparseGraph, i) =
    length(SparseArrays.getcolptr(SparseMatrixCSC(sparse(sg)), i))

"""
    incident_edges(sg, i)

Return the edges incident to vertex `i` in sparse graph `sg`.

# Arguments

- `sg::SparseGraph`: sparse graph to query.
- `i`: vertex index.
"""
incident_edges(sg::SparseGraph{false}) = edgevals(sg)

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
    S = sparse(sg)
    inedges = V()
    for j in 1:size(S, 2)
        mask = i in rowvalview(S, j)
        edges = edgevals(sg, j)
        append!(inedges, edges[findall(mask)])
    end
    return inedges
end

Base.getindex(sg::SparseGraph, ind...) = getindex(sparse(sg), ind...)
# Base.getindex(ss::SparseSubgraph, ind...) = 

edge_index(sg::SparseGraph, i, j) = sg.edges[get_csc_index(sparse(sg), j, i)]
# edge_index(ss::SparseSubgraph, i, j) = 

"""
Transform a CSC-based edge index `edges[eidx]` into a regular cartesian index `A[i, j]`.
"""
function get_cartesian_index(sg::SparseGraph, eidx::Int)
    r = rowvals(sparse(sg))
    idx = findfirst(x -> x == eidx, edgevals(sg))
    i = r[idx]
    j = 1
    while idx > noutneighbors(sg, 1:j)
        j += 1
    end
    return (i, j)
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

aggregate_index(sg::SparseGraph{true}, ::Val{:edge}, ::Val{:inward}) = rowvals(sparse(sg))

aggregate_index(sg::SparseGraph{true}, ::Val{:edge}, ::Val{:outward}) = colvals(sparse(sg))

function aggregate_index(sg::SparseGraph{false}, ::Val{:edge}, ::Val{:inward})
    # for undirected graph, upper traingle of matrix is considered only.
    S = sparse(sg)
    res = Int[]
    for j in 1:size(S, 2)
        r = rowvalview(S, j)
        r = view(r, r .≤ j)
        append!(res, r)
    end
    return res
end

# for undirected graph, upper traingle of matrix is considered only.
aggregate_index(sg::SparseGraph{false}, ::Val{:edge}, ::Val{:outward}) = colvals(sparse(sg), upper_traingle=true)

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


## Graph representations

adjacency_list(sg::SparseGraph) = [SparseArrays.rowvals(sparse(sg), j) for j in 1:size(sparse(sg), 2)]
adjacency_matrix(sg::SparseGraph) = adjacency_matrix(sparse(sg))


## Linear algebra

degrees(sg::SparseGraph, T::DataType=eltype(sg); dir::Symbol=:out) =
    degrees(sparse(sg), T; dir=dir)

degree_matrix(sg::SparseGraph, T::DataType=eltype(sg); dir::Symbol=:out) =
    degree_matrix(sparse(sg), T; dir=dir)

normalized_adjacency_matrix(sg::SparseGraph, T::DataType=eltype(sg); selfloop::Bool=false) =
    normalized_adjacency_matrix(sparse(sg), T; selfloop=selfloop)

laplacian_matrix(sg::SparseGraph, T::DataType=eltype(sg); dir::Symbol=:out) =
    laplacian_matrix(sparse(sg), T; dir=dir)

normalized_laplacian(sg::SparseGraph, T::DataType=eltype(sg); dir::Symbol=:both, selfloop::Bool=false) =
    normalized_laplacian(sparse(sg), T; selfloop=selfloop)

scaled_laplacian(sg::SparseGraph, T::DataType=eltype(sg)) = scaled_laplacian(sparse(sg), T)


## Edge iterator

struct EdgeIter{G,S}
    sg::G
    start::S

    function EdgeIter(sg::SparseGraph)
        S = SparseMatrixCSC(sparse(sg))
        j = 1
        while 1 > length(SparseArrays.getcolptr(S, 1:j))
            j += 1
        end
        i = rowvals(S)[1]
        e = collect(edgevals(sg))[1]
        start = (e, (i, j))
        return new{typeof(sg),typeof(start)}(sg, start)
    end
end

graph(iter::EdgeIter) = iter.sg
Graphs.edges(sg::SparseGraph) = EdgeIter(sg)
Base.length(iter::EdgeIter) = nnz(sparse(graph(iter)))

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

function Base.collect(iter::EdgeIter)
    g = graph(iter)
    return edgevals(g), rowvals(sparse(g)), colvals(sparse(g))
end
