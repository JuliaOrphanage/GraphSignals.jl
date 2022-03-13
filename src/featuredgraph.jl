const MATRIX_TYPES = [:adjm, :normedadjm, :laplacian, :normalized, :scaled]
const DIRECTEDS = [:auto, :directed, :undirected]

abstract type AbstractFeaturedGraph end

"""
    NullGraph()

Null object for `FeaturedGraph`.
"""
struct NullGraph <: AbstractFeaturedGraph end

"""
    FeaturedGraph(g, [mt]; directed=:auto, nf, ef, gf, T, N, E)

A type representing a graph structure and storing also arrays 
that contain features associated to nodes, edges, and the whole graph. 

A `FeaturedGraph` can be constructed out of different objects `g` representing
the connections inside the graph.
When constructed from another featured graph `fg`, the internal graph representation
is preserved and shared.

# Arguments

- `g`: Data representing the graph topology. Possible type are 
    - An adjacency matrix.
    - An adjacency list.
    - A Graphs' graph, i.e. `SimpleGraph`, `SimpleDiGraph` from Graphs, or `SimpleWeightedGraph`,
        `SimpleWeightedDiGraph` from SimpleWeightedGraphs.
    - An `AbstractFeaturedGraph` object.
- `mt::Symbol`: Matrix type for `g` in matrix form. if `graph` is in matrix form, `mt` is recorded as one of `:adjm`,
    `:normedadjm`, `:laplacian`, `:normalized` or `:scaled`.
- `directed`: It specify that direction of a graph. It can be `:auto`, `:directed` and `:undirected`.
    Default value is `:auto`, which infers direction automatically.
- `nf`: Node features.
- `ef`: Edge features.
- `gf`: Global features.
- `T`: It specifies the element type of graph. Default value is the element type of `g`.
- `N`: Number of nodes for `g`.
- `E`: Number of edges for `g`.

# Usage

```
using GraphSignals, CUDA

# Construct from adjacency list representation
g = [[2,3], [1,4,5], [1], [2,5], [2,4]]
fg = FeaturedGraph(g)

# Number of nodes and edges
nv(fg)  # 5
ne(fg)  # 10

# From a Graphs' graph
fg = FeaturedGraph(erdos_renyi(100, 20))

# Copy featured graph while also adding node features
fg = FeaturedGraph(fg, nf=rand(100, 5))

# Send to gpu
fg = fg |> cu
```

See also [`graph`](@ref), [`node_feature`](@ref), [`edge_feature`](@ref), and [`global_feature`](@ref).
"""
mutable struct FeaturedGraph{T,Tn,Te,Tg} <: AbstractFeaturedGraph
    graph::T
    nf::Tn
    ef::Te
    gf::Tg
    matrix_type::Symbol

    function FeaturedGraph(graph::SparseGraph, nf::Tn, ef::Te, gf::Tg,
                           mt::Symbol) where {Tn<:AbstractMatrix,Te<:AbstractMatrix,Tg<:AbstractVector}
        mt ∈ MATRIX_TYPES || throw(ArgumentError("matrix_type must be one of :adjm, :normedadjm, :laplacian, :normalized or :scaled"))
        new{typeof(graph),Tn,Te,Tg}(graph, nf, ef, gf, mt)
    end
    function FeaturedGraph{T,Tn,Te,Tg}(graph, nf, ef, gf, mt
            ) where {T,Tn<:AbstractMatrix,Te<:AbstractMatrix,Tg<:AbstractVector}
        mt ∈ MATRIX_TYPES || throw(ArgumentError("matrix_type must be one of :adjm, :normedadjm, :laplacian, :normalized or :scaled"))
        new{T,Tn,Te,Tg}(T(graph), Tn(nf), Te(ef), Tg(gf), mt)
    end
end

@functor FeaturedGraph

FeaturedGraph() = NullGraph()

function FeaturedGraph(graph, mat_type::Symbol; directed::Symbol=:auto, T=eltype(graph), N=nv(graph), E=ne(graph),
                       nf=Fill(zero(T), (0, N)), ef=Fill(zero(T), (0, E)), gf=Fill(zero(T), 0))
    @assert directed ∈ DIRECTEDS "directed must be one of :auto, :directed and :undirected"
    dir = (directed == :auto) ? is_directed(graph) : directed == :directed
    return FeaturedGraph(SparseGraph(graph, dir, T), nf, ef, gf, mat_type)
end

## Graph from JuliaGraphs

FeaturedGraph(graph::AbstractGraph; kwargs...) = FeaturedGraph(graph, :adjm; T=Float32, kwargs...)

## Graph in adjacency list

function FeaturedGraph(graph::AbstractVector{T}; ET=eltype(graph[1]), kwargs...) where {T<:AbstractVector}
    return FeaturedGraph(graph, :adjm; T=ET, kwargs...)
end

## Graph in adjacency matrix

function FeaturedGraph(graph::AbstractMatrix{T}; N=nv(graph), nf=Fill(zero(T), (0, N)), kwargs...) where T
    return FeaturedGraph(graph, :adjm; N=N, nf=nf, kwargs...)
end

FeaturedGraph(ng::NullGraph) = ng

function FeaturedGraph(fg::FeaturedGraph; nf=node_feature(fg), ef=edge_feature(fg), gf=global_feature(fg))
    return FeaturedGraph(graph(fg), nf, ef, gf, matrixtype(fg))
end

"""
    ConcreteFeaturedGraph(fg; kwargs...)

This is a syntax sugar for construction for `FeaturedGraph` and `FeaturedSubgraph` object.
It is an idempotent operation, which gives the same type of object as inputs.
It wraps input `fg` again but reconfigures with `kwargs`.

# Arguments

- `fg`: `FeaturedGraph` and `FeaturedSubgraph` object.

# Usage

```jldoctest
julia> using GraphSignals

julia> adjm = [0 1 1 1;
               1 0 1 0;
               1 1 0 1;
               1 0 1 0];

julia> nf = rand(10, 4);

julia> fg = FeaturedGraph(adjm; nf=nf)
FeaturedGraph:
	Undirected graph with (#V=4, #E=5) in adjacency matrix
	Node feature:	ℝ^10 <Matrix{Float64}>

julia> ConcreteFeaturedGraph(fg, nf=rand(7, 4))
FeaturedGraph:
    Undirected graph with (#V=4, #E=5) in adjacency matrix
    Node feature:	ℝ^7 <Matrix{Float64}>
```

"""
ConcreteFeaturedGraph(fg::FeaturedGraph; kwargs...) = FeaturedGraph(fg; kwargs...)


## dimensional checks

function check_num_nodes(graph_nv::Real, N::Real)
    if graph_nv != N
        throw(DimensionMismatch("number of nodes must match between graph ($graph_nv) and node features ($N)"))
    end
end

function check_num_edges(graph_ne::Real, E::Real)
    if graph_ne != E
        throw(DimensionMismatch("number of edges must match between graph ($graph_ne) and edge features ($E)"))
    end
end

check_num_nodes(graph_nv::Real, nf) = check_num_nodes(graph_nv, size(nf, ndims(nf)))
check_num_edges(graph_ne::Real, ef) = check_num_edges(graph_ne, size(ef, ndims(ef)))

check_num_nodes(g, nf) = check_num_nodes(nv(g), nf)
check_num_edges(g, ef) = check_num_edges(ne(g), ef)

function check_precondition(graph, nf, ef, mt::Symbol)
    check_num_edges(ne(graph), ef)
    check_num_nodes(nv(graph), nf)
    return
end


## show

function Base.show(io::IO, fg::FeaturedGraph)
    direct = is_directed(fg) ? "Directed" : "Undirected"
    println(io, "FeaturedGraph:")
    print(io, "\t", direct, " graph with (#V=", nv(fg), ", #E=", ne(fg), ") in ", matrixrepr(fg))
    has_node_feature(fg) && print(io, "\n\tNode feature:\tℝ^", nf_dims_repr(fg), " <", typeof(fg.nf), ">")
    has_edge_feature(fg) && print(io, "\n\tEdge feature:\tℝ^", ef_dims_repr(fg), " <", typeof(fg.ef), ">")
    has_global_feature(fg) && print(io, "\n\tGlobal feature:\tℝ^", gf_dims_repr(fg), " <", typeof(fg.gf), ">")
end

matrixrepr(fg::FeaturedGraph) = matrixrepr(Val(matrixtype(fg)))
matrixrepr(::Val{:adjm}) = "adjacency matrix"
matrixrepr(::Val{:normedadjm}) = "normalized adjacency matrix"
matrixrepr(::Val{:laplacian}) = "Laplacian matrix"
matrixrepr(::Val{:normalized}) = "normalized Laplacian"
matrixrepr(::Val{:scaled}) = "scaled Laplacian"

nf_dims_repr(fg::FeaturedGraph) = size(fg.nf, 1)
ef_dims_repr(fg::FeaturedGraph) = size(fg.ef, 1)
gf_dims_repr(fg::FeaturedGraph) = size(fg.gf, 1)


## Accessing

matrixtype(fg::FeaturedGraph) = fg.matrix_type

Graphs.is_directed(fg::FeaturedGraph) = is_directed(graph(fg))

function Base.setproperty!(fg::FeaturedGraph, prop::Symbol, x)
    if prop == :graph
        check_num_nodes(x, fg.nf)
        check_num_edges(x, fg.ef)
    elseif prop == :nf
        check_num_nodes(fg.graph, x)
    elseif prop == :ef
        check_num_edges(fg.graph, x)
    end
    setfield!(fg, prop, x)
end

"""
    graph(fg)

Get referenced graph in `fg`.

# Arguments

- `fg::AbstractFeaturedGraph`: A concrete object of `AbstractFeaturedGraph` type.
"""
graph(::NullGraph) = nothing
graph(fg::FeaturedGraph) = fg.graph

Base.parent(fg::FeaturedGraph) = fg

"""
    node_feature(fg)

Get node feature attached to `fg`.

# Arguments

- `fg::AbstractFeaturedGraph`: A concrete object of `AbstractFeaturedGraph` type.
"""
node_feature(::NullGraph) = nothing
node_feature(fg::FeaturedGraph) = fg.nf

"""
    edge_feature(fg)

Get edge feature attached to `fg`.

# Arguments

- `fg::AbstractFeaturedGraph`: A concrete object of `AbstractFeaturedGraph` type.
"""
edge_feature(::NullGraph) = nothing
edge_feature(fg::FeaturedGraph) = fg.ef

"""
    global_feature(fg)

Get global feature attached to `fg`.

# Arguments

- `fg::AbstractFeaturedGraph`: A concrete object of `AbstractFeaturedGraph` type.
"""
global_feature(::NullGraph) = nothing
global_feature(fg::FeaturedGraph) = fg.gf

"""
    has_graph(fg)

Check if `graph` is available or not for `fg`.

# Arguments

- `fg::AbstractFeaturedGraph`: A concrete object of `AbstractFeaturedGraph` type.
"""
has_graph(::NullGraph) = false
has_graph(fg::FeaturedGraph) = fg.graph != Fill(0., (0,0))

"""
    has_node_feature(::AbstractFeaturedGraph)

Check if `node_feature` is available or not for `fg`.

# Arguments

- `fg::AbstractFeaturedGraph`: A concrete object of `AbstractFeaturedGraph` type.
"""
has_node_feature(::NullGraph) = false
has_node_feature(fg::FeaturedGraph) = !isempty(fg.nf)

"""
    has_edge_feature(::AbstractFeaturedGraph)

Check if `edge_feature` is available or not for `fg`.

# Arguments

- `fg::AbstractFeaturedGraph`: A concrete object of `AbstractFeaturedGraph` type.
"""
has_edge_feature(::NullGraph) = false
has_edge_feature(fg::FeaturedGraph) = !isempty(fg.ef)

"""
    has_global_feature(::AbstractFeaturedGraph)

Check if `global_feature` is available or not for `fg`.

# Arguments

- `fg::AbstractFeaturedGraph`: A concrete object of `AbstractFeaturedGraph` type.
"""
has_global_feature(::NullGraph) = false
has_global_feature(fg::FeaturedGraph) = !isempty(fg.gf)


## Graph property

"""
    nv(fg)

Get node number of graph in `fg`.

# Arguments

- `fg::AbstractFeaturedGraph`: A concrete object of `AbstractFeaturedGraph` type.
"""
Graphs.nv(::NullGraph) = 0
Graphs.nv(fg::FeaturedGraph) = nv(graph(fg))

"""
    ne(fg)

Get edge number of in `fg`.

# Arguments

- `fg::AbstractFeaturedGraph`: A concrete object of `AbstractFeaturedGraph` type.
"""
Graphs.ne(::NullGraph) = 0
Graphs.ne(fg::FeaturedGraph) = ne(graph(fg))

Graphs.vertices(fg::FeaturedGraph) = vertices(graph(fg))

Graphs.edges(fg::FeaturedGraph) = edges(graph(fg))

Graphs.neighbors(fg::FeaturedGraph; dir::Symbol=:out) = neighbors(graph(fg); dir=dir)
Graphs.neighbors(fg::FeaturedGraph, i::Integer; dir::Symbol=:out) = neighbors(graph(fg), i, dir=dir)

Graphs.has_edge(fg::FeaturedGraph, i::Integer, j::Integer) = has_edge(graph(fg), i, j)

incident_edges(fg::FeaturedGraph) = incident_edges(graph(fg))


## Graph representations

"""
    adjacency_list(fg)

Get adjacency list of graph in `fg`.

# Arguments

- `fg::AbstractFeaturedGraph`: A concrete object of `AbstractFeaturedGraph` type.
"""
adjacency_list(::NullGraph) = [zeros(0)]
adjacency_list(fg::FeaturedGraph) = adjacency_list(graph(fg))

adjacency_matrix(fg::FeaturedGraph) = adjacency_matrix(graph(fg))


## Linear algebra

degrees(fg::FeaturedGraph, T::DataType=eltype(graph(fg)); dir::Symbol=:out) =
    degrees(graph(fg), T; dir=dir)

degree_matrix(fg::FeaturedGraph, T::DataType=eltype(graph(fg)); dir::Symbol=:out) =
    degree_matrix(graph(fg), T; dir=dir)

normalized_adjacency_matrix(fg::FeaturedGraph, T::DataType=eltype(graph(fg)); selfloop::Bool=false) =
    normalized_adjacency_matrix(graph(fg), T; selfloop=selfloop)

laplacian_matrix(fg::FeaturedGraph, T::DataType=eltype(graph(fg)); dir::Symbol=:out) =
    laplacian_matrix(graph(fg), T; dir=dir)

normalized_laplacian(fg::FeaturedGraph, T::DataType=eltype(graph(fg));
                                     dir::Symbol=:both, selfloop::Bool=false) =
    normalized_laplacian(graph(fg), T; selfloop=selfloop)

scaled_laplacian(fg::FeaturedGraph, T::DataType=eltype(graph(fg))) = scaled_laplacian(graph(fg), T)


## Inplace operations

function normalized_adjacency_matrix!(fg::FeaturedGraph, T::DataType=eltype(graph(fg)); selfloop::Bool=false)
    if fg.matrix_type == :adjm
        fg.graph.S .= normalized_adjacency_matrix(graph(fg), T; selfloop=selfloop)
        fg.matrix_type = :normedadjm
    end
    fg
end

function laplacian_matrix!(fg::FeaturedGraph, T::DataType=eltype(graph(fg)); dir::Symbol=:out)
    if fg.matrix_type == :adjm
        fg.graph.S .= laplacian_matrix(graph(fg), T; dir=dir)
        fg.matrix_type = :laplacian
    end
    fg
end

function normalized_laplacian!(fg::FeaturedGraph, T::DataType=eltype(graph(fg));
                               dir::Symbol=:both, selfloop::Bool=false)
    if fg.matrix_type == :adjm
        fg.graph.S .= normalized_laplacian(graph(fg), T; dir=dir, selfloop=selfloop)
        fg.matrix_type = :normalized
    end
    fg
end

function scaled_laplacian!(fg::FeaturedGraph, T::DataType=eltype(graph(fg)))
    if fg.matrix_type == :adjm
        fg.graph.S .= scaled_laplacian(graph(fg), T)
        fg.matrix_type = :scaled
    end
    fg
end


## sample

StatsBase.sample(fg::FeaturedGraph, n::Int) =
    subgraph(fg, sample(vertices(graph(fg)), n; replace=false))
