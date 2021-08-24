const MATRIX_TYPES = [:adjm, :laplacian, :normalized, :scaled]
const DIRECTEDS = [:auto, :directed, :undirected]

abstract type AbstractFeaturedGraph end

"""
    NullGraph()

Null object for `FeaturedGraph`.
"""
struct NullGraph <: AbstractFeaturedGraph end

"""
    FeaturedGraph(g, [mt]; nf, ef, gf, directed)

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
    - A LightGraphs' graph, i.e. `SimpleGraph`, `SimpleDiGraph` from LightGraphs, or `SimpleWeightedGraph`,
        `SimpleWeightedDiGraph` from SimpleWeightedGraphs.
    - An `AbstractFeaturedGraph` object.
- `mt`: matrix type for `g` in matrix form. if `graph` is in matrix form, `mt` is recorded as one of `:adjm`,
    `:laplacian`, `:normalized` or `:scaled`.
- `nf`: Node features.
- `ef`: Edge features.
- `gf`: Global features.


# Usage

```
using GraphSignals, CUDA

# Construct from adjacency list representation
g = [[2,3], [1,4,5], [1], [2,5], [2,4]]
fg = FeaturedGraph(g)

# Number of nodes and edges
nv(fg)  # 5
ne(fg)  # 10

# From a LightGraphs' graph
fg = FeaturedGraph(erdos_renyi(100, 20))

# Copy featured graph while also adding node features
fg = FeaturedGraph(fg, nf=rand(100, 5))

# Send to gpu
fg = fg |> cu
```

See also [`graph`](@ref), [`node_feature`](@ref), [`edge_feature`](@ref), and [`global_feature`](@ref)
"""
mutable struct FeaturedGraph{T,Tn,Te,Tg} <: AbstractFeaturedGraph
    graph::T
    nf::Tn
    ef::Te
    gf::Tg
    matrix_type::Symbol

    function FeaturedGraph(graph::SparseGraph, nf::Tn, ef::Te, gf::Tg,
                           mt::Symbol) where {Tn<:AbstractMatrix,Te<:AbstractMatrix,Tg<:AbstractVector}
        check_precondition(graph, nf, ef, mt)
        new{typeof(graph),Tn,Te,Tg}(graph, nf, ef, gf, mt)
    end
    function FeaturedGraph{T,Tn,Te,Tg}(graph, nf, ef, gf, mt
            ) where {T,Tn<:AbstractMatrix,Te<:AbstractMatrix,Tg<:AbstractVector}
        check_precondition(graph, nf, ef, mt)
        new{T,Tn,Te,Tg}(T(graph), Tn(nf), Te(ef), Tg(gf), mt)
    end
end

@functor FeaturedGraph

FeaturedGraph() = NullGraph()

function FeaturedGraph(graph, mat_type::Symbol; directed::Symbol=:auto, T=eltype(graph), N=nv(graph), E=ne(graph),
                       nf=Fill(zero(T), (0, N)), ef=Fill(zero(T), (0, E)), gf=Fill(zero(T), 0))
    @assert directed ∈ DIRECTEDS "directed must be one of :auto, :directed and :undirected"
    dir = (directed == :auto) ? is_directed(graph) : directed == :directed
    return FeaturedGraph(SparseGraph(graph, dir), nf, ef, gf, mat_type)
end

## Graph from JuliaGraphs

FeaturedGraph(graph::AbstractGraph; kwargs...) = FeaturedGraph(graph, :adjm; kwargs...)

## Graph in adjacency list

function FeaturedGraph(graph::AbstractVector{T}; ET=eltype(graph[1]), kwargs...) where {T<:AbstractVector}
    return FeaturedGraph(graph, :adjm; T=ET, kwargs...)
end

## Graph in adjacency matrix

function FeaturedGraph(graph::AbstractMatrix{T}; N=nv(graph), nf=Fill(zero(T), (0, N)), kwargs...) where T
    graph = promote_graph(graph, nf)
    return FeaturedGraph(graph, :adjm; N=N, nf=nf, kwargs...)
end

FeaturedGraph(ng::NullGraph) = ng

function FeaturedGraph(fg::FeaturedGraph; nf=node_feature(fg), ef=edge_feature(fg), gf=global_feature(fg))
    return FeaturedGraph(graph(fg), nf, ef, gf, matrixtype(fg))
end


## dimensional checks

function check_num_node(graph_nv::Real, N::Real)
    if graph_nv != N
        throw(DimensionMismatch("number of nodes must match between graph ($graph_nv) and node features ($N)"))
    end
end

function check_num_edge(graph_ne::Real, E::Real)
    # allow for the number of edge in directed and undirected graph
    if graph_ne != E && 2*graph_ne != E
        throw(DimensionMismatch("number of edges must match between graph ($graph_ne) and edge features ($E)"))
    end
end

check_num_node(graph_nv::Real, nf) = check_num_node(graph_nv, size(nf, 2))
check_num_edge(graph_ne::Real, ef) = check_num_edge(graph_ne, size(ef, 2))

check_num_node(g, nf) = check_num_node(nv(g), nf)
check_num_edge(g, ef) = check_num_edge(ne(g), ef)

function check_precondition(graph, nf, ef, mt::Symbol)
    @assert mt ∈ MATRIX_TYPES "matrix_type must be one of :adjm, :laplacian, :normalized or :scaled"
    check_num_edge(ne(graph), ef)
    check_num_node(nv(graph), nf)
    return
end


## show

function Base.show(io::IO, fg::FeaturedGraph)
    direct = is_directed(fg) ? "Directed" : "Undirected"
    println(io, "FeaturedGraph(")
    print(io, "\t", direct, " graph with (#V=", nv(fg), ", #E=", ne(fg), ") in ")
    println(io, matrixrepr(fg), ",")
    has_node_feature(fg) && println(io, "\tNode feature:\tℝ^", nf_dims_repr(fg), " <", typeof(fg.nf), ">,")
    has_edge_feature(fg) && println(io, "\tEdge feature:\tℝ^", ef_dims_repr(fg), " <", typeof(fg.ef), ">,")
    has_global_feature(fg) && println(io, "\tGlobal feature:\tℝ^", gf_dims_repr(fg), " <", typeof(fg.gf), ">,")
    print(io, ")")
end

matrixrepr(fg::FeaturedGraph) = matrixrepr(Val(matrixtype(fg)))
matrixrepr(::Val{:adjm}) = "adjacency matrix"
matrixrepr(::Val{:laplacian}) = "Laplacian matrix"
matrixrepr(::Val{:normalized}) = "normalized Laplacian"
matrixrepr(::Val{:scaled}) = "scaled Laplacian"

nf_dims_repr(fg::FeaturedGraph) = size(fg.nf, 1)
ef_dims_repr(fg::FeaturedGraph) = size(fg.ef, 1)
gf_dims_repr(fg::FeaturedGraph) = size(fg.gf, 1)


## Accessing

matrixtype(fg::FeaturedGraph) = fg.matrix_type

GraphSignals.is_directed(fg::FeaturedGraph) = is_directed(graph(fg))

function Base.setproperty!(fg::FeaturedGraph, prop::Symbol, x)
    if prop == :graph
        check_num_node(x, fg.nf)
        check_num_edge(x, fg.ef)
    elseif prop == :nf
        check_num_node(fg.graph, x)
    elseif prop == :ef
        check_num_edge(fg.graph, x)
    end
    setfield!(fg, prop, x)
end

"""
    graph(::AbstractFeaturedGraph)

Get referenced graph.
"""
graph(::NullGraph) = nothing
graph(fg::FeaturedGraph) = fg.graph

"""
    node_feature(::AbstractFeaturedGraph)

Get node feature attached to graph.
"""
node_feature(::NullGraph) = nothing
node_feature(fg::FeaturedGraph) = fg.nf

"""
    edge_feature(::AbstractFeaturedGraph)

Get edge feature attached to graph.
"""
edge_feature(::NullGraph) = nothing
edge_feature(fg::FeaturedGraph) = fg.ef

"""
    global_feature(::AbstractFeaturedGraph)

Get global feature attached to graph.
"""
global_feature(::NullGraph) = nothing
global_feature(fg::FeaturedGraph) = fg.gf

"""
    has_graph(::AbstractFeaturedGraph)

Check if graph is available or not.
"""
has_graph(::NullGraph) = false
has_graph(fg::FeaturedGraph) = fg.graph != Fill(0., (0,0))

"""
    has_node_feature(::AbstractFeaturedGraph)

Check if node feature is available or not.
"""
has_node_feature(::NullGraph) = false
has_node_feature(fg::FeaturedGraph) = !isempty(fg.nf)

"""
    has_edge_feature(::AbstractFeaturedGraph)

Check if edge feature is available or not.
"""
has_edge_feature(::NullGraph) = false
has_edge_feature(fg::FeaturedGraph) = !isempty(fg.ef)

"""
    has_global_feature(::AbstractFeaturedGraph)

Check if global feature is available or not.
"""
has_global_feature(::NullGraph) = false
has_global_feature(fg::FeaturedGraph) = !isempty(fg.gf)


## Graph property

"""
    nv(::AbstractFeaturedGraph)

Get node number of graph.
"""
nv(::NullGraph) = 0
nv(fg::FeaturedGraph) = nv(graph(fg))

"""
    ne(::AbstractFeaturedGraph)

Get edge number of graph.
"""
ne(::NullGraph) = 0
ne(fg::FeaturedGraph) = ne(graph(fg))


## Graph representations

"""
adjacency_list(::AbstractFeaturedGraph)

Get adjacency list of graph.
"""
adjacency_list(::NullGraph) = [zeros(0)]
adjacency_list(fg::FeaturedGraph) = adjacency_list(graph(fg))

adjacency_matrix(fg::FeaturedGraph, T::DataType=eltype(graph(fg))) = adjacency_matrix(graph(fg), T)


## Linear algebra

degrees(fg::FeaturedGraph, T::DataType=eltype(graph(fg)); dir::Symbol=:out) =
    LightGraphs.degrees(graph(fg), T; dir=dir)

degree_matrix(fg::FeaturedGraph, T::DataType=eltype(graph(fg)); dir::Symbol=:out) =
    GraphSignals.degree_matrix(graph(fg), T; dir=dir)

inv_sqrt_degree_matrix(fg::FeaturedGraph, T::DataType=eltype(graph(fg)); dir::Symbol=:out) =
    GraphSignals.inv_sqrt_degree_matrix(graph(fg), T; dir=dir)

laplacian_matrix(fg::FeaturedGraph, T::DataType=eltype(graph(fg)); dir::Symbol=:out) =
    GraphSignals.laplacian_matrix(graph(fg), T; dir=dir)

normalized_laplacian(fg::FeaturedGraph, T::DataType=eltype(graph(fg)); selfloop::Bool=false) =
    GraphSignals.normalized_laplacian(graph(fg), T; selfloop=selfloop)

scaled_laplacian(fg::FeaturedGraph, T::DataType=eltype(graph(fg))) = scaled_laplacian(graph(fg), T)


## Inplace operations

function laplacian_matrix!(fg::FeaturedGraph, T::DataType=eltype(graph(fg)); dir::Symbol=:out)
    if fg.matrix_type == :adjm
        fg.graph.S .= laplacian_matrix(graph(fg), T; dir=dir)
        fg.matrix_type = :laplacian
    end
    fg
end

function normalized_laplacian!(fg::FeaturedGraph, T::DataType=eltype(graph(fg)); selfloop::Bool=false)
    if fg.matrix_type == :adjm
        fg.graph.S .= normalized_laplacian(graph(fg), T; selfloop=selfloop)
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
