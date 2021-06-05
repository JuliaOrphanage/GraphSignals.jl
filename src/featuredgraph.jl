const MATRIX_TYPES = [:nonmatrix, :adjm, :laplacian, :normalized, :scaled]
const DIRECTEDS = [:auto, :directed, :undirected]

abstract type AbstractFeaturedGraph end

"""
    NullGraph()

Null object for `FeaturedGraph`.
"""
struct NullGraph <: AbstractFeaturedGraph end

"""
    FeaturedGraph(graph, node_feature, edge_feature, global_feature, mt, directed)

A feature-equipped graph structure for passing graph to layer in order to provide graph dynamically.
References to graph or features are hold in this type.

# Arguments
- `graph`: should be a adjacency matrix, `SimpleGraph`, `SimpleDiGraph` (from LightGraphs) or `SimpleWeightedGraph`,
`SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `node_feature`: node features attached to graph.
- `edge_feature`: edge features attached to graph.
- `gloabl_feature`: gloabl graph features attached to graph.
- `mask`: mask for `graph`.
- `mt`: matrix type for `graph` in matrix form. if `graph` is in matrix form, `mt` is recorded as one of `:adjm`,
`:laplacian`, `:normalized` or `:scaled`. Otherwise, `:nonmatrix` is recorded.
- `directed`: the direction of `graph`. it is `true` for directed graph; it is `false` for undirected graph.
"""
mutable struct FeaturedGraph{T,S<:AbstractMatrix,R<:AbstractMatrix,Q<:AbstractVector} <: AbstractFeaturedGraph
    graph::T
    nf::S
    ef::R
    gf::Q
    matrix_type::Symbol
    directed::Bool

    function FeaturedGraph(graph::T, nf::S, ef::R, gf::Q, 
            mt::Symbol, directed::Bool) where {T,S<:AbstractMatrix,R<:AbstractMatrix,Q<:AbstractVector}
        check_precondition(graph, nf, ef, mt)
        new{T,S,R,Q}(graph, nf, ef, gf, mt, directed)
    end
    function FeaturedGraph{T,S,R,Q}(graph, nf, ef, gf,
            mt, directed) where {T,S<:AbstractMatrix,R<:AbstractMatrix,Q<:AbstractVector}
        check_precondition(graph, nf, ef, mt)
        new{T,S,R,Q}(T(graph), S(nf), R(ef), Q(gf), mt, directed)
    end
end

FeaturedGraph() = NullGraph()

function FeaturedGraph(graph, mat_type::Symbol; directed::Symbol=:auto, T=eltype(graph), N=nv(graph), E=ne(graph),
                       nf=Fill(zero(T), (0, N)), ef=Fill(zero(T), (0, E)), gf=Fill(zero(T), 0))
    @assert directed ∈ DIRECTEDS "directed must be one of :auto, :directed and :undirected"
    dir = (directed == :auto) ? GraphSignals.is_directed(graph) : directed == :directed
    FeaturedGraph(graph, nf, ef, gf, mat_type, dir)
end

## Graph from JuliaGraphs

FeaturedGraph(graph::AbstractGraph; kwargs...) = FeaturedGraph(graph, :nonmatrix; kwargs...)

## Graph in adjacency list

function FeaturedGraph(graph::AbstractVector{T}; ET=eltype(graph[1]), kwargs...) where {T<:AbstractVector}
    FeaturedGraph(graph, :nonmatrix; T=ET, kwargs...)
end

## Graph in adjacency matrix

function FeaturedGraph(graph::AbstractMatrix{T}; N=nv(graph), nf=Fill(zero(T), (0, N)), kwargs...) where {T<:Real}
    graph = promote_graph(graph, nf)
    FeaturedGraph(graph, :adjm; N=N, nf=nf, kwargs...)
end

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

function check_precondition(graph, nf, ef, mt)
    @assert mt ∈ MATRIX_TYPES "matrix_type must be one of :nonmatrix, :adjm, :laplacian, :normalized or :scaled"
    check_num_edge(ne(graph), ef)
    check_num_node(nv(graph), nf)
    return
end

function Base.show(io::IO, fg::FeaturedGraph)
    direct = fg.directed ? "Directed" : "Undirected"
    println(io, "FeaturedGraph(")
    print(io, "\t", direct, " graph with (#V=", nv(fg), ", #E=", ne(fg), ") in ")
    println(io, graphrepr(fg.graph), " <", typeof(fg.graph), ">,")
    has_node_feature(fg) && println(io, "\tNode feature:\tℝ^", nf_dims_repr(fg), " <", typeof(fg.nf), ">,")
    has_edge_feature(fg) && println(io, "\tEdge feature:\tℝ^", ef_dims_repr(fg), " <", typeof(fg.ef), ">,")
    has_global_feature(fg) && println(io, "\tGlobal feature:\tℝ^", gf_dims_repr(fg), " <", typeof(fg.gf), ">,")
    print(io, ")")
end

graphrepr(g::AbstractMatrix) = "adjacency matrix"
graphrepr(g::AbstractVector{<:AbstractVector}) = "adjacency list"
graphrepr(g::T) where {T<:AbstractGraph} = string(T)

nf_dims_repr(fg::FeaturedGraph) = size(fg.nf, 1)
ef_dims_repr(fg::FeaturedGraph) = size(fg.ef, 1)
gf_dims_repr(fg::FeaturedGraph) = size(fg.gf, 1)


## Accessing

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
Zygote.@nograd has_graph

"""
    has_node_feature(::AbstractFeaturedGraph)

Check if node feature is available or not.
"""
has_node_feature(::NullGraph) = false
has_node_feature(fg::FeaturedGraph) = !isempty(fg.nf)
Zygote.@nograd has_node_feature

"""
    has_edge_feature(::AbstractFeaturedGraph)

Check if edge feature is available or not.
"""
has_edge_feature(::NullGraph) = false
has_edge_feature(fg::FeaturedGraph) = !isempty(fg.ef)
Zygote.@nograd has_edge_feature

"""
    has_global_feature(::AbstractFeaturedGraph)

Check if global feature is available or not.
"""
has_global_feature(::NullGraph) = false
has_global_feature(fg::FeaturedGraph) = !isempty(fg.gf)
Zygote.@nograd has_global_feature

"""
    fetch_graph(g1, g2)

Fetch graph from `g1` or `g2`. If there is only one graph available, fetch that one.
Otherwise, fetch the first one.
"""
fetch_graph(::NullGraph, fg::FeaturedGraph) = graph(fg)
fetch_graph(fg::FeaturedGraph, ::NullGraph) = graph(fg)
fetch_graph(fg1::FeaturedGraph, fg2::FeaturedGraph) = has_graph(fg1) ? graph(fg1) : graph(fg2)


## Graph property

"""
    nv(::AbstractFeaturedGraph)

Get node number of graph.
"""
nv(::NullGraph) = 0
nv(fg::FeaturedGraph) = nv(graph(fg))
nv(fg::FeaturedGraph{T}) where {T<:AbstractMatrix} = size(graph(fg), 1)

"""
    ne(::AbstractFeaturedGraph)

Get edge number of graph.
"""
ne(::NullGraph) = 0
ne(fg::FeaturedGraph) = ne(graph(fg))
ne(fg::FeaturedGraph{T}) where {T<:AbstractMatrix} = ne(graph(fg))
ne(fg::FeaturedGraph{T}) where {T<:AbstractVector} = ne(graph(fg), fg.directed)

is_directed(fg::FeaturedGraph) = fg.directed


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
    degrees(graph(fg), T; dir=dir)

degree_matrix(fg::FeaturedGraph, T::DataType=eltype(graph(fg)); dir::Symbol=:out) =
    degree_matrix(graph(fg), T; dir=dir)

inv_sqrt_degree_matrix(fg::FeaturedGraph, T::DataType=eltype(graph(fg)); dir::Symbol=:out) =
    inv_sqrt_degree_matrix(graph(fg), T; dir=dir)

laplacian_matrix(fg::FeaturedGraph, T::DataType=eltype(graph(fg)); dir::Symbol=:out) =
    laplacian_matrix(graph(fg), T; dir=dir)

normalized_laplacian(fg::FeaturedGraph, T::DataType=eltype(graph(fg)); selfloop::Bool=false) =
    normalized_laplacian(graph(fg), T; selfloop=selfloop)

scaled_laplacian(fg::FeaturedGraph, T::DataType=eltype(graph(fg))) = scaled_laplacian(graph(fg), T)


## Inplace operations

function laplacian_matrix!(fg::FeaturedGraph, T::DataType=eltype(graph(fg)); dir::Symbol=:out)
    if fg.matrix_type == :adjm
        fg.graph .= laplacian_matrix(graph(fg), T; dir=dir)
        fg.matrix_type = :laplacian
    end
    fg
end

function normalized_laplacian!(fg::FeaturedGraph, T::DataType=eltype(graph(fg)); selfloop::Bool=false)
    if fg.matrix_type == :adjm
        fg.graph .= normalized_laplacian(graph(fg), T; selfloop=selfloop)
        fg.matrix_type = :normalized
    end
    fg
end

function scaled_laplacian!(fg::FeaturedGraph, T::DataType=eltype(graph(fg)))
    if fg.matrix_type == :adjm
        fg.graph .= scaled_laplacian(graph(fg), T)
        fg.matrix_type = :scaled
    end
    fg
end
