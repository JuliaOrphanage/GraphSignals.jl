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
    mask
    matrix_type::Symbol
    directed::Bool

    function FeaturedGraph(graph::T, nf::S, ef::R, gf::Q, mask, mt::Symbol, directed::Bool) where {T,S<:AbstractMatrix,R<:AbstractMatrix,Q<:AbstractVector}
        @assert mt ∈ MATRIX_TYPES "matrix_type must be one of :nonmatrix, :adjm, :laplacian, :normalized or :scaled"
        check_num_edge(ne(graph), ef)
        check_num_node(nv(graph), nf)
        new{T,S,R,Q}(graph, nf, ef, gf, mask, mt, directed)
    end
    function FeaturedGraph{T,S,R,Q}(graph, nf, ef, gf, mask, mt, directed) where {T,S<:AbstractMatrix,R<:AbstractMatrix,Q<:AbstractVector}
        @assert mt ∈ MATRIX_TYPES "matrix_type must be one of :nonmatrix, :adjm, :laplacian, :normalized or :scaled"
        new{T,S,R,Q}(T(graph), S(nf), R(ef), Q(gf), mask, mt, directed)
    end
end

FeaturedGraph() = NullGraph()

## Graph from JuliaGraphs

function FeaturedGraph(graph; directed::Symbol=:auto, T=eltype(graph), N=nv(graph), E=ne(graph),
                       nf=Fill(zero(T), (0, N)), ef=Fill(zero(T), (0, E)), gf=Fill(zero(T), 0))
    @assert directed ∈ DIRECTEDS "directed must be one of :auto, :directed and :undirected"
    dir = (directed == :auto) ? is_directed(graph) : directed == :directed
    mask = Fill(zero(T), (N, N))
    FeaturedGraph(graph, nf, ef, gf, mask, :nonmatrix, dir)
end

## Graph in adjacency list

function FeaturedGraph(graph::AbstractVector{T}; directed::Symbol=:auto, ET=eltype(graph[1]), N=nv(graph), E=ne(graph),
                       nf=Fill(zero(ET), (0, N)), ef=Fill(zero(ET), (0, E)), gf=Fill(zero(ET), 0)) where {T<:AbstractVector}
    @assert directed ∈ DIRECTEDS "directed must be one of :auto, :directed and :undirected"
    dir = (directed == :auto) ? is_directed(graph) : directed == :directed
    mask = Fill(zero(ET), (N, N))
    FeaturedGraph(graph, nf, ef, gf, mask, :nonmatrix, dir)
end

## Graph in adjacency matrix

function FeaturedGraph(graph::AbstractMatrix{T}; directed::Symbol=:auto, N=nv(graph), E=ne(graph),
                       nf=Fill(zero(T), (0, N)), ef=Fill(zero(T), (0, E)), gf=Fill(zero(T), 0)) where {T<:Real}
    @assert directed ∈ DIRECTEDS "directed must be one of :auto, :directed and :undirected"
    dir = (directed == :auto) ? !issymmetric(graph) : directed == :directed
    graph = promote_graph(graph, nf)
    mask = Fill(zero(T), (N, N))
    FeaturedGraph(graph, nf, ef, gf, mask, :adjm, dir)
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

mask(::NullGraph) = nothing
mask(fg::FeaturedGraph) = fg.mask

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
