abstract type AbstractFeaturedGraph end

"""
    NullGraph()

Null object for `FeaturedGraph`.
"""
struct NullGraph <: AbstractFeaturedGraph end

"""
    FeaturedGraph(graph, node_feature, edge_feature, global_feature)

A feature-equipped graph structure for passing graph to layer in order to provide graph dynamically.
References to graph or features are hold in this type.

# Arguments
- `graph`: should be a adjacency matrix, `SimpleGraph`, `SimpleDiGraph` (from LightGraphs) or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `node_feature`: node features attached to graph.
- `edge_feature`: edge features attached to graph.
- `gloabl_feature`: gloabl graph features attached to graph.
"""
mutable struct FeaturedGraph{T,S<:AbstractMatrix,R<:AbstractMatrix,Q<:AbstractVector} <: AbstractFeaturedGraph
    graph::T
    nf::S
    ef::R
    gf::Q
end

FeaturedGraph() = FeaturedGraph(zeros(0,0), zeros(0,0), zeros(0,0), zeros(0))

FeaturedGraph(graph::T) where {T} = FeaturedGraph(graph, zeros(0,0), zeros(0,0), zeros(0))

FeaturedGraph(graph::T, nf::AbstractMatrix) where {T} = FeaturedGraph(graph, nf, zeros(0,0), zeros(0))

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

has_graph(::NullGraph) = false
has_graph(fg::FeaturedGraph) = fg.graph != zeros(0,0)

has_node_feature(::NullGraph) = false
has_node_feature(fg::FeaturedGraph) = fg.nf != zeros(0,0)

has_edge_feature(::NullGraph) = false
has_edge_feature(fg::FeaturedGraph) = fg.ef != zeros(0,0)

has_global_feature(::NullGraph) = false
has_global_feature(fg::FeaturedGraph) = fg.gf != zeros(0)
