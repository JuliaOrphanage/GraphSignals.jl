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

    function FeaturedGraph(graph::T, nf::S, ef::R, gf::Q) where {T,S<:AbstractMatrix,R<:AbstractMatrix,Q<:AbstractVector}
        new{T,S,R,Q}(graph, nf, ef, gf)
    end
    function FeaturedGraph{T,S,R,Q}(graph, nf, ef, gf) where {T,S<:AbstractMatrix,R<:AbstractMatrix,Q<:AbstractVector}
        new{T,S,R,Q}(T(graph), S(nf), R(ef), Q(gf))
    end
end

FeaturedGraph() = FeaturedGraph(zeros(0,0), zeros(0,0), zeros(0,0), zeros(0))

FeaturedGraph(graph) = FeaturedGraph(graph, zeros(0,0), zeros(0,0), zeros(0))

function FeaturedGraph(graph::T) where {T<:AbstractMatrix}
    z = zero(eltype(graph))
    nf = similar(graph,0,0).*z
    ef = similar(graph,0,0).*z
    gf = similar(graph,0).*z
    FeaturedGraph(graph, nf, ef, gf)
end

function FeaturedGraph(graph, nf::S) where {S<:AbstractMatrix}
    z = zero(eltype(nf))
    ef = similar(nf,0,0).*z
    gf = similar(nf,0).*z
    FeaturedGraph(graph, nf, ef, gf)
end

function FeaturedGraph(graph::T, nf::S) where {T<:AbstractMatrix,S<:AbstractMatrix}
    z = zero(eltype(nf))
    graph = convert(typeof(nf), graph)
    ef = similar(nf,0,0).*z
    gf = similar(nf,0).*z
    FeaturedGraph(graph, nf, ef, gf)
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

has_graph(::NullGraph) = false
has_graph(fg::FeaturedGraph) = fg.graph != zeros(0,0)

has_node_feature(::NullGraph) = false
has_node_feature(fg::FeaturedGraph) = fg.nf != zeros(0,0)

has_edge_feature(::NullGraph) = false
has_edge_feature(fg::FeaturedGraph) = fg.ef != zeros(0,0)

has_global_feature(::NullGraph) = false
has_global_feature(fg::FeaturedGraph) = fg.gf != zeros(0)
