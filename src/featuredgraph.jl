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
    mask

    function FeaturedGraph(graph::T, nf::S, ef::R, gf::Q, mask) where {T,S<:AbstractMatrix,R<:AbstractMatrix,Q<:AbstractVector}
        new{T,S,R,Q}(graph, nf, ef, gf, mask)
    end
    function FeaturedGraph{T,S,R,Q}(graph, nf, ef, gf, mask) where {T,S<:AbstractMatrix,R<:AbstractMatrix,Q<:AbstractVector}
        new{T,S,R,Q}(T(graph), S(nf), R(ef), Q(gf), mask)
    end
end

FeaturedGraph() = FeaturedGraph(zeros(0,0), zeros(0,0), zeros(0,0), zeros(0), zeros(0,0))

function FeaturedGraph(graph)
    T = eltype(graph)
    N = nv(graph)
    E = ne(graph)

    nf = Fill(zero(T), (0, N))
    ef = Fill(zero(T), (0, E))
    gf = Fill(zero(T), 0)
    mask = Fill(zero(T), (N, N))
    FeaturedGraph(graph, nf, ef, gf, mask)
end

function FeaturedGraph(graph::T) where {T<:AbstractMatrix}
    z = zero(eltype(graph))
    N = nv(graph)
    E = ne(graph)

    nf = Fill(z, (0, N))
    ef = Fill(z, (0, E))
    gf = Fill(z, 0)
    mask = Fill(z, (N, N))
    FeaturedGraph(graph, nf, ef, gf, mask)
end

function FeaturedGraph(graph, nf::S) where {S<:AbstractMatrix}
    z = zero(eltype(nf))
    N = nv(graph)
    E = ne(graph)
    check_num_node(N, nf)

    ef = Fill(z, (0, E))
    gf = Fill(z, 0)
    mask = Fill(z, (N, N))
    FeaturedGraph(graph, nf, ef, gf, mask)
end

function FeaturedGraph(graph::T, nf::S) where {T<:AbstractMatrix,S<:AbstractMatrix}
    z = zero(eltype(nf))
    N = nv(graph)
    E = ne(graph)
    check_num_node(N, nf)

    graph = convert(typeof(nf), graph)
    ef = Fill(z, (0, E))
    gf = Fill(z, 0)
    mask = Fill(z, (N, N))
    FeaturedGraph(graph, nf, ef, gf, mask)
end

function FeaturedGraph(graph::T, nf::S, ef::R, gf::Q) where {T,S<:AbstractMatrix,R<:AbstractMatrix,Q<:AbstractVector}
    ET = eltype(graph)
    N = nv(graph)
    E = ne(graph)
    check_num_node(N, nf)
    check_num_node(E, ef)

    mask = Fill(zero(ET), (N, N))
    FeaturedGraph(graph, nf, ef, gf, mask)
end

check_num_node(g, nf) = @assert nv(g) == size(nf, 2)
check_num_edge(g, ef) = @assert ne(g) == size(ef, 2)

check_num_node(nv::Real, nf) = @assert nv == size(nf, 2)
check_num_edge(ne::Real, ef) = @assert ne == size(ef, 2)

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

has_graph(::NullGraph) = false
has_graph(fg::FeaturedGraph) = fg.graph != zeros(0,0)

has_node_feature(::NullGraph) = false
has_node_feature(fg::FeaturedGraph) = !isempty(fg.nf)

has_edge_feature(::NullGraph) = false
has_edge_feature(fg::FeaturedGraph) = !isempty(fg.ef)

has_global_feature(::NullGraph) = false
has_global_feature(fg::FeaturedGraph) = !isempty(fg.gf)
