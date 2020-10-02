const MATRIX_TYPES = [:nonmatrix, :adjm, :laplacian, :normalized, :scaled]

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
    matrix_type::Symbol

    function FeaturedGraph(graph::T, nf::S, ef::R, gf::Q, mask, mt::Symbol) where {T,S<:AbstractMatrix,R<:AbstractMatrix,Q<:AbstractVector}
        @assert mt ∈ MATRIX_TYPES "matrix_type must be one of :nonmatrix, :adjm, :laplacian, :normalized or :scaled"
        new{T,S,R,Q}(graph, nf, ef, gf, mask, mt)
    end
    function FeaturedGraph{T,S,R,Q}(graph, nf, ef, gf, mask, mt) where {T,S<:AbstractMatrix,R<:AbstractMatrix,Q<:AbstractVector}
        @assert mt ∈ MATRIX_TYPES "matrix_type must be one of :nonmatrix, :adjm, :laplacian, :normalized or :scaled"
        new{T,S,R,Q}(T(graph), S(nf), R(ef), Q(gf), mask, mt)
    end
end

FeaturedGraph() = NullGraph()

function FeaturedGraph(graph)
    T = eltype(graph)
    N = nv(graph)
    E = ne(graph)

    nf = Fill(zero(T), (0, N))
    ef = Fill(zero(T), (0, E))
    gf = Fill(zero(T), 0)
    mask = Fill(zero(T), (N, N))
    FeaturedGraph(graph, nf, ef, gf, mask, :nonmatrix)
end

function FeaturedGraph(graph::T) where {T<:AbstractMatrix}
    z = zero(eltype(graph))
    N = nv(graph)
    E = ne(graph)

    nf = Fill(z, (0, N))
    ef = Fill(z, (0, E))
    gf = Fill(z, 0)
    mask = Fill(z, (N, N))
    FeaturedGraph(graph, nf, ef, gf, mask, :adjm)
end

function FeaturedGraph(graph, nf::S) where {S<:AbstractMatrix}
    z = zero(eltype(nf))
    N = nv(graph)
    E = ne(graph)
    check_num_node(N, nf)

    ef = Fill(z, (0, E))
    gf = Fill(z, 0)
    mask = Fill(z, (N, N))
    FeaturedGraph(graph, nf, ef, gf, mask, :nonmatrix)
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
    FeaturedGraph(graph, nf, ef, gf, mask, :adjm)
end

function FeaturedGraph(graph, nf::S, ef::R, gf::Q) where {S<:AbstractMatrix,R<:AbstractMatrix,Q<:AbstractVector}
    ET = eltype(graph)
    N = nv(graph)
    E = ne(graph)
    check_num_node(N, nf)
    check_num_node(E, ef)

    mask = Fill(zero(ET), (N, N))
    FeaturedGraph(graph, nf, ef, gf, mask, :nonmatrix)
end

function FeaturedGraph(graph::AbstractMatrix{T}, nf::S, ef::R, gf::Q) where {T<:Real,S<:AbstractMatrix,R<:AbstractMatrix,Q<:AbstractVector}
    N = nv(graph)
    E = ne(graph)
    check_num_node(N, nf)
    check_num_node(E, ef)

    mask = Fill(zero(T), (N, N))
    FeaturedGraph(graph, nf, ef, gf, mask, :adjm)
end

function check_num_node(nv::Real, nf)
    N = size(nf, 2)
    if nv != N
        throw(DimensionMismatch("number of nodes must match between graph ($nv) and node features ($N)"))
    end
end

function check_num_edge(ne::Real, ef)
    E = size(ef, 2)
    if ne != E
        throw(DimensionMismatch("number of nodes must match between graph ($ne) and edge features ($E)"))
    end
end

check_num_node(g, nf) = check_num_node(nv(g), nf)
check_num_edge(g, ef) = check_num_edge(ne(g), ef)

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

mask(::NullGraph) = nothing
mask(fg::FeaturedGraph) = fg.mask

has_graph(::NullGraph) = false
has_graph(fg::FeaturedGraph) = fg.graph != Fill(0., (0,0))

has_node_feature(::NullGraph) = false
has_node_feature(fg::FeaturedGraph) = !isempty(fg.nf)

has_edge_feature(::NullGraph) = false
has_edge_feature(fg::FeaturedGraph) = !isempty(fg.ef)

has_global_feature(::NullGraph) = false
has_global_feature(fg::FeaturedGraph) = !isempty(fg.gf)
