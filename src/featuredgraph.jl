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
        new{T,S,R,Q}(graph, nf, ef, gf, mask, mt, directed)
    end
    function FeaturedGraph{T,S,R,Q}(graph, nf, ef, gf, mask, mt, directed) where {T,S<:AbstractMatrix,R<:AbstractMatrix,Q<:AbstractVector}
        @assert mt ∈ MATRIX_TYPES "matrix_type must be one of :nonmatrix, :adjm, :laplacian, :normalized or :scaled"
        new{T,S,R,Q}(T(graph), S(nf), R(ef), Q(gf), mask, mt, directed)
    end
end

FeaturedGraph() = NullGraph()

function FeaturedGraph(graph; directed::Symbol=:auto)
    @assert directed ∈ DIRECTEDS "directed must be one of :auto, :directed and :undirected"
    dir = (directed == :auto) ? is_directed(graph) : directed == :directed
    T = eltype(graph)
    N = nv(graph)
    E = ne(graph)

    nf = Fill(zero(T), (0, N))
    ef = Fill(zero(T), (0, E))
    gf = Fill(zero(T), 0)
    mask = Fill(zero(T), (N, N))
    FeaturedGraph(graph, nf, ef, gf, mask, :nonmatrix, dir)
end

function FeaturedGraph(graph::AbstractVector{T}; directed::Symbol=:auto) where {T<:AbstractVector}
    @assert directed ∈ DIRECTEDS "directed must be one of :auto, :directed and :undirected"
    dir = (directed == :auto) ? is_directed(graph) : directed == :directed
    ET = eltype(graph[1])
    N = nv(graph)
    E = ne(graph, dir)

    nf = Fill(zero(ET), (0, N))
    ef = Fill(zero(ET), (0, E))
    gf = Fill(zero(ET), 0)
    mask = Fill(zero(ET), (N, N))
    FeaturedGraph(graph, nf, ef, gf, mask, :nonmatrix, dir)
end

## Graph in adjacency matrix

function FeaturedGraph(graph::T; directed::Symbol=:auto, N=nv(graph), E=ne(graph)) where {T<:AbstractMatrix}
    z = zero(eltype(graph))
    nf = Fill(z, (0, N))
    ef = Fill(z, (0, E))
    gf = Fill(z, 0)
    FeaturedGraph(graph, nf, ef, gf; directed=directed, N=N, E=E)
end

function FeaturedGraph(graph::T, nf::S; directed::Symbol=:auto, N=nv(graph), E=ne(graph)) where {T<:AbstractMatrix,S<:AbstractMatrix}
    z = zero(eltype(nf))
    graph = convert(S, graph)
    ef = Fill(z, (0, E))
    gf = Fill(z, 0)
    FeaturedGraph(graph, nf, ef, gf; directed=directed, N=N, E=E)
end

FeaturedGraph(graph::T, nf::Transpose{S,R}, args...; kwargs...) where {T<:AbstractMatrix,S,R<:AbstractMatrix} =
    FeaturedGraph(graph, R(nf), args...; kwargs...)

function FeaturedGraph(graph::AbstractMatrix{T}, nf::S, ef::R, gf::Q; directed::Symbol=:auto,
                       N=nv(graph), E=ne(graph)) where {T<:Real,S<:AbstractMatrix,R<:AbstractMatrix,Q<:AbstractVector}
    @assert directed ∈ DIRECTEDS "directed must be one of :auto, :directed and :undirected"
    dir = (directed == :auto) ? !issymmetric(graph) : directed == :directed
    # check_num_node(N, nf)
    # check_num_edge(E, ef)
    mask = Fill(zero(T), (N, N))
    FeaturedGraph(graph, nf, ef, gf, mask, :adjm, dir)
end

function FeaturedGraph(graph, nf::S; directed::Symbol=:auto) where {S<:AbstractMatrix}
    @assert directed ∈ DIRECTEDS "directed must be one of :auto, :directed and :undirected"
    dir = (directed == :auto) ? is_directed(graph) : directed == :directed
    z = zero(eltype(nf))
    N = nv(graph)
    E = ne(graph)
    # check_num_node(N, nf)

    ef = Fill(z, (0, E))
    gf = Fill(z, 0)
    mask = Fill(z, (N, N))
    FeaturedGraph(graph, nf, ef, gf, mask, :nonmatrix, dir)
end

function FeaturedGraph(graph::AbstractVector{T}, nf::S; directed::Symbol=:auto) where {T<:AbstractVector,S<:AbstractMatrix}
    @assert directed ∈ DIRECTEDS "directed must be one of :auto, :directed and :undirected"
    dir = (directed == :auto) ? is_directed(graph) : directed == :directed
    z = zero(eltype(nf))
    N = nv(graph)
    E = ne(graph, dir)
    # check_num_node(N, nf)

    ef = Fill(z, (0, E))
    gf = Fill(z, 0)
    mask = Fill(z, (N, N))
    FeaturedGraph(graph, nf, ef, gf, mask, :nonmatrix, dir)
end

function FeaturedGraph(graph, nf::S, ef::R, gf::Q; directed::Symbol=:auto) where {S<:AbstractMatrix,R<:AbstractMatrix,Q<:AbstractVector}
    @assert directed ∈ DIRECTEDS "directed must be one of :auto, :directed and :undirected"
    dir = (directed == :auto) ? is_directed(graph) : directed == :directed
    ET = eltype(graph)
    N = nv(graph)
    E = ne(graph)
    # check_num_node(N, nf)
    # check_num_edge(E, ef)

    mask = Fill(zero(ET), (N, N))
    FeaturedGraph(graph, nf, ef, gf, mask, :nonmatrix, dir)
end

function FeaturedGraph(graph::AbstractVector{T}, nf::S, ef::R, gf::Q; directed::Symbol=:auto) where {T<:AbstractVector,S<:AbstractMatrix,R<:AbstractMatrix,Q<:AbstractVector}
    @assert directed ∈ DIRECTEDS "directed must be one of :auto, :directed and :undirected"
    dir = (directed == :auto) ? is_directed(graph) : directed == :directed
    ET = eltype(graph[1])
    N = nv(graph)
    E = ne(graph, dir)
    # check_num_node(N, nf)
    # check_num_edge(E, ef)

    mask = Fill(zero(ET), (N, N))
    FeaturedGraph(graph, nf, ef, gf, mask, :nonmatrix, dir)
end

function check_num_node(graph_nv::Real, nf)
    N = size(nf, 2)
    if graph_nv != N
        throw(DimensionMismatch("number of nodes must match between graph ($graph_nv) and node features ($N)"))
    end
end

function check_num_edge(graph_ne::Real, ef)
    E = size(ef, 2)
    if graph_ne != E
        throw(DimensionMismatch("number of edges must match between graph ($graph_ne) and edge features ($E)"))
    end
end

check_num_node(g, nf) = check_num_node(nv(g), nf)
check_num_edge(g, ef) = check_num_edge(ne(g), ef)
function check_num_edge(g::AbstractMatrix, ef)
    graph_ne = ne(g)
    E = size(ef, 2)
    if issymmetric(g)
        if 2*graph_ne != E
            throw(DimensionMismatch("number of edges must match between graph ($graph_ne) and edge features ($E)"))
        end
    else
        if graph_ne != E
            throw(DimensionMismatch("number of edges must match between graph ($graph_ne) and edge features ($E)"))
        end
    end
end

# function Base.setproperty!(fg::FeaturedGraph, prop::Symbol, x)
#     if prop == :graph
#         check_num_node(x, fg.nf)
#         check_num_edge(x, fg.ef)
#     elseif prop == :nf
#         check_num_node(fg.graph, x)
#     elseif prop == :ef
#         check_num_edge(fg.graph, x)
#     end
#     setfield!(fg, prop, x)
# end

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
