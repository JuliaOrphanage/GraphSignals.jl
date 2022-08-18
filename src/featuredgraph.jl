const MATRIX_TYPES = [:adjm, :normedadjm, :laplacian, :normalized, :scaled]
const DIRECTEDS = [:auto, :directed, :undirected]

_string(s::Symbol) = ":$(s)"

abstract type AbstractFeaturedGraph end

"""
    FeaturedGraph(g, [mt]; directed=:auto, nf, ef, gf, pf=nothing,
        T, N, E, with_batch=false)

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
- `mt::Symbol`: Matrix type for `g` in matrix form. if `graph` is in matrix form, `mt` is
    recorded as one of `:adjm`, `:normedadjm`, `:laplacian`, `:normalized` or `:scaled`.
- `directed`: It specify that direction of a graph. It can be `:auto`, `:directed` and
    `:undirected`. Default value is `:auto`, which infers direction automatically.
- `nf`: Node features.
- `ef`: Edge features.
- `gf`: Global features.
- `pf`: Positional features. If `nothing` is given, positional encoding is turned off. If an
    array is given, positional encoding is assigned as given array. If `:auto` is given,
    positional encoding is generated automatically for node features and `with_batch` is considered.
- `T`: It specifies the element type of graph. Default value is the element type of `g`.
- `N`: Number of nodes for `g`.
- `E`: Number of edges for `g`.
- `with_batch::Bool`: Consider last dimension of all features as batch dimension.

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
mutable struct FeaturedGraph{T,Tn<:AbstractGraphSignal,Te<:AbstractGraphSignal,Tg<:AbstractGraphSignal,Tp<:AbstractGraphDomain} <: AbstractFeaturedGraph
    graph::T
    nf::Tn
    ef::Te
    gf::Tg
    pf::Tp
    matrix_type::Symbol

    function FeaturedGraph(graph::SparseGraph, nf, ef, gf, pf, mt::Symbol)
        check_matrix_type(mt)
        check_features(graph, nf, ef, pf)
        nf = NodeSignal(nf)
        ef = EdgeSignal(ef)
        gf = GlobalSignal(gf)
        pf = NodeDomain(pf)
        new{typeof(graph),typeof(nf),typeof(ef),typeof(gf),typeof(pf)}(graph, nf, ef, gf, pf, mt)
    end
    function FeaturedGraph{T,Tn,Te,Tg,Tp}(graph, nf, ef, gf, pf, mt
            ) where {T,Tn,Te,Tg,Tp}
        check_matrix_type(mt)
        check_features(graph, nf, ef, pf)
        graph = T(graph)
        nf = NodeSignal(Tn(nf))
        ef = EdgeSignal(Te(ef))
        gf = GlobalSignal(Tg(gf))
        pf = NodeDomain(Tp(pf))
        new{T,typeof(nf),typeof(ef),typeof(gf),typeof(pf)}(graph, nf, ef, gf, pf, mt)
    end
end

@functor FeaturedGraph

function FeaturedGraph(graph, mat_type::Symbol; directed::Symbol=:auto, T=eltype(graph), N=nv(graph), E=ne(graph),
                       nf=nothing, ef=nothing, gf=nothing, pf=nothing, with_batch::Bool=false)
    @assert directed ∈ DIRECTEDS "directed must be one of $(join(_string.(DIRECTEDS), ", ", " or "))"
    dir = (directed == :auto) ? is_directed(graph) : directed == :directed
    if pf == :auto
        A = nf[1, ntuple(i -> Colon(), length(size(nf))-1)...]
        pf = generate_grid(A, with_batch=with_batch)
    end
    nf = NodeSignal(nf)
    ef = EdgeSignal(ef)
    gf = GlobalSignal(gf)
    pf = NodeDomain(pf)
    return FeaturedGraph(SparseGraph(graph, dir, T), nf, ef, gf, pf, mat_type)
end

## Graph from JuliaGraphs

FeaturedGraph(graph::AbstractGraph; kwargs...) =
    FeaturedGraph(graph, :adjm; T=Float32, kwargs...)

## Graph in adjacency list

FeaturedGraph(graph::AbstractVector{T};
              ET=eltype(graph[1]), kwargs...) where {T<:AbstractVector} =
    FeaturedGraph(graph, :adjm; T=ET, kwargs...)

## Graph in adjacency matrix

FeaturedGraph(graph::AbstractMatrix{T}; N=nv(graph), nf=nothing, kwargs...) where T =
    FeaturedGraph(graph, :adjm; N=N, nf=nf, kwargs...)

function FeaturedGraph(fg::FeaturedGraph;
              nf=node_feature(fg), ef=edge_feature(fg), gf=global_feature(fg),
              pf=positional_feature(fg))
    nf = NodeSignal(nf)
    ef = EdgeSignal(ef)
    gf = GlobalSignal(gf)
    pf = NodeDomain(pf)
    return FeaturedGraph(graph(fg), nf, ef, gf, pf, matrixtype(fg))
end

"""
    ConcreteFeaturedGraph(fg; nf=node_feature(fg), ef=edge_feature(fg),
                          gf=global_feature(fg), pf=positional_feature(fg))

This is a syntax sugar for construction for `FeaturedGraph` and `FeaturedSubgraph` object.
It is an idempotent operation, which gives the same type of object as inputs.
It wraps input `fg` again but reconfigures with `kwargs`.

# Arguments

- `fg`: `FeaturedGraph` and `FeaturedSubgraph` object.
- `nf`: Node features.
- `ef`: Edge features.
- `gf`: Global features.
- `pf`: Positional features.

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
    msg = "number of nodes must match between graph ($graph_nv) and features ($N)"
    graph_nv == N || throw(DimensionMismatch(msg))
end

function check_num_edges(graph_ne::Real, E::Real)
    msg = "number of edges must match between graph ($graph_ne) and features ($E)"
    graph_ne == E || throw(DimensionMismatch(msg))
end

# generic fallback
check_num_nodes(graph_nv::Real, feat) = check_num_nodes(graph_nv, size(feat, 2))
check_num_edges(graph_ne::Real, feat) = check_num_edges(graph_ne, size(feat, 2))

check_num_nodes(g, feat) = check_num_nodes(nv(g), feat)
check_num_edges(g, feat) = check_num_edges(ne(g), feat)

function check_matrix_type(mt::Symbol)
    errmsg = "matrix_type must be one of $(join(_string.(MATRIX_TYPES), ", ", " or "))"
    mt ∈ MATRIX_TYPES || throw(ArgumentError(errmsg))
end

function check_features(graph, nf, ef, pf)
    check_num_edges(ne(graph), ef)
    check_num_nodes(nv(graph), nf)
    check_num_nodes(nv(graph), pf)
    return
end


## show

function Base.show(io::IO, fg::FeaturedGraph)
    direct = is_directed(fg) ? "Directed" : "Undirected"
    println(io, "FeaturedGraph:")
    print(io, "\t", direct, " graph with (#V=", nv(fg), ", #E=", ne(fg), ") in ", matrixrepr(fg))
    has_node_feature(fg) && print(io, "\n\tNode feature:\tℝ^", nf_dims_repr(fg.nf), " <", typeof(fg.nf), ">")
    has_edge_feature(fg) && print(io, "\n\tEdge feature:\tℝ^", ef_dims_repr(fg.ef), " <", typeof(fg.ef), ">")
    has_global_feature(fg) && print(io, "\n\tGlobal feature:\tℝ^", gf_dims_repr(fg.gf), " <", typeof(fg.gf), ">")
    has_positional_feature(fg) && print(io, "\n\tPositional feature:\tℝ^", pf_dims_repr(fg.pf), " <", typeof(fg.pf), ">")
end

matrixrepr(fg::FeaturedGraph) = matrixrepr(Val(matrixtype(fg)))
matrixrepr(::Val{:adjm}) = "adjacency matrix"
matrixrepr(::Val{:normedadjm}) = "normalized adjacency matrix"
matrixrepr(::Val{:laplacian}) = "Laplacian matrix"
matrixrepr(::Val{:normalized}) = "normalized Laplacian"
matrixrepr(::Val{:scaled}) = "scaled Laplacian"


## Accessing

matrixtype(fg::FeaturedGraph) = fg.matrix_type

Graphs.is_directed(fg::AbstractFeaturedGraph) = is_directed(graph(fg))

function Base.setproperty!(fg::FeaturedGraph, prop::Symbol, x)
    if prop == :graph
        check_num_nodes(x, fg.nf)
        check_num_edges(x, fg.ef)
        check_num_nodes(x, fg.pf)
    elseif prop == :nf
        x = NodeSignal(x)
        check_num_nodes(fg.graph, x)
    elseif prop == :ef
        x = EdgeSignal(x)
        check_num_edges(fg.graph, x)
    elseif prop == :gf
        x = GlobalSignal(x)
    elseif prop == :pf
        x = NodeDomain(x)
        check_num_nodes(fg.graph, x)
    end
    setfield!(fg, prop, x)
end

"""
    graph(fg)

Get referenced graph in `fg`.

# Arguments

- `fg::AbstractFeaturedGraph`: A concrete object of `AbstractFeaturedGraph` type.
"""
graph(fg::FeaturedGraph) = fg.graph

"""
    node_feature(fg)

Get node feature attached to `fg`.

# Arguments

- `fg::AbstractFeaturedGraph`: A concrete object of `AbstractFeaturedGraph` type.
"""
node_feature(fg::FeaturedGraph) = node_feature(fg.nf)

"""
    edge_feature(fg)

Get edge feature attached to `fg`.

# Arguments

- `fg::AbstractFeaturedGraph`: A concrete object of `AbstractFeaturedGraph` type.
"""
edge_feature(fg::FeaturedGraph) = edge_feature(fg.ef)

"""
    global_feature(fg)

Get global feature attached to `fg`.

# Arguments

- `fg::AbstractFeaturedGraph`: A concrete object of `AbstractFeaturedGraph` type.
"""
global_feature(fg::FeaturedGraph) = global_feature(fg.gf)

"""
    positional_feature(fg)

Get positional feature attached to `fg`.

# Arguments

- `fg::AbstractFeaturedGraph`: A concrete object of `AbstractFeaturedGraph` type.
"""
positional_feature(fg::FeaturedGraph) = positional_feature(fg.pf)

"""
    has_graph(fg)

Check if `graph` is available or not for `fg`.

# Arguments

- `fg::AbstractFeaturedGraph`: A concrete object of `AbstractFeaturedGraph` type.
"""
has_graph(fg::FeaturedGraph) = fg.graph != Fill(0., (0,0))

"""
    has_node_feature(::AbstractFeaturedGraph)

Check if `node_feature` is available or not for `fg`.

# Arguments

- `fg::AbstractFeaturedGraph`: A concrete object of `AbstractFeaturedGraph` type.
"""
has_node_feature(fg::FeaturedGraph) = has_node_feature(fg.nf)

"""
    has_edge_feature(::AbstractFeaturedGraph)

Check if `edge_feature` is available or not for `fg`.

# Arguments

- `fg::AbstractFeaturedGraph`: A concrete object of `AbstractFeaturedGraph` type.
"""
has_edge_feature(fg::FeaturedGraph) = has_edge_feature(fg.ef)

"""
    has_global_feature(::AbstractFeaturedGraph)

Check if `global_feature` is available or not for `fg`.

# Arguments

- `fg::AbstractFeaturedGraph`: A concrete object of `AbstractFeaturedGraph` type.
"""
has_global_feature(fg::FeaturedGraph) = has_global_feature(fg.gf)

"""
    has_positional_feature(::AbstractFeaturedGraph)

Check if `positional_feature` is available or not for `fg`.

# Arguments

- `fg::AbstractFeaturedGraph`: A concrete object of `AbstractFeaturedGraph` type.
"""
has_positional_feature(fg::FeaturedGraph) = has_positional_feature(fg.pf)


## Graph property

"""
    nv(fg)

Get node number of graph in `fg`.

# Arguments

- `fg::AbstractFeaturedGraph`: A concrete object of `AbstractFeaturedGraph` type.
"""
Graphs.nv(fg::FeaturedGraph) = nv(graph(fg))

"""
    ne(fg)

Get edge number of in `fg`.

# Arguments

- `fg::AbstractFeaturedGraph`: A concrete object of `AbstractFeaturedGraph` type.
"""
Graphs.ne(fg::FeaturedGraph) = ne(graph(fg))

to_namedtuple(fg::AbstractFeaturedGraph) = to_namedtuple(graph(fg))

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
adjacency_list(fg::FeaturedGraph) = adjacency_list(graph(fg))

adjacency_matrix(fg::FeaturedGraph, ::Type{T}=eltype(graph(fg))) where {T} =
    adjacency_matrix(graph(fg), T)

degrees(fg::AbstractFeaturedGraph, ::Type{T}=eltype(graph(fg)); dir::Symbol=:out) where {T} =
    degrees(graph(fg), T; dir=dir)


## sample

StatsBase.sample(fg::AbstractFeaturedGraph, n::Int) =
    subgraph(fg, sample(vertices(fg), n; replace=false))
