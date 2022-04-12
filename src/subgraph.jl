struct FeaturedSubgraph{G<:AbstractFeaturedGraph,T} <: AbstractFeaturedGraph
    fg::G
    nodes::T
end

@functor FeaturedSubgraph

FeaturedSubgraph(ng::NullGraph, ::AbstractVector) = ng

function ConcreteFeaturedGraph(fsg::FeaturedSubgraph; nf=node_feature(fsg.fg),
                               ef=edge_feature(fsg.fg), gf=global_feature(fsg.fg),
                               subgraph=fsg.nodes)
    fg = FeaturedGraph(graph(fsg.fg), nf, ef, gf, matrixtype(fsg.fg))
    return FeaturedSubgraph(fg, subgraph)
end

"""
    subgraph(fg, nodes)

Returns a subgraph of type `FeaturedSubgraph` from a given featured graph `fg`.
It constructs a subgraph by reserving `nodes` in a graph.

# Arguments

- `fg::AbstractFeaturedGraph`: A base featured graph to construct a subgraph.
- `nodes::AbstractVector`: It specifies nodes to be reserved from `fg`.
"""
subgraph(fg::AbstractFeaturedGraph, nodes::AbstractVector) = FeaturedSubgraph(fg, nodes)
subgraph(fsg::FeaturedSubgraph, nodes::AbstractVector) = FeaturedSubgraph(fsg.fg, nodes)


## show

function Base.show(io::IO, fsg::FeaturedSubgraph)
    print(io, "FeaturedSubgraph of ", fsg.fg, ",$(fsg.nodes))")
end

graph(fsg::FeaturedSubgraph) = graph(fsg.fg)

Base.parent(fsg::FeaturedSubgraph) = fsg.fg

Graphs.vertices(fsg::FeaturedSubgraph) = fsg.nodes

function Graphs.edges(fsg::FeaturedSubgraph)
    sg = graph(fsg.fg)
    S = SparseMatrixCSC(sparse(sg))
    nodes = collect(fsg.nodes)
    sel = map(x -> x in nodes, colvals(S))
    sel .&= map(x -> x in nodes, rowvals(S))
    return sort!(unique!(collect(edgevals(sg))[sel]))
end

Graphs.adjacency_matrix(fsg::FeaturedSubgraph) = view(adjacency_matrix(fsg.fg), fsg.nodes, fsg.nodes)

node_feature(fsg::FeaturedSubgraph) = node_feature(fsg.fg)

edge_feature(fsg::FeaturedSubgraph) = edge_feature(fsg.fg)

global_feature(fsg::FeaturedSubgraph) = global_feature(fsg.fg)

Graphs.is_directed(fsg::FeaturedSubgraph) = is_directed(fsg.fg)

Graphs.neighbors(fsg::FeaturedSubgraph) = mapreduce(i -> neighbors(graph(fsg), i), vcat, fsg.nodes)

incident_edges(fsg::FeaturedSubgraph) = mapreduce(i -> incident_edges(graph(fsg), i), vcat, fsg.nodes)

repeat_nodes(fsg::FeaturedSubgraph) = mapreduce(i -> repeat_nodes(graph(fsg), i), vcat, fsg.nodes)


## Linear algebra

degrees(fsg::FeaturedSubgraph, T::DataType=eltype(graph(fsg.fg)); dir::Symbol=:out) =
    degrees(fsg.fg, T; dir=dir)[fsg.nodes]

degree_matrix(fsg::FeaturedSubgraph, T::DataType=eltype(graph(fsg.fg)); dir::Symbol=:out) =
    degree_matrix(fsg.fg, T; dir=dir)[fsg.nodes, fsg.nodes]

normalized_adjacency_matrix(fsg::FeaturedSubgraph, T::DataType=eltype(graph(fsg.fg)); selfloop::Bool=false) =
    normalized_adjacency_matrix(fsg.fg, T; selfloop=selfloop)[fsg.nodes, fsg.nodes]

laplacian_matrix(fsg::FeaturedSubgraph, T::DataType=eltype(graph(fsg.fg)); dir::Symbol=:out) =
    laplacian_matrix(fsg.fg, T; dir=dir)[fsg.nodes, fsg.nodes]

normalized_laplacian(fsg::FeaturedSubgraph, T::DataType=eltype(graph(fsg.fg));
                                     dir::Symbol=:both, selfloop::Bool=false) =
    normalized_laplacian(fsg.fg, T; selfloop=selfloop)[fsg.nodes, fsg.nodes]

scaled_laplacian(fsg::FeaturedSubgraph, T::DataType=eltype(graph(fsg.fg))) =
    scaled_laplacian(fsg.fg, T)[fsg.nodes, fsg.nodes]

"""
    mask(fg, m)

A syntax sugar for masking graph.

Returns a `FeaturedSubgraph`.
"""
mask(fg::AbstractFeaturedGraph, m::AbstractVector) = subgraph(fg, m)
