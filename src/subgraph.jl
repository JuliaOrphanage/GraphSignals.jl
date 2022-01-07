struct FeaturedSubgraph{G<:AbstractFeaturedGraph,T} <: AbstractFeaturedGraph
    fg::G
    nodes::T
end

@functor FeaturedSubgraph

FeaturedSubgraph(ng::NullGraph, ::AbstractVector) = ng

subgraph(fg::AbstractFeaturedGraph, nodes::AbstractVector) = FeaturedSubgraph(fg, nodes)
subgraph(fsg::FeaturedSubgraph, nodes::AbstractVector) = FeaturedSubgraph(fsg.fg, nodes)

StatsBase.sample(fsg::FeaturedSubgraph, n::Int) =
    subgraph(fsg, sample(fsg.nodes, n; replace=false))


## show

function Base.show(io::IO, fsg::FeaturedSubgraph)
    print(io, "FeaturedSubgraph of ", fsg.fg, ",$(fsg.nodes))")
end

graph(fsg::FeaturedSubgraph) = graph(fsg.fg)

Base.parent(fsg::FeaturedSubgraph) = fsg.fg

Graphs.vertices(fsg::FeaturedSubgraph) = fsg.nodes

function Graphs.edges(fsg::FeaturedSubgraph)
    sg = graph(fsg.fg)
    sel = map(x -> x in fsg.nodes, colvals(sg.S))
    sel .&= map(x -> x in fsg.nodes, rowvals(sg.S))
    return sort!(unique!(edgevals(sg)[sel]))
end

Graphs.adjacency_matrix(fsg::FeaturedSubgraph) = view(adjacency_matrix(fsg.fg), fsg.nodes, fsg.nodes)

node_feature(fsg::FeaturedSubgraph) = view(node_feature(fsg.fg), :, fsg.nodes)

edge_feature(fsg::FeaturedSubgraph) = view(edge_feature(fsg.fg), :, edges(fsg))

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
