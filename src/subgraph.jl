struct FeaturedSubgraph{G<:AbstractFeaturedGraph,T} <: AbstractFeaturedGraph
    fg::G
    nodes::T
end

FeaturedSubgraph(ng::NullGraph, ::AbstractVector) = ng

subgraph(fg::AbstractFeaturedGraph, nodes::AbstractVector) = FeaturedSubgraph(fg, nodes)
subgraph(fsg::FeaturedSubgraph, nodes::AbstractVector) = FeaturedSubgraph(fsg.fg, nodes)

StatsBase.sample(fsg::FeaturedSubgraph, n::Int) =
    subgraph(fsg, sample(fsg.nodes, n; replace=false))

graph(fsg::FeaturedSubgraph) = graph(fsg.fg)

Graphs.vertices(fsg::FeaturedSubgraph) = fsg.nodes

Graphs.adjacency_matrix(fsg::FeaturedSubgraph) = view(adjacency_matrix(fsg.fg), fsg.nodes, fsg.nodes)

node_feature(fsg::FeaturedSubgraph) = view(node_feature(fsg.fg), :, fsg.nodes)

function edge_feature(fsg::FeaturedSubgraph)
    sg = graph(fsg.fg)
    sel = map(x -> x in fsg.nodes, colvals(sg.S))
    sel .&= map(x -> x in fsg.nodes, rowvals(sg.S))
    eidx = sort!(unique!(edgevals(sg)[sel]))
    return view(edge_feature(fsg.fg), :, eidx)
end

global_feature(fsg::FeaturedSubgraph) = global_feature(fsg.fg)

Graphs.is_directed(fsg::FeaturedSubgraph) = is_directed(fsg.fg)

Graphs.neighbors(fsg::FeaturedSubgraph) = mapreduce(i -> neighbors(graph(fsg), i), vcat, fsg.nodes)

incident_edges(fsg::FeaturedSubgraph) = mapreduce(i -> incident_edges(graph(fsg), i), vcat, fsg.nodes)

repeat_nodes(fsg::FeaturedSubgraph) = mapreduce(i -> repeat_nodes(graph(fsg), i), vcat, fsg.nodes)
