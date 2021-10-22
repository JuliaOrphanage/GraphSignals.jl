struct FeaturedSubgraph{G<:AbstractFeaturedGraph,T} <: AbstractFeaturedGraph
    fg::G
    nodes::T
end

FeaturedSubgraph(ng::NullGraph, ::AbstractVector) = ng

subgraph(fg::AbstractFeaturedGraph, nodes::AbstractVector) = FeaturedSubgraph(fg, nodes)

Graphs.adjacency_matrix(fsg::FeaturedSubgraph) = view(adjacency_matrix(fsg.fg), fsg.nodes, fsg.nodes)

node_feature(fsg::FeaturedSubgraph) = view(node_feature(fsg.fg), :, fsg.nodes)

function edge_feature(fsg::FeaturedSubgraph)
    sg = graph(fsg.fg)
    sel = map(x -> x in fsg.nodes, colvals(sg.S, nv(sg)))
    sel .&= map(x -> x in fsg.nodes, rowvals(sg.S))
    eidx = sort!(unique!(edgevals(sg)[sel]))
    return view(edge_feature(fsg.fg), :, eidx)
end

global_feature(fsg::FeaturedSubgraph) = global_feature(fsg.fg)

Graphs.is_directed(fsg::FeaturedSubgraph) = is_directed(fsg.fg)
