struct GraphMask <: AbstractDeterministicGraphSampler
    fg::AbstractFeaturedGraph
    mask::AbstractArray
end

"""
    mask(fg, m)

A syntax sugar for masking graph.

Returns a `GraphMask`.
"""
mask(fg::AbstractFeaturedGraph, m::AbstractArray) = GraphMask(fg, m)

# consider direction
"""
    Support adjacency matrix
"""
graph(gm::GraphMask) = adjacency_matrix(gm.fg) .* gm.mask

function node_feature(gm::GraphMask)
    M = gm.mask .!= 0
    m = [.|(M[:, i]...) for i = 1:size(M, 2)]
    return node_feature(gm.fg) .* m'
end

# edge_feature(gm::GraphMask)

global_feature(gm::GraphMask) = global_feature(gm.fg)
