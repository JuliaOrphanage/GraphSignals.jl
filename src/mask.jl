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

# node_feature(gm::GraphMask)

# edge_feature(gm::GraphMask)

global_feature(gm::GraphMask) = global_feature(gm.fg)
