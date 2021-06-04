struct GraphMask{G<:AbstractFeaturedGraph,T} <: AbstractDeterministicGraphSampler
    fg::G
    mask::T
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

binarize_mask(gm::GraphMask) = gm.mask .!= 0

function node_feature(gm::GraphMask)
    M = binarize_mask(gm)
    m = [.|(M[:, i]...) for i = 1:size(M, 2)]
    return node_feature(gm.fg) .* m'
end

# function edge_feature(gm::GraphMask)
#     M = binarize_mask(gm)
#     m
#     return edge_feature(gm.fg) .* m
# end

global_feature(gm::GraphMask) = global_feature(gm.fg)
