"""
    mask(fg, m)

A syntax sugar for masking graph.

Returns a `FeaturedSubgraph`.
"""
mask(fg::AbstractFeaturedGraph, m::AbstractVector) = subgraph(fg, m)
