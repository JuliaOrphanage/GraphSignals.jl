module GraphSignals

using LinearAlgebra
using SparseArrays
using SparseArrays: AbstractSparseMatrixCSC

using CUDA
using CUDA.CUSPARSE
using ChainRulesCore: @non_differentiable
using FillArrays
using Functors: @functor
using Graphs
using Graphs: AbstractGraph, AbstractSimpleGraph
using SimpleWeightedGraphs: AbstractSimpleWeightedGraph, weights
using StatsBase
using NNlib, NNlibCUDA

import Graphs: laplacian_matrix

export
    # featuredgraph
    AbstractFeaturedGraph,
    NullGraph,
    FeaturedGraph,
    ConcreteFeaturedGraph,
    graph,
    matrixtype,
    node_feature,
    edge_feature,
    global_feature,
    has_graph,
    has_node_feature,
    has_edge_feature,
    has_global_feature,

    # graph
    adjacency_list,

    # sparsegraph
    SparseGraph,
    incident_edges,
    edge_index,
    edge_scatter,
    neighbor_scatter,

    # linalg
    laplacian_matrix,
    normalized_laplacian,
    scaled_laplacian,
    laplacian_matrix!,
    normalized_laplacian!,
    scaled_laplacian!,

    # subgraph
    FeaturedSubgraph,
    subgraph,

    # mask
    mask,

    # random
    random_walk,
    neighbor_sample

include("graph.jl")
include("linalg.jl")
include("sparsematrix.jl")
include("sparsegraph.jl")

include("featuredgraph.jl")
include("utils.jl")

include("cuda.jl")

include("subgraph.jl")
# include("sampling.jl")
include("mask.jl")
include("random.jl")

# Non-differentiables

@non_differentiable nv(x...)
@non_differentiable ne(x...)
@non_differentiable adjacency_list(x...)
@non_differentiable GraphSignals.adjacency_matrix(x...)
@non_differentiable is_directed(x...)
@non_differentiable has_graph(x...)
@non_differentiable has_node_feature(x...)
@non_differentiable has_edge_feature(x...)
@non_differentiable has_global_feature(x...)
@non_differentiable SparseGraph(x...)
@non_differentiable neighbors(x...)
@non_differentiable incident_edges(x...)
@non_differentiable order_edges(x...)
@non_differentiable aggregate_index(x...)

end
