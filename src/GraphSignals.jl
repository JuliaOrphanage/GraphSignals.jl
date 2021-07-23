module GraphSignals

using LinearAlgebra: issymmetric, diag, diagm, Transpose

using CUDA: AnyCuArray, AnyCuVector, CuArray, CuVector, cu
using ChainRulesCore: @non_differentiable
using FillArrays
using Functors: @functor
using GraphLaplacians
using LightGraphs
using LightGraphs: AbstractGraph, outneighbors
using NNlib

import Base: get
import GraphLaplacians: degrees, degree_matrix, inv_sqrt_degree_matrix, laplacian_matrix,
    normalized_laplacian, scaled_laplacian
import LightGraphs: nv, ne, adjacency_matrix, is_directed, neighbors

export
    # featuredgraph
    AbstractFeaturedGraph,
    NullGraph,
    FeaturedGraph,
    graph,
    node_feature,
    edge_feature,
    global_feature,
    has_graph,
    has_node_feature,
    has_edge_feature,
    has_global_feature,

    # graph
    nv,
    ne,
    adjacency_list,
    is_directed,
    fetch_graph,

    # edgeindex
    EdgeIndex,
    neighbors,
    get,
    edge_scatter,
    neighbor_scatter,

    # linalg
    adjacency_matrix,
    degrees,
    degree_matrix,
    inv_sqrt_degree_matrix,
    laplacian_matrix,
    normalized_laplacian,
    scaled_laplacian,
    laplacian_matrix!,
    normalized_laplacian!,
    scaled_laplacian!,

    # mask
    GraphMask,
    mask

include("graph.jl")
include("linalg.jl")

include("featuredgraph.jl")
include("utils.jl")

include("edgeindex.jl")
include("cuda.jl")
include("sampling.jl")
include("mask.jl")

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
@non_differentiable EdgeIndex(x...)
@non_differentiable neighbors(x...)
@non_differentiable order_edges(x...)
@non_differentiable aggregate_index(x...)

end
