module GraphSignals

using NNlib
using LinearAlgebra: issymmetric, diag, diagm, Transpose

using CUDA: AnyCuVector, CuArray, CuVector, cu
using FillArrays
using GraphLaplacians
using LightGraphs
using LightGraphs: AbstractGraph, outneighbors
using NNlib
using Zygote

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

include("featuredgraph.jl")
include("graph.jl")
include("linalg.jl")
include("utils.jl")

include("edgeindex.jl")
include("cuda.jl")
include("sampling.jl")
include("mask.jl")

end
