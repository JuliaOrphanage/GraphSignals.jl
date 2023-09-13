using GraphSignals
using CUDA
using Distances
using Flux
using FillArrays
using Graphs
using LinearAlgebra
using MLUtils
using NNlib
using SimpleWeightedGraphs
using SparseArrays
using StatsBase
using Test
CUDA.allowscalar(false)

include("test_utils.jl")

cuda_tests = [
    "cuda/linalg",
    "cuda/featuredgraph",
    "cuda/sparsematrix",
    "cuda/sparsegraph",
    "cuda/graphdomain",
]

tests = [
    "positional",
    "graph",
    "linalg",
    "sparsegraph",
    "graphdomain",
    "graphsignal",
    "featuredgraph",
    "subgraph",
    "random",
    "neighbor_graphs",
    "dataloader",
    "tokenizer",
]

if CUDA.functional()
    append!(tests, cuda_tests)
end

@testset "GraphSignals.jl" begin
    for t in tests
        include("$(t).jl")
    end
end
