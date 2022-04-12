using GraphSignals
using CUDA
using Distances
using Flux
using FillArrays
using Graphs
using LinearAlgebra
using SimpleWeightedGraphs
using SparseArrays
using StatsBase
using Test
CUDA.allowscalar(false)

include("test_utils.jl")

tests = [
    "graph",
    "linalg",
    "sparsegraph",
    "featuredgraph",
    "subgraph",
    "random",
    "neighbor_graphs"
]

if CUDA.functional()
    push!(tests, "cuda")
end

@testset "GraphSignals.jl" begin
    for t in tests
        include("$(t).jl")
    end
end
