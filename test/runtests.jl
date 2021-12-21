using GraphSignals
using CUDA
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
    "mask",
    "random",
]

if CUDA.functional()
    push!(tests, "cuda")
end

@testset "GraphSignals.jl" begin
    for t in tests
        include("$(t).jl")
    end
end
