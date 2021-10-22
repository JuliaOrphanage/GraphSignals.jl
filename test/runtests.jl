using GraphSignals
using CUDA
using Flux
using FillArrays
using GraphLaplacians
using Graphs
using LinearAlgebra
using SimpleWeightedGraphs
using SparseArrays
using Test
CUDA.allowscalar(false)

include("test_utils.jl")

tests = [
    "graph",
    "linalg",
    "sparsegraph",
    "featuredgraph",
    "mask",
]

if CUDA.functional()
    push!(tests, "cuda")
end

@testset "GraphSignals.jl" begin
    for t in tests
        include("$(t).jl")
    end
end
