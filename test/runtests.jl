using GraphSignals
using CUDA
using FillArrays
using GraphLaplacians
using LightGraphs
using LinearAlgebra
using SimpleWeightedGraphs
using SparseArrays
using Test

include("test_utils.jl")

tests = [
    "graph",
    "linalg",
    "sparsegraph",
    "featuredgraph",
    "mask",
]

# if CUDA.functional()
#     push!(tests, "cuda")
# end

@testset "GraphSignals.jl" begin
    for t in tests
        include("$(t).jl")
    end
end
