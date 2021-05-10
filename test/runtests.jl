using GraphSignals
using CUDA
using FillArrays
using GraphLaplacians
using LightGraphs
using SimpleWeightedGraphs
using Test

tests = [
    "featuredgraph",
    "graph",
    "linalg",
    "simplegraph",
    "weightedgraph",
    "edgeindex",
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
