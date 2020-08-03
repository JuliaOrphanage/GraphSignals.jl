using GraphSignals
using Test

tests = [
    "featuredgraph",
    "linalg",
]

@testset "GraphSignals.jl" begin
    for t in tests
        include("$(t).jl")
    end
end
