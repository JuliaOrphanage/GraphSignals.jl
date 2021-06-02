# undirected simple graph
adjl1 = [
    [(2, 1), (4, 2), (5, 3)],
    [(1, 1)],
    [],
    [(1, 2), (5, 4)],
    [(1, 3), (4, 4)]
]

E1 = rand(10, 4)

# directed graph with self loop and multiple edges
adjl2 = [
    [(2, 1), (5, 2), (5, 3)],
    [],
    [(1, 4), (4, 5)],
    [(4, 6)],
    [(1, 7), (4, 8)],
]

E2 = rand(10, 8)

@testset "EdgeIndex" begin
    ei1 = EdgeIndex(adjl1)
    @test ei1.adjl isa Vector{Vector{Tuple{Int64, Int64}}}
    @test nv(ei1) == 5
    @test ne(ei1) == 4
    @test neighbors(ei1, 1) == [(2, 1), (4, 2), (5, 3)]
    @test neighbors(ei1, 3) == []
    @test get(ei1, (1, 5)) == 3
    @test isnothing(get(ei1, (2, 3)))
    @test generate_cluster_index(E1, ei1) == ([2, 4, 5, 5], [1, 1, 1, 4])

    ei2 = EdgeIndex(adjl2)
    @test nv(ei2) == 5
    @test ne(ei2) == 8
    @test neighbors(ei2, 1) == [(2, 1), (5, 2), (5, 3)]
    @test neighbors(ei2, 2) == []
    @test get(ei2, (3, 1)) == 4
    @test isnothing(get(ei2, (1, 3)))
    @test generate_cluster_index(E2, ei2; direction=:inward) == [2, 5, 5, 1, 4, 4, 1, 4]
    @test generate_cluster_index(E2, ei2; direction=:outward) == [1, 1, 1, 3, 3, 4, 5, 5]
end