# undirected graph with self loop
adjl1 = Vector{Int64}[
    [2, 4, 5],
    [1],
    [3],
    [1, 5],
    [1, 4]
]

iadjl1 = [
    [(2, 1), (4, 2), (5, 3)],
    [(1, 1)],
    [(3, 4)],
    [(1, 2), (5, 5)],
    [(1, 3), (4, 5)]
]

E1 = rand(10, 5)

# directed graph with self loop and multiple edges
adjl2 = Vector{Int64}[
    [2, 5, 5],
    [],
    [1, 4],
    [4],
    [1, 4],
]

iadjl2 = [
    [(2, 1), (5, 2), (5, 3)],
    [],
    [(1, 4), (4, 5)],
    [(4, 6)],
    [(1, 7), (4, 8)],
]

E2 = rand(10, 8)

adjm = [0 1 0 1 1; # symmetric
        1 0 0 0 0;
        0 0 1 0 0;
        1 0 0 0 1;
        1 0 0 1 0]

ug = SimpleGraph(5)
add_edge!(ug, 1, 2); add_edge!(ug, 1, 4); add_edge!(ug, 1, 5)
add_edge!(ug, 3, 3); add_edge!(ug, 4, 5)

wug = SimpleWeightedGraph(5)
add_edge!(wug, 1, 2, 2); add_edge!(wug, 1, 4, 2); add_edge!(wug, 1, 5, 1)
add_edge!(wug, 3, 3, 5); add_edge!(wug, 4, 5, 2)

@testset "EdgeIndex" begin
    ei1 = EdgeIndex(iadjl1)
    @test ei1.iadjl isa Vector{Vector{Tuple{Int64, Int64}}}
    @test nv(ei1) == 5
    @test ne(ei1) == 5
    @test neighbors(ei1, 1) == [(2, 1), (4, 2), (5, 3)]
    @test neighbors(ei1, 3) == [(3, 4)]
    @test get(ei1, (1, 5)) == 3
    @test isnothing(get(ei1, (2, 3)))
    @test GraphSignals.generate_cluster_index(ei1) == ([1, 1, 1, 3, 4], [2, 4, 5, 3, 5])
    @test_throws ArgumentError GraphSignals.generate_cluster_index(ei1, direction=:in)
    @test size(edge_scatter(+, E1, ei1)) == (10, 5)

    ei2 = EdgeIndex(iadjl2)
    @test nv(ei2) == 5
    @test ne(ei2) == 8
    @test neighbors(ei2, 1) == [(2, 1), (5, 2), (5, 3)]
    @test neighbors(ei2, 2) == []
    @test get(ei2, (3, 1)) == 4
    @test isnothing(get(ei2, (1, 3)))
    @test GraphSignals.generate_cluster_index(ei2, direction=:inward) == [2, 5, 5, 1, 4, 4, 1, 4]
    @test GraphSignals.generate_cluster_index(ei2, direction=:outward) == [1, 1, 1, 3, 3, 4, 5, 5]
    @test size(edge_scatter(+, E2, ei2, direction=:inward)) == (10, 5)
    @test size(edge_scatter(+, E2, ei2, direction=:outward)) == (10, 5)

    @test GraphSignals.order_edges(adjl1, directed=false) == iadjl1
    @test GraphSignals.order_edges(adjl2, directed=true) == iadjl2
    
    fg1 = FeaturedGraph(adjl1, directed=:undirected)
    @test EdgeIndex(fg1).iadjl == iadjl1
    fg2 = FeaturedGraph(adjl2, directed=:directed)
    @test EdgeIndex(fg2).iadjl == iadjl2
    @test EdgeIndex(adjm).iadjl == iadjl1
    @test EdgeIndex(ug).iadjl == iadjl1
    @test EdgeIndex(wug).iadjl == iadjl1
end