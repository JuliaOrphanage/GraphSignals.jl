@testset "SparseGraph" begin
    # undirected graph with self loop

    adjm1 = [0 1 0 1 1;
            1 0 0 0 0;
            0 0 1 0 0;
            1 0 0 0 1;
            1 0 0 1 0]

    adjl1 = Vector{Int64}[
        [2, 4, 5],
        [1],
        [3],
        [1, 5],
        [1, 4]
    ]

    V1 = rand(10, 5)
    E1 = rand(10, 5)

    # directed graph with self loop

    adjm2 = [0 0 1 0 1;
            1 0 0 0 0;
            0 0 0 0 0;
            0 0 1 1 1;
            1 0 0 0 0]

    adjl2 = Vector{Int64}[
        [2, 5],
        [],
        [1, 4],
        [4],
        [1, 4],
    ]

    V2 = rand(10, 5)
    E2 = rand(10, 7)

    ug = SimpleGraph(5)
    add_edge!(ug, 1, 2); add_edge!(ug, 1, 4); add_edge!(ug, 1, 5)
    add_edge!(ug, 3, 3); add_edge!(ug, 4, 5)

    wug = SimpleWeightedGraph(5)
    add_edge!(wug, 1, 2, 2); add_edge!(wug, 1, 4, 2); add_edge!(wug, 1, 5, 1)
    add_edge!(wug, 3, 3, 5); add_edge!(wug, 4, 5, 2)

    adjm3 = [0 2 0 2 1;
             2 0 0 0 0;
             0 0 5 0 0;
             2 0 0 0 2;
             1 0 0 2 0]

    sg1 = SparseGraph(adjm1, false)
    @test sg1.S isa SparseMatrixCSC
    @test sg1.edges == [1, 3, 4, 1, 2, 3, 5, 4, 5]
    @test sg1 == SparseGraph(adjm1, false)
    @test nv(sg1) == 5
    @test ne(sg1) == 5
    @test !is_directed(sg1)
    @test repr(sg1) == "SparseGraph(#V=5, #E=5)"

    @test neighbors(sg1, 1) == adjl1[1]
    @test neighbors(sg1, 3) == adjl1[3]
    @test incident_edges(sg1, 1) == [1, 3, 4]
    @test incident_edges(sg1, 3) == [2]

    @test sg1[1, 5] == 1
    @test sg1[CartesianIndex((1, 5))] == 1
    @test edge_index(sg1, 1, 5) == 4
    @test collect(edges(sg1)) == [
        (1, (2, 1)),
        (2, (3, 3)),
        (3, (4, 1)),
        (4, (5, 1)),
        (5, (5, 4)),
    ]

    @test GraphSignals.aggregate_index(sg1, :edge, :inward) == [1, 3, 1, 1, 4]
    @test GraphSignals.aggregate_index(sg1, :edge, :outward) == [2, 3, 4, 5, 5]
    @test GraphSignals.aggregate_index(sg1, :vertex, :inward) == [[2, 4, 5], [1], [3], [1, 5], [1, 4]]
    @test GraphSignals.aggregate_index(sg1, :vertex, :outward) == [[2, 4, 5], [1], [3], [1, 5], [1, 4]]
    @test_throws ArgumentError GraphSignals.aggregate_index(sg1, :edge, :in)
    @test_throws ArgumentError GraphSignals.aggregate_index(sg1, :foo, :inward)
    @test size(edge_scatter(+, E1, sg1)) == (10, 5)
    X = neighbor_scatter(+, V1, sg1, direction=:undirected)
    @test size(X) == (10, 5)
    @test X[:,1] == vec(sum(V1[:, [2, 4, 5]], dims=2))


    sg2 = SparseGraph(adjm2, true)
    @test sg2.S isa SparseMatrixCSC
    @test sg2.edges == collect(1:7)
    @test nv(sg2) == 5
    @test ne(sg2) == 7
    @test is_directed(sg2)
    @test repr(sg2) == "SparseGraph(#V=5, #E=7)"

    @test neighbors(sg2, 1) == adjl2[1]
    @test neighbors(sg2, 2) == adjl2[2]
    @test_throws ArgumentError neighbors(sg2, 2, dir=:none)
    @test incident_edges(sg2, 1, dir=:out) == [1, 2]
    @test incident_edges(sg2, 1, dir=:in) == [3, 6]
    @test_throws ArgumentError incident_edges(sg2, 2, dir=:none)

    @test sg2[1, 3] == 1
    @test sg2[CartesianIndex((1, 3))] == 1
    @test edge_index(sg2, 1, 3) == 3
    @test collect(edges(sg2)) == [
        (1, (2, 1)),
        (2, (5, 1)),
        (3, (1, 3)),
        (4, (4, 3)),
        (5, (4, 4)),
        (6, (1, 5)),
        (7, (4, 5)),
    ]

    @test GraphSignals.aggregate_index(sg2, :edge, :inward) == [2, 5, 1, 4, 4, 1, 4]
    @test GraphSignals.aggregate_index(sg2, :edge, :outward) == [1, 1, 3, 3, 4, 5, 5]
    @test GraphSignals.aggregate_index(sg2, :vertex, :inward) == [[2, 5], [], [1, 4], [4], [1, 4]]
    @test GraphSignals.aggregate_index(sg2, :vertex, :outward) == [[3, 5], [1], [], [3, 4, 5], [1]]
    @test size(edge_scatter(+, E2, sg2, direction=:inward)) == (10, 5)
    @test size(edge_scatter(+, E2, sg2, direction=:outward)) == (10, 5)
    X = neighbor_scatter(+, V2, sg2, direction=:inward)
    @test size(X) == (10, 5)
    @test X[:,1] == vec(sum(V2[:, [2, 5]], dims=2))
    X = neighbor_scatter(+, V2, sg2, direction=:outward)
    @test size(X) == (10, 5)
    @test X[:,1] == vec(sum(V2[:, [3, 5]], dims=2))

    sg1 = SparseGraph(adjl1, false)
    @test sg1.S == adjm1
    sg2 = SparseGraph(adjl2, true)
    @test sg2.S == adjm2
    @test SparseGraph(ug).S == adjm1
    @test SparseGraph(wug).S == adjm3

    gradtest(x -> edge_scatter(+, x, sg1), E1)
    gradtest(x -> edge_scatter(+, x, sg2, direction=:inward), E2)
    gradtest(x -> edge_scatter(+, x, sg2, direction=:outward), E2)
    gradtest(x -> neighbor_scatter(+, x, sg1, direction=:undirected), V1)
    gradtest(x -> neighbor_scatter(+, x, sg2, direction=:inward), V2)
    gradtest(x -> neighbor_scatter(+, x, sg2, direction=:outward), V2)
end