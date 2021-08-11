@testset "graph" begin
    N = 6
    adj1 = [0 1 0 1; # symmetric
            1 0 1 0;
            0 1 0 1;
            1 0 1 0]
    adj2 = [1 1 0 1 0; # asymmetric
            1 1 1 0 0;
            0 1 1 1 1;
            1 0 1 1 0;
            1 0 1 0 1]
    adj3 = [1 1 0 1 0; # symmetric
            1 1 1 0 0;
            0 1 1 1 1;
            1 0 1 1 0;
            0 0 1 0 1]
    adjl = [
        [2, 4],
        [1, 3],
        [2, 4],
        [1, 3]
    ]

    ug = SimpleGraph(N)
    add_edge!(ug, 1, 2); add_edge!(ug, 1, 3); add_edge!(ug, 2, 3)
    add_edge!(ug, 3, 4); add_edge!(ug, 2, 5); add_edge!(ug, 3, 6)

    dg = SimpleDiGraph(N)
    add_edge!(dg, 1, 3); add_edge!(dg, 2, 3); add_edge!(dg, 1, 6)
    add_edge!(dg, 2, 5); add_edge!(dg, 3, 4); add_edge!(dg, 3, 5)

    wug = SimpleWeightedGraph(N)
    add_edge!(wug, 1, 2, 2); add_edge!(wug, 1, 3, 2); add_edge!(wug, 2, 3, 1)
    add_edge!(wug, 3, 4, 5); add_edge!(wug, 2, 5, 2); add_edge!(wug, 3, 6, 2)

    wdg = SimpleWeightedDiGraph(N)
    add_edge!(wdg, 1, 3, 2); add_edge!(wdg, 2, 3, 2); add_edge!(wdg, 1, 6, 1)
    add_edge!(wdg, 2, 5, -2); add_edge!(wdg, 3, 4, -2); add_edge!(wdg, 3, 5, -1)

    adjl_ug = Vector{Int64}[[2, 3], [1, 3, 5], [1, 2, 4, 6], [3], [2], [3]]
    adjl_dg = Vector{Int64}[[3, 6], [3, 5], [4, 5], [], [], []]

    ng = NullGraph()
    fg1 = FeaturedGraph(adj1)
    fg2 = FeaturedGraph(adj2)
    fg3 = FeaturedGraph(adjl)
    fg4 = FeaturedGraph(ug)
    fg5 = FeaturedGraph(dg)
    fg6 = FeaturedGraph(wug)
    fg7 = FeaturedGraph(wdg)

    @test adjacency_list(ng) == [zeros(0)]
    @test adjacency_list(fg1) == adjl
    @test adjacency_list(adj1) == adjl
    @test adjacency_list(adjl) == adjl
    @test adjacency_list(adj2) == [[1,2,4,5], [1,2,3], [2,3,4,5], [1,3,4], [1,3,5]]
    @test adjacency_list(fg4) == adjl_ug
    @test adjacency_list(fg5) == adjl_dg
    @test adjacency_list(fg6) == adjl_ug
    @test adjacency_list(fg7) == adjl_dg

    @test nv(ng) == 0
    @test nv(fg1) == 4
    @test nv(fg2) == 5
    @test nv(adjl) == 4

    @test ne(ng) == 0
    @test ne(fg1) == 4
    @test ne(fg2) == 16
    @test ne(adj2) == 16
    @test ne(adj3) == 10
    @test ne(fg3) == 4
    @test ne(adjl) == 4
    @test ne(adjl, false) == 4
    @test ne(adjl, true) == 8

    @test !is_directed(fg1)
    @test is_directed(fg2)
    @test !is_directed(adjl)

    @test fetch_graph(ng, fg1).S == adj1
    @test fetch_graph(fg1, ng).S == adj1
    @test fetch_graph(fg1, fg2).S == adj1
end
