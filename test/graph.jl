N = 4
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
adjl = [[2, 4], [1, 3], [2, 4], [1, 3]]

ug = SimpleGraph(N)
add_edge!(ug, 1, 2); add_edge!(ug, 1, 3); add_edge!(ug, 1, 4)
add_edge!(ug, 2, 3); add_edge!(ug, 3, 4)

@testset "graph" begin
    ng = NullGraph()
    fg1 = FeaturedGraph(adj1)
    fg2 = FeaturedGraph(adj2)
    fg3 = FeaturedGraph(ug)
    fg4 = FeaturedGraph(adjl)

    @test adjacency_list(ng) == [zeros(0)]
    @test adjacency_list(fg1) == adjl
    @test adjacency_list(adj1) == adjl
    @test adjacency_list(adjl) == adjl
    @test adjacency_list(adj2) == [[1,2,4,5], [1,2,3], [2,3,4,5], [1,3,4], [1,3,5]]

    @test nv(ng) == 0
    @test nv(fg1) == 4
    @test nv(fg2) == 5
    @test nv(fg3) == 4
    @test nv(adjl) == 4

    @test ne(ng) == 0
    @test ne(fg1) == 4
    @test ne(fg2) == 11
    @test ne(adj2) == 11
    @test ne(adj2, self_loop=true) == 16
    @test ne(adj3) == 5
    @test ne(adj3, self_loop=true) == 10
    @test ne(fg3) == 5
    @test ne(adjl, false) == 4
    @test ne(adjl, true) == 8
    @test ne(fg4) == 4

    @test fetch_graph(ng, fg1) == adj1
    @test fetch_graph(fg1, ng) == adj1
    @test fetch_graph(fg1, fg2) == adj1
end
