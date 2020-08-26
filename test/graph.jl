adj1 = [0 1 0 1;
        1 0 1 0;
        0 1 0 1;
        1 0 1 0]
adj2 = [1 1 0 1 0;
        1 1 1 0 0;
        0 1 1 1 1;
        1 0 1 1 0;
        1 0 1 0 1]
adjl = [[2, 4], [1, 3], [2, 4], [1, 3]]

@testset "graph" begin
    ng = NullGraph()
    fg1 = FeaturedGraph(adj1)
    fg2 = FeaturedGraph(adj2)

    @test adjacency_list(ng) == [zeros(0)]
    @test adjacency_list(fg1) == adjl
    @test adjacency_list(adj1) == adjl
    @test adjacency_list(adjl) == adjl

    @test nv(ng) == 0
    @test nv(fg1) == 4
    @test nv(fg2) == 5

    @test ne(ng) == 0
    @test ne(fg1) == 8
    @test ne(fg2) == 16

    @test fetch_graph(ng, fg1) == adj1
    @test fetch_graph(fg1, ng) == adj1
    @test fetch_graph(fg1, fg2) == adj1
end
