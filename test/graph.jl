adj1 = [0 1 0 1;
        1 0 1 0;
        0 1 0 1;
        1 0 1 0]
adj2 = [1 1 0 1;
       1 1 1 0;
       0 1 1 1;
       1 0 1 1]

@testset "graph" begin
    @testset "fetch_graph" begin
        ng = NullGraph()
        fg1 = FeaturedGraph(adj1)
        fg2 = FeaturedGraph(adj2)
        @test fetch_graph(ng, fg1) == adj1
        @test fetch_graph(fg1, ng) == adj1
        @test fetch_graph(fg1, fg2) == adj1
    end
end
