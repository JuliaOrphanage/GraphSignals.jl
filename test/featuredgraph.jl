adj = [0 1 0 1;
       1 0 1 0;
       0 1 0 1;
       1 0 1 0]

@testset "featuredgraph" begin
    fg = FeaturedGraph(adj)
    @test graph(fg) === adj
    @test length(node_feature(fg)) == 0
    @test nv(fg) == 4
end
