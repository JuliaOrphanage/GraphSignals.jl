adj = [0 1 0 1;
       1 0 1 0;
       0 1 0 1;
       1 0 1 0]
nf = rand(3, 4)
ef = rand(5, 6)
gf = rand(7)


@testset "featuredgraph" begin
    ng = NullGraph()
    @test has_graph(ng) == false
    @test has_node_feature(ng) == false
    @test has_edge_feature(ng) == false
    @test has_global_feature(ng) == false
    @test isnothing(graph(ng))
    @test isnothing(node_feature(ng))
    @test isnothing(edge_feature(ng))
    @test isnothing(global_feature(ng))


    fg = FeaturedGraph(adj)
    @test has_graph(fg)
    @test has_node_feature(fg) == false
    @test has_edge_feature(fg) == false
    @test has_global_feature(fg) == false
    @test graph(fg) == adj
    @test node_feature(fg) == zeros(0,0)
    @test edge_feature(fg) == zeros(0,0)
    @test global_feature(fg) == zeros(0)
    @test nv(fg) == 4


    fg = FeaturedGraph(adj, nf)
    @test has_graph(fg)
    @test has_node_feature(fg)
    @test has_edge_feature(fg) == false
    @test has_global_feature(fg) == false
    @test graph(fg) == adj
    @test node_feature(fg) == nf
    @test edge_feature(fg) == zeros(0,0)
    @test global_feature(fg) == zeros(0)


    fg = FeaturedGraph(adj, nf, ef ,gf)
    @test has_graph(fg)
    @test has_node_feature(fg)
    @test has_edge_feature(fg)
    @test has_global_feature(fg)
    @test graph(fg) == adj
    @test node_feature(fg) == nf
    @test edge_feature(fg) == ef
    @test global_feature(fg) == gf
end
