adj = [0 1 0 1;
       1 0 1 0;
       0 1 0 1;
       1 0 1 0]
nf = rand(3, 4)

@testset "cuda" begin
    fg = FeaturedGraph(cu(adj))
    @test has_graph(fg)
    @test has_node_feature(fg) == false
    @test has_edge_feature(fg) == false
    @test has_global_feature(fg) == false
    @test typeof(graph(fg)) == CuMatrix{Int64}
    @test typeof(node_feature(fg)) == CuMatrix{Int64}
    @test typeof(edge_feature(fg)) == CuMatrix{Int64}
    @test typeof(global_feature(fg)) == CuVector{Int64}


    fg = FeaturedGraph(adj, cu(nf))
    @test has_graph(fg)
    @test has_node_feature(fg)
    @test has_edge_feature(fg) == false
    @test has_global_feature(fg) == false
    @test typeof(graph(fg)) == CuMatrix{Float32}
    @test typeof(node_feature(fg)) == CuMatrix{Float32}
    @test typeof(edge_feature(fg)) == CuMatrix{Float32}
    @test typeof(global_feature(fg)) == CuVector{Float32}
end
