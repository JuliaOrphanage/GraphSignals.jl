T = Float32
adj = T[0 1 0 1;
       1 0 1 0;
       0 1 0 1;
       1 0 1 0]
nf = rand(T, 3, 4)

@testset "cuda" begin
    fg = FeaturedGraph(cu(adj))
    @test has_graph(fg)
    @test !has_node_feature(fg)
    @test !has_edge_feature(fg)
    @test !has_global_feature(fg)
    @test typeof(graph(fg)) == CuMatrix{T}
    @test typeof(node_feature(fg)) == Fill{T,2,Tuple{Base.OneTo{Int64},Base.OneTo{Int64}}}
    @test typeof(edge_feature(fg)) == Fill{T,2,Tuple{Base.OneTo{Int64},Base.OneTo{Int64}}}
    @test typeof(global_feature(fg)) == Fill{T,1,Tuple{Base.OneTo{Int64}}}


    fg = FeaturedGraph(adj, nf=cu(nf))
    @test has_graph(fg)
    @test has_node_feature(fg)
    @test !has_edge_feature(fg)
    @test !has_global_feature(fg)
    @test typeof(graph(fg)) == CuMatrix{T}
    @test typeof(node_feature(fg)) == CuMatrix{T}
    @test typeof(edge_feature(fg)) == Fill{T,2,Tuple{Base.OneTo{Int64},Base.OneTo{Int64}}}
    @test typeof(global_feature(fg)) == Fill{T,1,Tuple{Base.OneTo{Int64}}}
end
