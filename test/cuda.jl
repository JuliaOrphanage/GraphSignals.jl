T = Float32
adj = cu(T[0 1 0 1;
       1 0 1 0;
       0 1 0 1;
       1 0 1 0])
nf = cu(rand(T, 3, 4))

@testset "cuda" begin
    @testset "featuredgraph" begin
        fg = FeaturedGraph(adj)
        @test has_graph(fg)
        @test !has_node_feature(fg)
        @test !has_edge_feature(fg)
        @test !has_global_feature(fg)
        @test graph(fg) isa CuMatrix{T}
        @test node_feature(fg) isa Fill{T,2,Tuple{Base.OneTo{Int64},Base.OneTo{Int64}}}
        @test edge_feature(fg) isa Fill{T,2,Tuple{Base.OneTo{Int64},Base.OneTo{Int64}}}
        @test global_feature(fg) isa Fill{T,1,Tuple{Base.OneTo{Int64}}}


        fg = FeaturedGraph(adj, nf=nf)
        @test has_graph(fg)
        @test has_node_feature(fg)
        @test !has_edge_feature(fg)
        @test !has_global_feature(fg)
        @test graph(fg) isa CuMatrix{T}
        @test node_feature(fg) isa CuMatrix{T}
        @test edge_feature(fg) isa Fill{T,2,Tuple{Base.OneTo{Int64},Base.OneTo{Int64}}}
        @test global_feature(fg) isa Fill{T,1,Tuple{Base.OneTo{Int64}}}
    end

    @testset "EdgeIndex" begin
        T = CuVector{NTuple{2,Int64}}
        # undirected simple graph
        adjl1 = T[
            [(2, 1), (4, 2), (5, 3)],
            [(1, 1)],
            [],
            [(1, 2), (5, 4)],
            [(1, 3), (4, 4)]
        ]

        # directed graph with self loop and multiple edges
        adjl2 = T[
            [(2, 1), (5, 2), (5, 3)],
            [],
            [(1, 4), (4, 5)],
            [(4, 6)],
            [(1, 7), (4, 8)],
        ]

        ei1 = EdgeIndex(adjl1)
        @test ei1.adjl isa Vector{CuVector{Tuple{Int64, Int64}}}
        @test nv(ei1) == 5
        @test ne(ei1) == 4
        @test neighbors(ei1, 1) == adjl1[1]
        @test neighbors(ei1, 2) == adjl1[2]
        @test get(ei1, (1, 5)) == 3
        @test isnothing(get(ei1, (2, 3)))

        ei2 = EdgeIndex(adjl2)
        @test nv(ei2) == 5
        @test ne(ei2) == 8
        @test neighbors(ei2, 1) == adjl2[1]
        @test neighbors(ei2, 3) == adjl2[3]
        @test get(ei2, (3, 1)) == 4
        @test isnothing(get(ei2, (1, 3)))
    end
end
