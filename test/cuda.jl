T = Float32
adj = cu(T[0 1 0 1;
       1 0 1 0;
       0 1 0 1;
       1 0 1 0])
nf = cu(rand(T, 3, 4))

@testset "cuda" begin
    @testset "SparseGraph" begin
        T = CuVector{NTuple{2,Int64}}
        # undirected graph with self loop
        iadjl1 = T[
            [(2, 1), (4, 2), (5, 3)],
            [(1, 1)],
            [(3, 4)],
            [(1, 2), (5, 5)],
            [(1, 3), (4, 5)]
        ]
        
        E1 = cu(rand(10, 5))

        # directed graph with self loop and multiple edges
        iadjl2 = T[
            [(2, 1), (5, 2), (5, 3)],
            [],
            [(1, 4), (4, 5)],
            [(4, 6)],
            [(1, 7), (4, 8)],
        ]
        
        E2 = cu(rand(10, 8))

        sg1 = SparseGraph(iadjl1)
        @test sg1.iadjl isa Vector{CuVector{Tuple{Int64, Int64}}}
        @test nv(sg1) == 5
        @test ne(sg1) == 5
        @test Array(neighbors(sg1, 1)) == Array(iadjl1[1])
        @test Array(neighbors(sg1, 2)) == Array(iadjl1[2])
        @test_throws ErrorException get(sg1, (1, 5))
        @test GraphSignals.aggregate_index(sg1) == ([1, 1, 1, 3, 4], [2, 4, 5, 3, 5])
        @test_throws ArgumentError GraphSignals.aggregate_index(sg1, direction=:in)
        @test size(edge_scatter(+, E1, sg1)) == (10, 5)

        sg2 = SparseGraph(iadjl2)
        @test nv(sg2) == 5
        @test ne(sg2) == 8
        @test Array(neighbors(sg2, 1)) == Array(iadjl2[1])
        @test Array(neighbors(sg2, 3)) == Array(iadjl2[3])
        @test_throws ErrorException get(sg2, (3, 1))
        @test GraphSignals.aggregate_index(sg2, direction=:inward) == [2, 5, 5, 1, 4, 4, 1, 4]
        @test GraphSignals.aggregate_index(sg2, direction=:outward) == [1, 1, 1, 3, 3, 4, 5, 5]
        @test size(edge_scatter(+, E2, sg2, direction=:inward)) == (10, 5)
        @test size(edge_scatter(+, E2, sg2, direction=:outward)) == (10, 5)
    end

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

        adjm_cpu = T[0 1 0 1;
                    1 0 1 0;
                    0 1 0 1;
                    1 0 1 0]
        fg = FeaturedGraph(adj) |> cu
        @test graph(fg) isa CuMatrix{T}
    end
end
