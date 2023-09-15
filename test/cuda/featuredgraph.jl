T = Float32

@testset "cuda/featuredgraph" begin
    @testset "undirected graph" begin
        # undirected graph with self loop
        V = 5
        E = 5
        nf = rand(10, V)

        adjm = T[0 1 0 1 1;
                1 0 0 0 0;
                0 0 1 0 0;
                1 0 0 0 1;
                1 0 0 1 0]

        fg = FeaturedGraph(adjm; directed=:undirected, nf=nf) |> gpu
        @test has_graph(fg)
        @test has_node_feature(fg)
        @test !has_edge_feature(fg)
        @test !has_global_feature(fg)
        @test graph(fg) isa SparseGraph
        @test node_feature(fg) isa CuMatrix{T}
    end

    @testset "directed graph" begin
        # directed graph with self loop
        V = 5
        E = 7
        nf = rand(10, V)

        adjm = T[0 0 1 0 1;
                1 0 0 0 0;
                0 0 0 0 0;
                0 0 1 1 1;
                1 0 0 0 0]

        fg = FeaturedGraph(adjm; directed=:directed, nf=nf) |> gpu
        @test has_graph(fg)
        @test has_node_feature(fg)
        @test !has_edge_feature(fg)
        @test !has_global_feature(fg)
        @test graph(fg) isa SparseGraph
        @test node_feature(fg) isa CuMatrix{T}
    end
end
