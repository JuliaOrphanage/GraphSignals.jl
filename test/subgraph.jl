@testset "subgraph" begin
    T = Float64
    V = 5
    nf = rand(3, V)
    gf = rand(7)

    nodes = [1,2,3,5]

    @testset "null graph" begin
        fg = NullGraph()

        subg = FeaturedSubgraph(fg, nodes)
        @test subg === fg
        @test subgraph(fg, nodes) === fg
    end

    @testset "undirected graph" begin
        E = 5
        ef = rand(5, E)
        adjm = T[0 1 0 1 0; # symmetric
                1 0 1 0 0;
                0 1 0 1 0;
                1 0 1 0 0;
                0 0 0 0 1]
        fg = FeaturedGraph(adjm, nf=nf, ef=ef, gf=gf)

        subg = FeaturedSubgraph(fg, nodes)
        @test graph(subg) === graph(fg)
        @test subgraph(fg, nodes) == subg
        @test is_directed(subg) == is_directed(fg)
        @test adjacency_matrix(subg) == view(adjm, nodes, nodes)
        @test adjacency_matrix(subg) isa SubArray
        @test node_feature(subg) == view(nf, :, nodes)
        @test edge_feature(subg) == view(ef, :, [1,2,5])
        @test global_feature(subg) == gf

        @test vertices(subg) == nodes
        @test neighbors(subg) == [2, 4, 1, 3, 2, 4, 5]
        @test incident_edges(subg) == [1, 3, 1, 2, 2, 4, 5]
        @test GraphSignals.repeat_nodes(subg) == [1, 1, 2, 2, 3, 3, 5]

        @test GraphSignals.degrees(subg) == [2, 2, 2, 1]
        @test GraphSignals.degree_matrix(subg) == diagm([2, 2, 2, 1])
        @test GraphSignals.normalized_adjacency_matrix(subg) ≈ [0 .5 0 0;
                                                                .5 0 .5 0;
                                                                0 .5 0 0;
                                                                0 0 0 1]
        @test GraphSignals.laplacian_matrix(subg) == [2 -1 0 0;
                                                     -1 2 -1 0;
                                                      0 -1 2 0;
                                                      0 0 0 0]
        @test GraphSignals.normalized_laplacian(subg) ≈ [1 -.5 0 0;
                                                        -.5 1 -.5 0;
                                                         0 -.5 1 0;
                                                         0 0 0 0]
        @test GraphSignals.scaled_laplacian(subg) ≈ [0 -.5 0 0;
                                                     -.5 0 -.5 0;
                                                     0 -.5 0 0;
                                                     0 0 0 -1]

        rand_subgraph = sample(subg, 3)
        @test rand_subgraph isa FeaturedSubgraph
        @test length(rand_subgraph.nodes) == 3
        @test rand_subgraph.nodes ⊆ subg.nodes
    end

    @testset "directed graph" begin
        E = 16
        ef = rand(5, E)
        adjm = [1 1 0 1 0; # asymmetric
                1 1 1 0 0;
                0 1 1 1 1;
                1 0 1 1 0;
                1 0 1 0 1]
        fg = FeaturedGraph(adjm, nf=nf, ef=ef, gf=gf)

        subg = FeaturedSubgraph(fg, nodes)
        @test graph(subg) === graph(fg)
        @test subgraph(fg, nodes) == subg
        @test is_directed(subg) == is_directed(fg)
        @test adjacency_matrix(subg) == view(adjm, nodes, nodes)
        @test adjacency_matrix(subg) isa SubArray
        @test node_feature(subg) == view(nf, :, nodes)
        @test edge_feature(subg) == view(ef, :, [1,2,4,5,6,7,8,9,11,15,16])
        @test global_feature(subg) == gf
        @test parent(subg) === subg.fg

        @test vertices(subg) == nodes
        @test neighbors(subg) == [1, 2, 4, 5, 1, 2, 3, 2, 3, 4, 5, 3, 5]
        @test incident_edges(subg) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16]
        @test GraphSignals.repeat_nodes(subg) == [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 5, 5]

        rand_subgraph = sample(subg, 3)
        @test rand_subgraph isa FeaturedSubgraph
        @test length(rand_subgraph.nodes) == 3
        @test rand_subgraph.nodes ⊆ subg.nodes
    end
end
