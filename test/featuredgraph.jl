@testset "featuredgraph" begin
    V = 4
    E = 5
    vdim = 3
    edim = 5
    gdim = 7
    pdim = 11

    nf = rand(vdim, V)
    ef = rand(edim, E)
    gf = rand(gdim)
    pf = rand(pdim, V)

    @testset "null graph" begin
        ng = NullGraph()
        @test FeaturedGraph() == ng
        @test FeaturedGraph(ng) == ng
        @test !has_graph(ng)
        @test !has_node_feature(ng)
        @test !has_edge_feature(ng)
        @test !has_global_feature(ng)
        @test !has_positional_feature(ng)
        @test isnothing(graph(ng))
        @test isnothing(node_feature(ng))
        @test isnothing(edge_feature(ng))
        @test isnothing(global_feature(ng))
        @test isnothing(positional_feature(ng))
    end

    @testset "features" begin
        adjm = [0 1 1 1;
                1 0 1 0;
                1 1 0 1;
                1 0 1 0]

        fg = FeaturedGraph(adjm; nf=nf, ef=ef ,gf=gf, pf=pf)
        @test has_graph(fg)
        @test has_node_feature(fg)
        @test has_edge_feature(fg)
        @test has_global_feature(fg)
        @test has_positional_feature(fg)
        @test graph(fg).S == adjm
        @test node_feature(fg) == nf
        @test edge_feature(fg) == ef
        @test global_feature(fg) == gf
        @test positional_feature(fg) == pf
        @test GraphSignals.nf_dims_repr(fg) == vdim
        @test GraphSignals.ef_dims_repr(fg) == edim
        @test GraphSignals.gf_dims_repr(fg) == gdim
        @test GraphSignals.pf_dims_repr(fg) == pdim
        @test parent(fg) === fg

        fg2 = FeaturedGraph(fg)
        @test GraphSignals.adjacency_matrix(fg2) == adjm
        @test node_feature(fg2) == nf
        @test edge_feature(fg2) == ef
        @test global_feature(fg2) == gf

        new_nf = rand(vdim, V)
        new_fg = ConcreteFeaturedGraph(fg, nf=new_nf)
        @test node_feature(new_fg) == new_nf
        @test edge_feature(new_fg) == ef
        @test global_feature(new_fg) == gf
        @test positional_feature(new_fg) == pf
    
        # Test with transposed features
        nf_t = rand(V, vdim)'
        fg = FeaturedGraph(adjm; nf=nf_t)
        @test has_graph(fg)
        @test has_node_feature(fg)
        @test !has_edge_feature(fg)
        @test !has_global_feature(fg)
        @test !has_positional_feature(fg)
        @test graph(fg).S == adjm
        @test node_feature(fg) == nf_t
        @test edge_feature(fg) == Fill(0., (0, E))
        @test global_feature(fg) == zeros(0)
        @test positional_feature(fg) == Fill(0., (0, V))
    end

    @testset "setting properties" begin
        adjm1 = [0 1 1 1;
                 1 0 1 0;
                 1 1 0 1;
                 1 0 1 0]
        adjm2 = [0 1 0 1;
                 1 0 1 1;
                 0 1 0 1;
                 1 1 1 0]
        fg = FeaturedGraph(adjm1; nf=nf, ef=ef, gf=gf, pf=pf)
        fg.graph.S .= adjm2
        @test fg.graph.S == adjm2
        @test_throws DimensionMismatch fg.graph = [0 1; 0 1]

        nf_ = rand(10, V)
        fg.nf = nf_
        @test node_feature(fg) == nf_
        @test_throws DimensionMismatch fg.nf = rand(10, 11)

        ef_ = rand(10, E)
        fg.ef = ef_
        @test edge_feature(fg) == ef_
        @test_throws DimensionMismatch fg.ef = rand(10, 11)

        pf_ = rand(10, V)
        fg.pf = pf_
        @test positional_feature(fg) == pf_
        @test_throws DimensionMismatch fg.pf = rand(10, 11)
    end

    @testset "graph properties" begin
        adjm = [0 1 1 1;
                1 0 1 0;
                1 1 0 1;
                1 0 1 0]

        fg = FeaturedGraph(adjm)
        @test vertices(fg) == 1:V
        @test edges(fg) isa GraphSignals.EdgeIter
        @test neighbors(fg) == [2, 3, 4, 1, 3, 1, 2, 4, 1, 3]
        @test incident_edges(fg) == fg |> graph |> GraphSignals.edgevals

        el = GraphSignals.to_namedtuple(fg)
        @test el.N == V
        @test el.E == E
        @test el.es == [1, 2, 4, 1, 3, 2, 3, 5, 4, 5]
        @test el.nbrs == [2, 3, 4, 1, 3, 1, 2, 4, 1, 3]
        @test el.xs == [1, 1, 1, 2, 2, 3, 3, 3, 4, 4]
    end

    @testset "generate coordinates" begin
        adjm = [0 1 1 1;
                1 0 1 0;
                1 1 0 1;
                1 0 1 0]

        fg = FeaturedGraph(adjm; nf=nf, pf=:auto)
        @test size(positional_feature(fg)) == (1, V)
    end

    @testset "random subgraph" begin
        adjm = [0 1 1 1;
                1 0 1 0;
                1 1 0 1;
                1 0 1 0]

        fg = FeaturedGraph(adjm)
        rand_subgraph = sample(fg, 3)
        @test rand_subgraph isa FeaturedSubgraph
        @test length(rand_subgraph.nodes) == 3
        @test_throws ErrorException sample(fg, 5)
    end
end
