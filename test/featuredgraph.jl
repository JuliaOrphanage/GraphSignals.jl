@testset "featuredgraph" begin
    N = 4
    E1 = 5
    E2 = 10
    adjm1 = [0 1 1 1;
             1 0 1 0;
             1 1 0 1;
             1 0 1 0]
    adjm2 = [0 1 0 1;
             1 0 1 1;
             0 1 0 1;
             1 1 1 0]

    ug = SimpleGraph(N)
    add_edge!(ug, 1, 2); add_edge!(ug, 1, 3); add_edge!(ug, 1, 4)
    add_edge!(ug, 2, 3); add_edge!(ug, 3, 4)

    nf = rand(3, N)
    ef1 = rand(5, E1)
    ef2 = rand(5, E2)
    gf = rand(7)

    ng = NullGraph()
    @test FeaturedGraph() == ng
    @test FeaturedGraph(ng) == ng
    @test has_graph(ng) == false
    @test has_node_feature(ng) == false
    @test has_edge_feature(ng) == false
    @test has_global_feature(ng) == false
    @test isnothing(graph(ng))
    @test isnothing(node_feature(ng))
    @test isnothing(edge_feature(ng))
    @test isnothing(global_feature(ng))


    fg = FeaturedGraph(adjm1)
    @test graph(FeaturedGraph(fg)).S == adjm1
    @test has_graph(fg)
    @test has_node_feature(fg) == false
    @test has_edge_feature(fg) == false
    @test has_global_feature(fg) == false
    @test graph(fg).S == adjm1
    @test node_feature(fg) == Fill(0, (0, N))
    @test edge_feature(fg) == Fill(0, (0, E1))
    @test global_feature(fg) == Fill(0, 0)


    fg = FeaturedGraph(adjm1; nf=nf)
    @test has_graph(fg)
    @test has_node_feature(fg)
    @test has_edge_feature(fg) == false
    @test has_global_feature(fg) == false
    @test graph(fg).S == adjm1
    @test node_feature(fg) == nf
    @test edge_feature(fg) == Fill(0., (0, E1))
    @test global_feature(fg) == zeros(0)

    # Test with transposed features
    nf_t = rand(N, 3)'
    fg = FeaturedGraph(adjm1; nf=nf_t)
    @test has_graph(fg)
    @test has_node_feature(fg)
    @test has_edge_feature(fg) == false
    @test has_global_feature(fg) == false
    @test graph(fg).S == adjm1
    @test node_feature(fg) == nf_t
    @test edge_feature(fg) == Fill(0., (0, E1))
    @test global_feature(fg) == zeros(0)


    fg = FeaturedGraph(ug; nf=nf)
    @test has_graph(fg)
    @test has_node_feature(fg)
    @test has_edge_feature(fg) == false
    @test has_global_feature(fg) == false
    @test graph(fg).S == adjm1
    @test node_feature(fg) == nf
    @test edge_feature(fg) == Fill(0., (0, 5))
    @test global_feature(fg) == zeros(0)


    fg = FeaturedGraph(ug; nf=nf, ef=ef1 ,gf=gf)
    @test has_graph(fg)
    @test has_node_feature(fg)
    @test has_edge_feature(fg)
    @test has_global_feature(fg)
    @test graph(fg).S == adjm1
    @test node_feature(fg) == nf
    @test edge_feature(fg) == ef1
    @test global_feature(fg) == gf


    fg = FeaturedGraph(adjm1; nf=nf, ef=ef1 ,gf=gf)
    @test has_graph(fg)
    @test has_node_feature(fg)
    @test has_edge_feature(fg)
    @test has_global_feature(fg)
    @test graph(fg).S == adjm1
    @test node_feature(fg) == nf
    @test edge_feature(fg) == ef1
    @test global_feature(fg) == gf
    @test GraphSignals.nf_dims_repr(fg) == 3
    @test GraphSignals.ef_dims_repr(fg) == 5
    @test GraphSignals.gf_dims_repr(fg) == 7

    T = Matrix{Float32}
    fg = FeaturedGraph{T,T,T,Vector{Float32}}(adjm1, nf, ef1, gf, :adjm)
    @test typeof(graph(fg)) == T
    @test typeof(node_feature(fg)) == T
    @test typeof(edge_feature(fg)) == T
    @test typeof(global_feature(fg)) == Vector{Float32}

    # Check number of node and edge features in the constructor.
    @test_throws AssertionError FeaturedGraph(rand(10,11); nf=nf, ef=ef1, gf=gf)
    @test_throws DimensionMismatch FeaturedGraph(adjm1; nf=rand(10,11), ef=ef1, gf=gf)
    @test_throws DimensionMismatch FeaturedGraph(adjm1; nf=nf, ef=rand(10,11), gf=gf)

    # Check number of node and edge features before setting properties.
    fg = FeaturedGraph(adjm1; nf=nf, ef=ef1, gf=gf)
    fg.graph.S .= adjm2
    @test fg.graph.S == adjm2
    @test_throws DimensionMismatch fg.graph = [0 1; 0 1]

    nf_ = rand(10, N)
    fg.nf = nf_
    @test fg.nf == nf_
    @test_throws DimensionMismatch fg.nf = rand(10, 11)

    ef_ = rand(10, E1)
    fg.ef = ef_
    @test fg.ef == ef_
    @test_throws DimensionMismatch fg.ef = rand(10, 11)
end
