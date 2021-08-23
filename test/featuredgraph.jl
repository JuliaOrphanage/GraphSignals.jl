@testset "featuredgraph" begin
    N = 4
    E = 5

    nf = rand(3, N)
    ef = rand(5, E)
    gf = rand(7)

    @testset "null graph" begin
        ng = NullGraph()
        @test FeaturedGraph() == ng
        @test FeaturedGraph(ng) == ng
        @test !has_graph(ng)
        @test !has_node_feature(ng)
        @test !has_edge_feature(ng)
        @test !has_global_feature(ng)
        @test isnothing(graph(ng))
        @test isnothing(node_feature(ng))
        @test isnothing(edge_feature(ng))
        @test isnothing(global_feature(ng))
    end

    @testset "features" begin
        adjm = [0 1 1 1;
                1 0 1 0;
                1 1 0 1;
                1 0 1 0]

        fg = FeaturedGraph(adjm; nf=nf, ef=ef ,gf=gf)
        @test has_graph(fg)
        @test has_node_feature(fg)
        @test has_edge_feature(fg)
        @test has_global_feature(fg)
        @test graph(fg).S == adjm
        @test node_feature(fg) == nf
        @test edge_feature(fg) == ef
        @test global_feature(fg) == gf
        @test GraphSignals.nf_dims_repr(fg) == 3
        @test GraphSignals.ef_dims_repr(fg) == 5
        @test GraphSignals.gf_dims_repr(fg) == 7
    
        # Test with transposed features
        nf_t = rand(N, 3)'
        fg = FeaturedGraph(adjm; nf=nf_t)
        @test has_graph(fg)
        @test has_node_feature(fg)
        @test !has_edge_feature(fg)
        @test !has_global_feature(fg)
        @test graph(fg).S == adjm
        @test node_feature(fg) == nf_t
        @test edge_feature(fg) == Fill(0., (0, E))
        @test global_feature(fg) == zeros(0)

        T = Matrix{Float32}
        fg = FeaturedGraph{T,T,T,Vector{Float32}}(adjm, nf, ef, gf, :adjm)
        @test node_feature(fg) isa T
        @test edge_feature(fg) isa T
        @test global_feature(fg) isa Vector{Float32}
    end

    @testset "constructor" begin
        adjm = [0 1 1 1;
                1 0 1 0;
                1 1 0 1;
                1 0 1 0]
        @test_throws AssertionError FeaturedGraph(rand(10,11); nf=nf, ef=ef, gf=gf)
        @test_throws DimensionMismatch FeaturedGraph(adjm; nf=rand(10,11), ef=ef, gf=gf)
        @test_throws DimensionMismatch FeaturedGraph(adjm; nf=nf, ef=rand(10,11), gf=gf)
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
        fg = FeaturedGraph(adjm1; nf=nf, ef=ef, gf=gf)
        fg.graph.S .= adjm2
        @test fg.graph.S == adjm2
        @test_throws DimensionMismatch fg.graph = [0 1; 0 1]

        nf_ = rand(10, N)
        fg.nf = nf_
        @test fg.nf == nf_
        @test_throws DimensionMismatch fg.nf = rand(10, 11)

        ef_ = rand(10, E)
        fg.ef = ef_
        @test fg.ef == ef_
        @test_throws DimensionMismatch fg.ef = rand(10, 11)
    end
end
