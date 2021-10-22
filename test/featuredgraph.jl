@testset "featuredgraph" begin
    V = 4
    E = 5
    vdim = 3
    edim = 5
    gdim = 7

    nf = rand(vdim, V)
    ef = rand(edim, E)
    gf = rand(gdim)

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
        @test GraphSignals.nf_dims_repr(fg) == vdim
        @test GraphSignals.ef_dims_repr(fg) == edim
        @test GraphSignals.gf_dims_repr(fg) == gdim

        fg2 = FeaturedGraph(fg)
        @test Graphs.adjacency_matrix(fg2) == adjm
        @test node_feature(fg2) == nf
        @test edge_feature(fg2) == ef
        @test global_feature(fg2) == gf
    
        # Test with transposed features
        nf_t = rand(V, vdim)'
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
        @test_throws DimensionMismatch FeaturedGraph(rand(10,11); nf=nf, ef=ef, gf=gf)
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

        nf_ = rand(10, V)
        fg.nf = nf_
        @test fg.nf == nf_
        @test_throws DimensionMismatch fg.nf = rand(10, 11)

        ef_ = rand(10, E)
        fg.ef = ef_
        @test fg.ef == ef_
        @test_throws DimensionMismatch fg.ef = rand(10, 11)
    end

    @testset "scatter" begin
        adjm = [0 1 1 1;
                1 0 1 0;
                1 1 0 1;
                1 0 1 0]

        fg = FeaturedGraph(adjm; nf=nf, ef=ef, gf=gf)
        @test size(edge_scatter(fg, +, direction=:undirected)) == (edim, V)
        @test size(neighbor_scatter(fg, +, direction=:undirected)) == (vdim, V)
    end
end
