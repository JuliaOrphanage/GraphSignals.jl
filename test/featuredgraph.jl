N = 4
E = 5
adj = [0 1 1 1;
       1 0 1 0;
       1 1 0 1;
       1 0 1 0]
adj2 = [0 1 0 1;
        1 0 1 1;
        0 1 0 1;
        1 1 1 0]

ug = SimpleGraph(N)
add_edge!(ug, 1, 2); add_edge!(ug, 1, 3); add_edge!(ug, 1, 4)
add_edge!(ug, 2, 3); add_edge!(ug, 3, 4)

nf = rand(3, N)
ef = rand(5, E)
gf = rand(7)


@testset "featuredgraph" begin
    ng = NullGraph()
    @test FeaturedGraph() == ng
    @test has_graph(ng) == false
    @test has_node_feature(ng) == false
    @test has_edge_feature(ng) == false
    @test has_global_feature(ng) == false
    @test isnothing(graph(ng))
    @test isnothing(node_feature(ng))
    @test isnothing(edge_feature(ng))
    @test isnothing(global_feature(ng))
    @test isnothing(mask(ng))


    fg = FeaturedGraph(adj)
    @test has_graph(fg)
    @test has_node_feature(fg) == false
    @test has_edge_feature(fg) == false
    @test has_global_feature(fg) == false
    @test graph(fg) == adj
    @test node_feature(fg) == Fill(0, (0, N))
    @test edge_feature(fg) == Fill(0, (0, E))
    @test global_feature(fg) == Fill(0, 0)
    @test mask(fg) == zeros(N, N)


    fg = FeaturedGraph(adj; nf=nf)
    @test has_graph(fg)
    @test has_node_feature(fg)
    @test has_edge_feature(fg) == false
    @test has_global_feature(fg) == false
    @test graph(fg) == adj
    @test node_feature(fg) == nf
    @test edge_feature(fg) == Fill(0., (0, E))
    @test global_feature(fg) == zeros(0)
    @test mask(fg) == zeros(N, N)

    # Test with transposed features
    nf_t = rand(N, 3)'
    fg = FeaturedGraph(adj; nf=nf_t)
    @test has_graph(fg)
    @test has_node_feature(fg)
    @test has_edge_feature(fg) == false
    @test has_global_feature(fg) == false
    @test graph(fg) == adj
    @test node_feature(fg) == nf_t
    @test edge_feature(fg) == Fill(0., (0, E))
    @test global_feature(fg) == zeros(0)
    @test mask(fg) == zeros(N, N)


    fg = FeaturedGraph(ug; nf=nf)
    @test has_graph(fg)
    @test has_node_feature(fg)
    @test has_edge_feature(fg) == false
    @test has_global_feature(fg) == false
    @test graph(fg) == ug
    @test node_feature(fg) == nf
    @test edge_feature(fg) == Fill(0., (0, E))
    @test global_feature(fg) == zeros(0)
    @test mask(fg) == zeros(N, N)


    fg = FeaturedGraph(ug; nf=nf, ef=ef ,gf=gf)
    @test has_graph(fg)
    @test has_node_feature(fg)
    @test has_edge_feature(fg)
    @test has_global_feature(fg)
    @test graph(fg) == ug
    @test node_feature(fg) == nf
    @test edge_feature(fg) == ef
    @test global_feature(fg) == gf
    @test mask(fg) == zeros(N, N)


    fg = FeaturedGraph(adj; nf=nf, ef=ef ,gf=gf)
    @test has_graph(fg)
    @test has_node_feature(fg)
    @test has_edge_feature(fg)
    @test has_global_feature(fg)
    @test graph(fg) == adj
    @test node_feature(fg) == nf
    @test edge_feature(fg) == ef
    @test global_feature(fg) == gf
    @test mask(fg) == zeros(N, N)

    T = Matrix{Float32}
    fg = FeaturedGraph{T,T,T,Vector{Float32}}(adj, nf, ef ,gf, zeros(N, N), :adjm, true)
    @test typeof(graph(fg)) == T
    @test typeof(node_feature(fg)) == T
    @test typeof(edge_feature(fg)) == T
    @test typeof(global_feature(fg)) == Vector{Float32}

    # Check number of node and edge features in the constructor.
    @test_throws DimensionMismatch FeaturedGraph(rand(10,11); nf=nf, ef=ef, gf=gf)
    @test_throws DimensionMismatch FeaturedGraph(adj; nf=rand(10,11), ef=ef, gf=gf)
    @test_throws DimensionMismatch FeaturedGraph(adj; nf=nf, ef=rand(10,11), gf=gf)

    fg = FeaturedGraph(adj; nf=nf, ef=ef, gf=gf)
    fg.graph = adj2
    @test fg.graph == adj2
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
