@testset "graphsignals" begin
    T = Float32
    dim = 2
    num = 3
    batch_size = 4

    s = GraphSignals.NullGraphSignal()
    @test isnothing(GraphSignals.signal(s))
    @test isnothing(node_feature(s))
    @test isnothing(node_feature(s))
    @test isnothing(node_feature(s))
    @test !has_node_feature(s)
    @test !has_edge_feature(s)
    @test !has_global_feature(s)
    @test GraphSignals.nf_dims_repr(s) == 0
    @test GraphSignals.ef_dims_repr(s) == 0
    @test GraphSignals.gf_dims_repr(s) == 0
    GraphSignals.check_num_nodes(0, s)
    GraphSignals.check_num_edges(0, s)

    nf = rand(T, dim, num, batch_size)
    s = GraphSignals.NodeSignal(nf)
    @test GraphSignals.signal(s) == nf
    @test node_feature(s) == nf
    @test has_node_feature(s)
    @test GraphSignals.nf_dims_repr(s) == dim
    GraphSignals.check_num_nodes(3, s)

    # permutation equivariant

    ef = rand(T, dim, num, batch_size)
    s = GraphSignals.EdgeSignal(ef)
    @test GraphSignals.signal(s) == ef
    @test edge_feature(s) == ef
    @test has_edge_feature(s)
    @test GraphSignals.ef_dims_repr(s) == dim
    GraphSignals.check_num_edges(3, s)

    # permutation equivariant

    gf = rand(T, dim, batch_size)
    s = GraphSignals.GlobalSignal(gf)
    @test GraphSignals.signal(s) == gf
    @test global_feature(s) == gf
    @test has_global_feature(s)
    @test GraphSignals.gf_dims_repr(s) == dim

    # permutation invariant
end
