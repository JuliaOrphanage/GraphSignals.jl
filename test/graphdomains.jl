@testset "graphdomains" begin
    T = Float32

    d = GraphSignals.NullDomain()
    @test isnothing(GraphSignals.domain(d))
    @test isnothing(positional_feature(d))
    @test !has_positional_feature(d)
    @test GraphSignals.pf_dims_repr(d) == 0
    GraphSignals.check_num_nodes(0, d)

    pf = rand(T, 2, 3, 4)
    d = GraphSignals.NodeDomain(pf)
    @test GraphSignals.domain(d) == pf
    @test positional_feature(d) == pf
    @test has_positional_feature(d)
    @test GraphSignals.pf_dims_repr(d) == 2
    GraphSignals.check_num_nodes(3, d)
end
