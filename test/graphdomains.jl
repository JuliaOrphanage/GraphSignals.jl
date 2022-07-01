@testset "graphdomains" begin
    T = Float32

    d = GraphSignals.NullDomain()
    @test isnothing(GraphSignals.domain(d))
    @test isnothing(positional_feature(d))
    @test !has_positional_feature(d)

    pf = rand(T, 2, 3, 4)
    d = GraphSignals.NodeDomain(pf)
    @test GraphSignals.domain(d) == pf
    @test positional_feature(d) == pf
    @test has_positional_feature(d)

    @testset "permutation equivariant" begin

    end
end
