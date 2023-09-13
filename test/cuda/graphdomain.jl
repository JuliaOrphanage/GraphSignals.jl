T = Float32

@testset "cuda/graphdomains" begin
    d = GraphSignals.NullDomain() |> gpu
    @test isnothing(GraphSignals.domain(d))
    @test isnothing(positional_feature(d))
    @test !has_positional_feature(d)

    pf = rand(T, 2, 3, 4)
    d = GraphSignals.NodeDomain(pf) |> gpu
    @test collect(GraphSignals.domain(d)) == pf
    @test collect(positional_feature(d)) == pf
    @test has_positional_feature(d)

    V = 5
    nf = rand(10, V)
    pf = rand(10, V)

    adjm = T[0 1 0 1 1;
            1 0 0 0 0;
            0 0 1 0 0;
            1 0 0 0 1;
            1 0 0 1 0]

    fg = FeaturedGraph(adjm; nf=nf, pf=pf) |> gpu
    gs = gradient(x -> sum(positional_feature(FeaturedGraph(x))), fg)[1]
    @test :domain in keys(gs.pf)
    @test gs.pf.domain isa CuArray
end
