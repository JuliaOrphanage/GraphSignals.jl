@testset "neighbor_graphs" begin
    T = Float32
    V = 100
    vdim = 10
    k = 5

    @testset "Euclidean distance" begin
        nf = rand(T, vdim, V)
        fg = kneighbors_graph(nf, k; include_self=false)
        @test fg isa FeaturedGraph
        @test nv(fg) == V
        @test ne(fg) == V*k
        @test node_feature(fg) == nf
    end

    @testset "Minkowski distance" begin
        nf = rand(T, vdim, V)
        fg = kneighbors_graph(nf, k, Minkowski(3); include_self=true)
        @test fg isa FeaturedGraph
        @test nv(fg) == V
        @test ne(fg) == V*k
        @test node_feature(fg) == nf
    end

    @testset "Jaccard distance" begin
        nf = rand(T[0, 1], vdim, V)
        fg = kneighbors_graph(nf, k, Jaccard(); include_self=true)
        @test fg isa FeaturedGraph
        @test nv(fg) == V
        @test ne(fg) == V*k
        @test node_feature(fg) == nf
    end
end
