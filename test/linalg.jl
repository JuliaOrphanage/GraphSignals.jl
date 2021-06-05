adj = [0 1 0 1;
       1 0 1 0;
       0 1 0 1;
       1 0 1 0]
deg = [2 0 0 0;
       0 2 0 0;
       0 0 2 0;
       0 0 0 2]
isd = [√2, √2, √2, √2]
lap = [2 -1 0 -1;
       -1 2 -1 0;
       0 -1 2 -1;
       -1 0 -1 2]
norm_lap = [1. -.5 0. -.5;
           -.5 1. -.5 0.;
           0. -.5 1. -.5;
           -.5 0. -.5 1.]
scaled_lap =   [0 -0.5 0 -0.5;
                -0.5 0 -0.5 -0;
                0 -0.5 0 -0.5;
                -0.5 0 -0.5 0]

@testset "linalg" begin
    fg = FeaturedGraph(adj)
    @test fg.matrix_type == :adjm
    for T in [Int8, Int16, Int32, Int64, Int128]
        @test GraphSignals.adjacency_matrix(adj, T) == T.(adj)
        @test GraphSignals.adjacency_matrix(fg, T) == T.(adj)
        @test GraphSignals.degrees(fg; dir=:both) == [2, 2, 2, 2]
        dm = GraphSignals.degree_matrix(fg, T; dir=:out)
        @test dm == T.(deg)
        @test GraphSignals.degree_matrix(adj, T; dir=:in) == dm
        @test GraphSignals.degree_matrix(adj, T; dir=:both) == dm
        @test GraphSignals.laplacian_matrix(fg, T) == T.(lap)
    end

    fg = FeaturedGraph(Float64.(adj))
    @test fg.matrix_type == :adjm
    for T in [Float16, Float32, Float64]
        dm = GraphSignals.degree_matrix(fg, T; dir=:out)
        @test dm == T.(deg)
        @test GraphSignals.degree_matrix(adj, T; dir=:in) == dm
        @test GraphSignals.degree_matrix(adj, T; dir=:both) == dm
        @test GraphSignals.inv_sqrt_degree_matrix(fg, T) == T.(diagm(1 ./ isd))
        @test GraphSignals.laplacian_matrix(fg, T) == T.(lap)
        @test GraphSignals.normalized_laplacian(fg, T) ≈ T.(norm_lap)
        @test GraphSignals.scaled_laplacian(fg, T) ≈ T.(scaled_lap)

        fg_ = GraphSignals.laplacian_matrix!(deepcopy(fg), T)
        @test fg_.graph == T.(lap)
        @test fg_.matrix_type == :laplacian
        fg_ = GraphSignals.normalized_laplacian!(deepcopy(fg), T)
        @test fg_.graph ≈ T.(norm_lap)
        @test fg_.matrix_type == :normalized
        fg_ = GraphSignals.scaled_laplacian!(deepcopy(fg), T)
        @test fg_.graph ≈ T.(scaled_lap)
        @test fg_.matrix_type == :scaled
    end
end
