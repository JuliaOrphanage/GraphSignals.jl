adj = [0 1 0 1;
       1 0 1 0;
       0 1 0 1;
       1 0 1 0]
deg = [2 0 0 0;
       0 2 0 0;
       0 0 2 0;
       0 0 0 2]
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

    for T in [Int8, Int16, Int32, Int64, Int128]
        dm = GraphSignals.degree_matrix(fg, T; dir=:out)
        @test dm == T.(deg)
        @test GraphSignals.degree_matrix(adj, T; dir=:in) == dm
        @test GraphSignals.degree_matrix(adj, T; dir=:both) == dm
        @test GraphSignals.laplacian_matrix(fg, T) == T.(lap)
    end
    for T in [Float16, Float32, Float64]
        dm = GraphSignals.degree_matrix(fg, T; dir=:out)
        @test dm == T.(deg)
        @test GraphSignals.degree_matrix(adj, T; dir=:in) == dm
        @test GraphSignals.degree_matrix(adj, T; dir=:both) == dm
        @test GraphSignals.laplacian_matrix(fg, T) == T.(lap)
        @test GraphSignals.normalized_laplacian(fg, T) ≈ T.(norm_lap)
        @test GraphSignals.scaled_laplacian(fg, T) ≈ T.(scaled_lap)
    end
end
