@testset "linalg" begin
    @testset "undirected graph" begin
        adjm = [0 1 0 1;
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
        scaled_lap = [0 -0.5 0 -0.5;
                      -0.5 0 -0.5 -0;
                      0 -0.5 0 -0.5;
                      -0.5 0 -0.5 0]
        
        fg = FeaturedGraph(adjm)
        @test matrixtype(fg) == :adjm
        @test repr(fg) == "FeaturedGraph(\n\tUndirected graph with (#V=4, #E=4) in adjacency matrix,\n)"

        for T in [Int8, Int16, Int32, Int64, Int128, Float16, Float32, Float64]
            @test LightGraphs.adjacency_matrix(adjm, T) == T.(adjm)
            @test LightGraphs.adjacency_matrix(fg, T) == T.(adjm)
            @test LightGraphs.degrees(fg; dir=:both) == [2, 2, 2, 2]
            dm = GraphLaplacians.degree_matrix(fg, T; dir=:out)
            @test dm == T.(deg)
            @test GraphLaplacians.degree_matrix(adjm, T; dir=:in) == dm
            @test GraphLaplacians.degree_matrix(adjm, T; dir=:both) == dm
            @test GraphLaplacians.laplacian_matrix(fg, T) == T.(lap)

            fg_ = laplacian_matrix!(deepcopy(fg), T)
            @test graph(fg_).S == T.(lap)
            @test matrixtype(fg_) == :laplacian
            @test repr(fg_) == "FeaturedGraph(\n\tUndirected graph with (#V=4, #E=4) in Laplacian matrix,\n)"
        end

        fg = FeaturedGraph(Float64.(adjm))
        @test matrixtype(fg) == :adjm
        for T in [Float16, Float32, Float64]
            @test GraphLaplacians.normalized_laplacian(fg, T) ≈ T.(norm_lap)
            @test GraphLaplacians.scaled_laplacian(fg, T) ≈ T.(scaled_lap)
    
            fg_ = normalized_laplacian!(deepcopy(fg), T)
            @test graph(fg_).S ≈ T.(norm_lap)
            @test matrixtype(fg_) == :normalized
            @test repr(fg_) == "FeaturedGraph(\n\tUndirected graph with (#V=4, #E=4) in normalized Laplacian,\n)"

            fg_ = scaled_laplacian!(deepcopy(fg), T)
            @test graph(fg_).S ≈ T.(scaled_lap)
            @test matrixtype(fg_) == :scaled
            @test repr(fg_) == "FeaturedGraph(\n\tUndirected graph with (#V=4, #E=4) in scaled Laplacian,\n)"
        end
    end
end
