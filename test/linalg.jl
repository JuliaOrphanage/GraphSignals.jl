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
        rw_lap = [1 -.5 0 -.5;
                  -.5 1 -.5 0;
                  0 -.5 1 -.5;
                  -.5 0 -.5 1]
        
        fg = FeaturedGraph(adjm)
        @test matrixtype(fg) == :adjm
        @test repr(fg) == "FeaturedGraph(\n\tUndirected graph with (#V=4, #E=4) in adjacency matrix,\n)"

        for T in [Int8, Int16, Int32, Int64, Int128, Float16, Float32, Float64]
            @test Graphs.degrees(fg; dir=:both) == [2, 2, 2, 2]

            D = GraphSignals.degree_matrix(adjm, T, dir=:out)
            @test D == T.(deg)
            @test GraphSignals.degree_matrix(adjm, T; dir=:in) == D
            @test GraphSignals.degree_matrix(adjm, T; dir=:both) == D
            @test eltype(D) == T
            D = GraphSignals.degree_matrix(sparse(adjm), T, dir=:out)
            @test D == T.(deg)
            @test eltype(D) == T

            @test Graphs.adjacency_matrix(adjm, T) == T.(adjm)
            @test Graphs.adjacency_matrix(fg, T) == T.(adjm)

            L = Graphs.laplacian_matrix(adjm, T)
            @test L == T.(lap)
            @test eltype(L) == T
            L = Graphs.laplacian_matrix(sparse(adjm), T)
            @test L == T.(lap)
            @test eltype(L) == T
            @test laplacian_matrix(fg, T) == T.(lap)

            fg_ = laplacian_matrix!(deepcopy(fg), T)
            @test graph(fg_).S == T.(lap)
            @test matrixtype(fg_) == :laplacian
            @test repr(fg_) == "FeaturedGraph(\n\tUndirected graph with (#V=4, #E=4) in Laplacian matrix,\n)"

            SL = GraphSignals.signless_laplacian(adjm, T)
            @test SL == T.(adjm + deg)
            @test eltype(SL) == T
            SL = GraphSignals.signless_laplacian(sparse(adjm), T)
            @test SL == T.(adjm + deg)
            @test eltype(SL) == T
        end

        fg = FeaturedGraph(Float64.(adjm))
        @test matrixtype(fg) == :adjm
        for T in [Float16, Float32, Float64]
            NL = normalized_laplacian(adjm, T)
            @test NL ≈ T.(norm_lap)
            @test eltype(NL) == T
            NL = normalized_laplacian(sparse(adjm), T)
            @test NL ≈ T.(norm_lap)
            @test eltype(NL) == T

            @test normalized_laplacian(fg, T) ≈ T.(norm_lap)
            fg_ = normalized_laplacian!(deepcopy(fg), T)
            @test graph(fg_).S ≈ T.(norm_lap)
            @test matrixtype(fg_) == :normalized
            @test repr(fg_) == "FeaturedGraph(\n\tUndirected graph with (#V=4, #E=4) in normalized Laplacian,\n)"

            SL = scaled_laplacian(adjm, T)
            @test SL ≈ T.(scaled_lap)
            @test eltype(SL) == T
            SL = scaled_laplacian(sparse(adjm), T)
            @test SL ≈ T.(scaled_lap)
            @test eltype(SL) == T

            @test scaled_laplacian(fg, T) ≈ T.(scaled_lap)
            fg_ = scaled_laplacian!(deepcopy(fg), T)
            @test graph(fg_).S ≈ T.(scaled_lap)
            @test matrixtype(fg_) == :scaled
            @test repr(fg_) == "FeaturedGraph(\n\tUndirected graph with (#V=4, #E=4) in scaled Laplacian,\n)"
            
            RW = GraphSignals.random_walk_laplacian(adjm, T)
            @test RW == T.(rw_lap)
            @test eltype(RW) == T
            RW = GraphSignals.random_walk_laplacian(sparse(adjm), T)
            @test RW == T.(rw_lap)
            @test eltype(RW) == T
        end
    end

    @testset "directed" begin
        adjm = [0 2 0 3;
                0 0 4 0;
                2 0 0 1;
                0 0 0 0]
        degs = Dict(
            :out  => diagm(0=>[2, 2, 4, 4]),
            :in   => diagm(0=>[5, 4, 3, 0]),
            :both => diagm(0=>[7, 6, 7, 4]),
        )
        laps = Dict(
            :out  => degs[:out] - adjm,
            :in   => degs[:in] - adjm,
            :both => degs[:both] - adjm,
        )
        norm_laps = Dict(
            :out  => I - diagm(0=>[1/2, 1/2, 1/4, 1/4])*adjm,
            :in   => I - diagm(0=>[1/5, 1/4, 1/3, 0])*adjm,
        )
        sig_laps = Dict(
            :out  => degs[:out] + adjm,
            :in   => degs[:in] + adjm,
            :both => degs[:both] + adjm,
        )
        rw_laps = Dict(
            :out  => I - diagm(0=>[1/2, 1/2, 1/4, 1/4]) * adjm,
            :in   => I - diagm(0=>[1/5, 1/4, 1/3, 0]) * adjm,
            :both => I - diagm(0=>[1/7, 1/6, 1/7, 1/4]) * adjm,
        )

        for T in [Int8, Int16, Int32, Int64, Int128, Float16, Float32, Float64]
            for dir in [:out, :in, :both]
                D = GraphSignals.degree_matrix(adjm, T, dir=dir)
                @test D == T.(degs[dir])
                @test eltype(D) == T
                D = GraphSignals.degree_matrix(sparse(adjm), T, dir=dir)
                @test D == T.(degs[dir])
                @test eltype(D) == T
            end
            @test_throws DomainError GraphSignals.degree_matrix(adjm, dir=:other)

            for dir in [:out, :in, :both]
                L = Graphs.laplacian_matrix(adjm, T, dir=dir)
                @test L == T.(laps[dir])
                @test eltype(L) == T
                L = Graphs.laplacian_matrix(sparse(adjm), T, dir=dir)
                @test L == T.(laps[dir])
                @test eltype(L) == T
            end

            for dir in [:out, :in, :both]
                SL = GraphSignals.signless_laplacian(adjm, T, dir=dir)
                @test SL == T.(sig_laps[dir])
                @test eltype(SL) == T
                SL = GraphSignals.signless_laplacian(sparse(adjm), T, dir=dir)
                @test SL == T.(sig_laps[dir])
                @test eltype(SL) == T
            end
        end

        for T in [Float32, Float64]
            for dir in [:out, :in]
                L = normalized_laplacian(adjm, T, dir=dir)
                @test L == T.(norm_laps[dir])
                @test eltype(L) == T
            end

            for dir in [:out, :in, :both]
                RW = GraphSignals.random_walk_laplacian(adjm, T, dir=dir)
                @test RW == T.(rw_laps[dir])
                @test eltype(RW) == T
                RW = GraphSignals.random_walk_laplacian(sparse(adjm), T, dir=dir)
                @test RW == T.(rw_laps[dir])
                @test eltype(RW) == T
            end
        end
    end
end
