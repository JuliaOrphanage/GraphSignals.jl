@testset "linalg" begin
    in_channel = 3
    out_channel = 5
    N = 6

    adjs = Dict(
        :simple => [0. 1. 1. 0. 0. 0.;
                    1. 0. 1. 0. 1. 0.;
                    1. 1. 0. 1. 0. 1.;
                    0. 0. 1. 0. 0. 0.;
                    0. 1. 0. 0. 0. 0.;
                    0. 0. 1. 0. 0. 0.],
        :weight => [0. 2. 2. 0. 0. 0.;
                    2. 0. 1. 0. 2. 0.;
                    2. 1. 0. 5. 0. 2.;
                    0. 0. 5. 0. 0. 0.;
                    0. 2. 0. 0. 0. 0.;
                    0. 0. 2. 0. 0. 0.],
    )

    degs = Dict(
        :simple => [2. 0. 0. 0. 0. 0.;
                    0. 3. 0. 0. 0. 0.;
                    0. 0. 4. 0. 0. 0.;
                    0. 0. 0. 1. 0. 0.;
                    0. 0. 0. 0. 1. 0.;
                    0. 0. 0. 0. 0. 1.],
        :weight => [4. 0. 0. 0. 0. 0.;
                    0. 5. 0. 0. 0. 0.;
                    0. 0. 10. 0. 0. 0.;
                    0. 0. 0. 5. 0. 0.;
                    0. 0. 0. 0. 2. 0.;
                    0. 0. 0. 0. 0. 2.]
    )

    laps = Dict(
        :simple => [2. -1. -1. 0. 0. 0.;
                    -1. 3. -1. 0. -1. 0.;
                    -1. -1. 4. -1. 0. -1.;
                    0. 0. -1. 1. 0. 0.;
                    0. -1. 0. 0. 1. 0.;
                    0. 0. -1. 0. 0. 1.],
        :weight => [4. -2. -2. 0. 0. 0.;
                    -2. 5. -1. 0. -2. 0.;
                    -2. -1. 10. -5. 0. -2.;
                    0. 0. -5. 5. 0. 0.;
                    0. -2. 0. 0. 2. 0.;
                    0. 0. -2. 0. 0. 2.],
    )

    norm_laps = Dict(
        :simple => [1. -1/sqrt(2*3) -1/sqrt(2*4) 0. 0. 0.;
                    -1/sqrt(2*3) 1. -1/sqrt(3*4) 0. -1/sqrt(3) 0.;
                    -1/sqrt(2*4) -1/sqrt(3*4) 1. -1/2 0. -1/2;
                    0. 0. -1/2 1. 0. 0.;
                    0. -1/sqrt(3) 0. 0. 1. 0.;
                    0. 0. -1/2 0. 0. 1.],
        :weight => [1. -2/sqrt(4*5) -2/sqrt(4*10) 0. 0. 0.;
                    -2/sqrt(4*5) 1. -1/sqrt(5*10) 0. -2/sqrt(2*5) 0.;
                    -2/sqrt(4*10) -1/sqrt(5*10) 1. -5/sqrt(5*10) 0. -2/sqrt(2*10);
                    0. 0. -5/sqrt(5*10) 1. 0. 0.;
                    0. -2/sqrt(2*5) 0. 0. 1. 0.;
                    0. 0. -2/sqrt(2*10) 0. 0. 1.]
    )

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
        
        ugs = Dict(
            :simple => SimpleGraph(N),
            :weight => SimpleWeightedGraph(N),
        )

        add_edge!(ugs[:simple], 1, 2); add_edge!(ugs[:simple], 1, 3);
        add_edge!(ugs[:simple], 2, 3); add_edge!(ugs[:simple], 3, 4)
        add_edge!(ugs[:simple], 2, 5); add_edge!(ugs[:simple], 3, 6)

        add_edge!(ugs[:weight], 1, 2, 2); add_edge!(ugs[:weight], 1, 3, 2)
        add_edge!(ugs[:weight], 2, 3, 1); add_edge!(ugs[:weight], 3, 4, 5)
        add_edge!(ugs[:weight], 2, 5, 2); add_edge!(ugs[:weight], 3, 6, 2)
        
        fg = FeaturedGraph(adjm)
        @test matrixtype(fg) == :adjm
        @test repr(fg) == "FeaturedGraph(\n\tUndirected graph with (#V=4, #E=4) in adjacency matrix,\n)"
        @test GraphSignals.adjacency_matrix(adjm, Int64) === adjm

        for T in [Int8, Int16, Int32, Int64, Int128, Float16, Float32, Float64]
            for g in [adjm, sparse(adjm), fg]
                @test GraphSignals.degrees(g; dir=:both) == [2, 2, 2, 2]

                D = GraphSignals.degree_matrix(g, T, dir=:out)
                @test D == T.(deg)
                @test GraphSignals.degree_matrix(g, T; dir=:in) == D
                @test GraphSignals.degree_matrix(g, T; dir=:both) == D
                @test eltype(D) == T

                @test GraphSignals.adjacency_matrix(g) == adjm

                L = Graphs.laplacian_matrix(g, T)
                @test L == T.(lap)
                @test eltype(L) == T
            end

            for kind in [:simple, :weight]
                @test GraphSignals.degrees(ugs[kind], T, dir=:out) == T.(diag(degs[kind]))

                D = GraphSignals.degree_matrix(ugs[kind], T, dir=:out)
                @test D == T.(degs[kind])
                @test GraphSignals.degree_matrix(ugs[kind], T; dir=:in) == D
                @test GraphSignals.degree_matrix(ugs[kind], T; dir=:both) == D
                @test eltype(D) == T

                @test Graphs.laplacian_matrix(ugs[kind], T) == T.(laps[kind])
            end

            fg_ = laplacian_matrix!(deepcopy(fg), T)
            @test graph(fg_).S == T.(lap)
            @test matrixtype(fg_) == :laplacian
            @test repr(fg_) == "FeaturedGraph(\n\tUndirected graph with (#V=4, #E=4) in Laplacian matrix,\n)"

            for g in [adjm, sparse(adjm)]
                SL = GraphSignals.signless_laplacian(g, T)
                @test SL == T.(adjm + deg)
                @test eltype(SL) == T
            end
        end

        fg = FeaturedGraph(Float64.(adjm))
        @test matrixtype(fg) == :adjm
        for T in [Float16, Float32, Float64]
            for g in [adjm, sparse(adjm), fg]
                NA = GraphSignals.normalized_adjacency_matrix(g, T)
                @test NA ≈ T.(I - norm_lap)
                @test eltype(NA) == T

                NA = GraphSignals.normalized_adjacency_matrix(g, T, selfloop=true)
                @test eltype(NA) == T

                NL = GraphSignals.normalized_laplacian(g, T)
                @test NL ≈ T.(norm_lap)
                @test eltype(NL) == T

                SL = GraphSignals.scaled_laplacian(g, T)
                @test SL ≈ T.(scaled_lap)
                @test eltype(SL) == T
            end

            for kind in [:simple, :weight]
                NA = GraphSignals.normalized_adjacency_matrix(ugs[kind], T)
                @test NA ≈ T.(I - norm_laps[kind])
                @test eltype(NA) == T

                NL = normalized_laplacian(ugs[kind], T)
                @test NL ≈ T.(norm_laps[kind])
                @test eltype(NL) == T
            end

            fg_ = normalized_laplacian!(deepcopy(fg), T)
            @test graph(fg_).S ≈ T.(norm_lap)
            @test matrixtype(fg_) == :normalized
            @test repr(fg_) == "FeaturedGraph(\n\tUndirected graph with (#V=4, #E=4) in normalized Laplacian,\n)"

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

        dgs = Dict(
            :simple => SimpleDiGraph(N),
            :weight => SimpleWeightedDiGraph(N),
        )
    
        add_edge!(dgs[:simple], 1, 3); add_edge!(dgs[:simple], 2, 3)
        add_edge!(dgs[:simple], 1, 6); add_edge!(dgs[:simple], 2, 5)
        add_edge!(dgs[:simple], 3, 4); add_edge!(dgs[:simple], 3, 5)
    
        add_edge!(dgs[:weight], 1, 3, 2); add_edge!(dgs[:weight], 2, 3, 2)
        add_edge!(dgs[:weight], 1, 6, 1); add_edge!(dgs[:weight], 2, 5, -2)
        add_edge!(dgs[:weight], 3, 4, -2); add_edge!(dgs[:weight], 3, 5, -1)

        for T in [Int8, Int16, Int32, Int64, Int128, Float16, Float32, Float64]
            for g in [adjm, sparse(adjm)]
                for dir in [:out, :in, :both]
                    D = GraphSignals.degree_matrix(g, T, dir=dir)
                    @test D == T.(degs[dir])
                    @test eltype(D) == T

                    L = Graphs.laplacian_matrix(g, T, dir=dir)
                    @test L == T.(laps[dir])
                    @test eltype(L) == T

                    SL = GraphSignals.signless_laplacian(g, T, dir=dir)
                    @test SL == T.(sig_laps[dir])
                    @test eltype(SL) == T
                end
                @test_throws DomainError GraphSignals.degree_matrix(g, dir=:other)
            end
        end

        for T in [Float32, Float64]
            for g in [adjm, sparse(adjm)]
                for dir in [:out, :in]
                    L = normalized_laplacian(g, T, dir=dir)
                    @test L == T.(norm_laps[dir])
                    @test eltype(L) == T
                end

                for dir in [:out, :in, :both]
                    RW = GraphSignals.random_walk_laplacian(g, T, dir=dir)
                    @test RW == T.(rw_laps[dir])
                    @test eltype(RW) == T
                end
            end
        end
    end
end
