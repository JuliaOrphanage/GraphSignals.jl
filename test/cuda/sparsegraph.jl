T = Float32

@testset "cuda/sparsegraph" begin
    @testset "undirected graph" begin
        # undirected graph with self loop
        V = 5
        E = 5
        ef = cu(rand(10, E))

        adjm = T[0 1 0 1 1;
                1 0 0 0 0;
                0 0 1 0 0;
                1 0 0 0 1;
                1 0 0 1 0]

        adjl = Vector{T}[
            [2, 4, 5],
            [1],
            [3],
            [1, 5],
            [1, 4]
        ]

        sg = SparseGraph(adjm, false) |> gpu
        @test (collect(sg.S) .!= 0) == adjm
        @test sg.S isa CUSPARSE.CuSparseMatrixCSC{T}
        @test collect(sg.edges) == [1, 3, 4, 1, 2, 3, 5, 4, 5]
        @test sg.edges isa CuVector
        @test sg.E == E
        @test nv(sg) == V
        @test ne(sg) == E
        @test collect(neighbors(sg, 1)) == adjl[1]
        @test collect(neighbors(sg, 2)) == adjl[2]
        @test collect(GraphSignals.dsts(sg)) == [1, 3, 1, 1, 4]
        @test collect(GraphSignals.srcs(sg)) == [2, 3, 4, 5, 5]
        @test_throws ArgumentError GraphSignals.aggregate_index(sg, :edge, :in)
        @test random_walk(sg, 1) ⊆ [2, 4, 5]
        @test neighbor_sample(sg, 1) ⊆ [2, 4, 5]
    end

    @testset "directed graph" begin
        # directed graph with self loop
        V = 5
        E = 7
        ef = cu(rand(10, E))

        adjm = T[0 0 1 0 1;
                1 0 0 0 0;
                0 0 0 0 0;
                0 0 1 1 1;
                1 0 0 0 0]

        adjl = Vector{T}[
            [2, 5],
            [],
            [1, 4],
            [4],
            [1, 4],
        ]

        sg = SparseGraph(adjm, true) |> gpu
        @test (collect(sg.S) .!= 0) == adjm
        @test sg.S isa CUSPARSE.CuSparseMatrixCSC{T}
        @test collect(sg.edges) == collect(1:7)
        @test sg.edges isa CuVector
        @test sg.E == E
        @test nv(sg) == V
        @test ne(sg) == E
        @test collect(neighbors(sg, 1)) == adjl[1]
        @test collect(neighbors(sg, 3)) == adjl[3]
        @test Array(GraphSignals.dsts(sg)) == [2, 5, 1, 4, 4, 1, 4]
        @test Array(GraphSignals.srcs(sg)) == [1, 1, 3, 3, 4, 5, 5]
        @test random_walk(sg, 1) ⊆ [2, 5]
        @test neighbor_sample(sg, 1) ⊆ [2, 5]
    end
end
