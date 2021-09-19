@testset "cuda" begin
    T = Float32

    @testset "CuSparseMatrixCSC" begin
        adjm = cu(sparse(
            T[0 1 0 1;
              1 0 1 0;
              0 1 0 1;
              1 0 1 0]))
        @test collect(rowvals(adjm, 2)) == [1, 3]
        @test collect(nonzeros(adjm, 2)) == [1, 1]
    end

    @testset "SparseGraph" begin
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
            @test collect(GraphSignals.aggregate_index(sg, :edge, :inward)) == [1, 3, 1, 1, 4]
            @test collect(GraphSignals.aggregate_index(sg, :edge, :outward)) == [2, 3, 4, 5, 5]
            @test_throws ArgumentError GraphSignals.aggregate_index(sg, :edge, :in)
            @test size(edge_scatter(+, ef, sg)) == (10, V)
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
            @test Array(GraphSignals.aggregate_index(sg, :edge, :inward)) == [2, 5, 1, 4, 4, 1, 4]
            @test Array(GraphSignals.aggregate_index(sg, :edge, :outward)) == [1, 1, 3, 3, 4, 5, 5]
            @test size(edge_scatter(+, ef, sg, direction=:inward)) == (10, V)
            @test size(edge_scatter(+, ef, sg, direction=:outward)) == (10, V)
        end
    end

    @testset "featuredgraph" begin
        @testset "undirected graph" begin
            # undirected graph with self loop
            V = 5
            E = 5
            nf = rand(10, V)
    
            adjm = T[0 1 0 1 1;
                    1 0 0 0 0;
                    0 0 1 0 0;
                    1 0 0 0 1;
                    1 0 0 1 0]
    
            fg = FeaturedGraph(adjm; directed=:undirected, nf=nf) |> gpu
            @test has_graph(fg)
            @test has_node_feature(fg)
            @test !has_edge_feature(fg)
            @test !has_global_feature(fg)
            @test graph(fg) isa SparseGraph
            @test node_feature(fg) isa CuMatrix{T}
            @test edge_feature(fg) isa Fill{T,2,Tuple{Base.OneTo{Int64},Base.OneTo{Int64}}}
            @test global_feature(fg) isa Fill{T,1,Tuple{Base.OneTo{Int64}}}
        end

        @testset "directed graph" begin
            # directed graph with self loop
            V = 5
            E = 7
            nf = rand(10, V)
    
            adjm = T[0 0 1 0 1;
                    1 0 0 0 0;
                    0 0 0 0 0;
                    0 0 1 1 1;
                    1 0 0 0 0]
    
            fg = FeaturedGraph(adjm; directed=:directed, nf=nf) |> gpu
            @test has_graph(fg)
            @test has_node_feature(fg)
            @test !has_edge_feature(fg)
            @test !has_global_feature(fg)
            @test graph(fg) isa SparseGraph
            @test node_feature(fg) isa CuMatrix{T}
            @test edge_feature(fg) isa Fill{T,2,Tuple{Base.OneTo{Int64},Base.OneTo{Int64}}}
            @test global_feature(fg) isa Fill{T,1,Tuple{Base.OneTo{Int64}}}
        end
    end
end
