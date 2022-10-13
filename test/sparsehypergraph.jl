@testset "SparseHyperGraph" begin
    T = Float32

    @testset "undirected hypergraph" begin
        V = 5
        E = 6

        H = T[0 1 1 0 1 1;
              1 1 0 0 0 1;
              0 1 1 0 0 0;
              1 0 1 1 0 1;
              0 0 1 1 1 0]

        hg = SparseHyperGraph{false}(H)
        @test nv(hg) == V
        @test ne(hg) == E
        @test !is_directed(hg)
        @test !is_directed(typeof(hg))
        @test eltype(hg) == T
        @test vertices(hg) == 1:V
        @test collect(edges(hg)) == (1:E, [(2, 4), (1, 2, 3), (1, 3, 4, 5), (4, 5), (1, 5), (1, 2, 4)])
        @test has_vertex(hg, V)
        @test has_edge(hg, 2, 4)
        @test SparseHyperGraph{false}(H) == hg
        @test incidence_matrix(hg) == H
        @test neighbors(hg) == [
            [2, 3, 4, 5],
            [1, 3, 4],
            [1, 2, 4, 5],
            [1, 2, 3, 5],
            [1, 3, 4]
        ]
        @test neighbors(hg, 1) == [2, 3, 4, 5]
        @test isneighbor(hg, 1, 2)
        @test !isneighbor(hg, 1, 2, 5)
        @test has_edge(hg, 1, 2, 3)
        @test degree(hg) == [4, 3, 2, 4, 3]
        @test degree(hg, 1) == 4
        # @test laplacian_matrix(hg) ==

        _, edge_list = collect(edges(hg))
        @test [e for (i, e) in edges(hg)] == edge_list
    end

    @testset "directed hypergraph" begin
        V = 5
        E = 6

        H = T[ 0  1  1  0 -1  1;
               1 -1  0  0  0 -1;
               0  1  1  0  0  0;
              -1  0 -1 -1  0 -1;
               0  0 -1  1  1  0]

        hg = SparseHyperGraph{true}(H)
        @test nv(hg) == V
        @test ne(hg) == E
        @test is_directed(hg)
        @test is_directed(typeof(hg))
        @test eltype(hg) == T
        @test vertices(hg) == 1:V
        @test collect(edges(hg)) == (1:E, [(2, 4), (1, 2, 3), (1, 3, 4, 5), (4, 5), (1, 5), (1, 2, 4)])
        @test has_vertex(hg, V)
        @test has_edge(hg, 1, 2, 4)
        @test SparseHyperGraph{true}(H) == hg
        @test incidence_matrix(hg) == H
        @test outneighbors(hg) == [[2, 4, 5], [4], [2, 4, 5], [2, 5], [1, 4]]
        @test outneighbors(hg, 1) == [2, 4, 5]
        @test inneighbors(hg) == [[3, 5], [1, 3], [1], [1, 2, 3, 5], [1, 3]]
        @test inneighbors(hg, 1) == [3, 5]
        @test has_edge(hg, 1, 3, 4, 5)
        @test degree(hg) == [4, 3, 2, 4, 3]
        @test degree(hg, 1) == 4
        @test outdegree(hg) == [1, 2, 0, 4, 1]
        @test outdegree(hg, 1) == 1
        @test indegree(hg) == [3, 1, 2, 0, 2]
        @test indegree(hg, 1) == 3
    end
end
