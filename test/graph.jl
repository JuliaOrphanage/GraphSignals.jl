@testset "graph" begin
    V = 6

    @testset "null graph" begin
        ng = NullGraph()
        @test adjacency_list(ng) == [zeros(0)]
        @test nv(ng) == 0
        @test ne(ng) == 0
    end

    @testset "undirected graph" begin
        E = 7
        adjm = [0 1 0 1 0 1;
                1 0 1 0 0 0;
                0 1 0 1 1 0;
                1 0 1 0 0 0;
                0 0 1 0 1 0;
                1 0 0 0 0 0]

        adjl = [
            [2, 4, 6],
            [1, 3],
            [2, 4, 5],
            [1, 3],
            [3, 5],
            [1],
        ]

        @testset "constructor" begin
            ug = SimpleGraph(V)
            add_edge!(ug, 1, 2); add_edge!(ug, 1, 4); add_edge!(ug, 1, 6)
            add_edge!(ug, 2, 3); add_edge!(ug, 3, 4); add_edge!(ug, 3, 5)
            add_edge!(ug, 5, 5)

            wug = SimpleWeightedGraph(V)
            add_edge!(wug, 1, 2, 2); add_edge!(wug, 1, 4, 2); add_edge!(wug, 1, 6, 1)
            add_edge!(wug, 2, 3, 5); add_edge!(wug, 3, 4, 2); add_edge!(wug, 3, 5, 2)
            add_edge!(wug, 5, 5, 3)
    
            for g in [adjm, adjl, ug, wug]
                fg = FeaturedGraph(g)
                @test adjacency_list(fg) == adjl
                @test nv(fg) == V
                @test ne(fg) == E
                @test !is_directed(fg)
                @test has_edge(fg, 1, 2)
                @test neighbors(fg, 1) == adjl[1]
            end
        end

        @testset "conversions" begin
            for g in [adjm, adjl]
                @test adjacency_list(g) == adjl
                @test nv(g) == V
                @test ne(g) == E
                @test !is_directed(g)
            end
        end
    end

    @testset "directed graph" begin
        E = 6

        adjm = [0 0 0 0 0 0; # asymmetric
                0 0 0 0 0 0;
                1 1 0 0 0 0;
                0 0 1 0 0 0;
                0 1 1 0 0 0;
                1 0 0 0 0 0]

        adjl = Vector{Int64}[
            [3, 6],
            [3, 5],
            [4, 5],
            [],
            [],
            []
        ]

        @testset "constructor" begin
            dg = SimpleDiGraph(V)
            add_edge!(dg, 1, 3); add_edge!(dg, 2, 3); add_edge!(dg, 1, 6)
            add_edge!(dg, 2, 5); add_edge!(dg, 3, 4); add_edge!(dg, 3, 5)
    
            wdg = SimpleWeightedDiGraph(V)
            add_edge!(wdg, 1, 3, 2); add_edge!(wdg, 2, 3, 2); add_edge!(wdg, 1, 6, 1)
            add_edge!(wdg, 2, 5, -2); add_edge!(wdg, 3, 4, -2); add_edge!(wdg, 3, 5, -1)
    
            for g in [adjm, adjl, dg, wdg]
                fg = FeaturedGraph(g)
                @test adjacency_list(fg) == adjl
                @test nv(fg) == V
                @test ne(fg) == E
                @test is_directed(fg)
                @test has_edge(fg, 1, 3)
                @test neighbors(fg, 1) == adjl[1]
            end
        end
    end
end
