@testset "SparseGraph" begin
    @testset "undirected graph" begin
        # undirected graph with self loop
        V = 5
        E = 5

        adjm = [0 1 0 1 1;
                1 0 0 0 0;
                0 0 1 0 0;
                1 0 0 0 1;
                1 0 0 1 0]

        adjl = Vector{Int64}[
            [2, 4, 5],
            [1],
            [3],
            [1, 5],
            [1, 4]
        ]

        @testset "constructor" begin
            ug = SimpleGraph(V)
            add_edge!(ug, 1, 2); add_edge!(ug, 1, 4); add_edge!(ug, 1, 5)
            add_edge!(ug, 3, 3); add_edge!(ug, 4, 5)
        
            wug = SimpleWeightedGraph(V)
            add_edge!(wug, 1, 2, 2); add_edge!(wug, 1, 4, 2); add_edge!(wug, 1, 5, 1)
            add_edge!(wug, 3, 3, 5); add_edge!(wug, 4, 5, 2)

            for g in [adjm, adjl, ug, wug]
                sg = SparseGraph(g, false)
                @test (sg.S .!= 0) == adjm
                @test sg.edges == [1, 3, 4, 1, 2, 3, 5, 4, 5]
                @test sg.E == E
            end
        end

        @testset "conversions" begin
            sg = SparseGraph(adjm, false)
            @test Graphs.adjacency_matrix(sg) == adjm
            @test_skip Array(sg) == adjm
            @test_skip SparseArrays.sprase(sg) == adjm
            @test adjacency_list(sg) == adjl
        end

        sg = SparseGraph(adjm, false)
        @test nv(sg) == V
        @test ne(sg) == E
        @test !Graphs.is_directed(sg)
        @test !Graphs.is_directed(typeof(sg))
        @test repr(sg) == "SparseGraph(#V=5, #E=5)"
        @test Graphs.has_self_loops(sg)

        @test Graphs.has_vertex(sg, 3)
        @test Graphs.vertices(sg) == 1:V
        @test Graphs.edgetype(sg) == typeof((1, 5))
        @test Graphs.has_edge(sg, 1, 5)
        @test sg[1, 5] == 1
        @test sg[CartesianIndex((1, 5))] == 1
        @test edge_index(sg, 1, 5) == 4
        @test collect(edges(sg)) == [
            (1, (2, 1)),
            (2, (3, 3)),
            (3, (4, 1)),
            (4, (5, 1)),
            (5, (5, 4)),
        ]
        @test GraphSignals.edgevals(sg) == sg.edges
        @test GraphSignals.edgevals(sg, 1) == sg.edges[1:3]
        @test GraphSignals.edgevals(sg, 1:2) == sg.edges[1:4]
    
        @test Graphs.neighbors(sg, 1) == adjl[1]
        @test Graphs.neighbors(sg, 3) == adjl[3]
        @test incident_edges(sg, 1) == [1, 3, 4]
        @test incident_edges(sg, 3) == [2]
    
        @test GraphSignals.aggregate_index(sg, :edge, :inward) == [1, 3, 1, 1, 4]
        @test GraphSignals.aggregate_index(sg, :edge, :outward) == [2, 3, 4, 5, 5]
        @test GraphSignals.aggregate_index(sg, :vertex, :inward) == [[2, 4, 5], [1], [3], [1, 5], [1, 4]]
        @test GraphSignals.aggregate_index(sg, :vertex, :outward) == [[2, 4, 5], [1], [3], [1, 5], [1, 4]]
        @test_throws ArgumentError GraphSignals.aggregate_index(sg, :edge, :in)
        @test_throws ArgumentError GraphSignals.aggregate_index(sg, :foo, :inward)

        @testset "scatter" begin
            vdim = 5
            edim = 7
            nf = rand(vdim, V)
            ef = rand(edim, E)

            @test size(edge_scatter(+, ef, sg)) == (edim, V)
            X = neighbor_scatter(+, nf, sg, direction=:undirected)
            @test size(X) == (vdim, V)
            @test X[:,1] == vec(sum(nf[:, [2, 4, 5]], dims=2))

            gradtest(x -> edge_scatter(+, x, sg), ef)
            gradtest(x -> neighbor_scatter(+, x, sg, direction=:undirected), nf)
        end
    end

    @testset "directed graph" begin
        # directed graph with self loop
        V = 5
        E = 7

        adjm = [0 0 1 0 1;
                1 0 0 0 0;
                0 0 0 0 0;
                0 0 1 1 1;
                1 0 0 0 0]

        adjl = Vector{Int64}[
            [2, 5],
            [],
            [1, 4],
            [4],
            [1, 4],
        ]

        @testset "constructor" begin
            dg = SimpleDiGraph(V)
            add_edge!(dg, 1, 2); add_edge!(dg, 1, 5); add_edge!(dg, 3, 1)
            add_edge!(dg, 3, 4); add_edge!(dg, 4, 4); add_edge!(dg, 5, 1)
            add_edge!(dg, 5, 4)
        
            wdg = SimpleWeightedDiGraph(V)
            add_edge!(wdg, 1, 2, 2); add_edge!(wdg, 1, 5, 2); add_edge!(wdg, 3, 1, 1)
            add_edge!(wdg, 3, 4, 5); add_edge!(wdg, 4, 4, 2); add_edge!(wdg, 5, 1, 2)
            add_edge!(wdg, 5, 4, 4)

            for g in [adjm, adjl, dg, wdg]
                sg = SparseGraph(g, true)
                @test (sg.S .!= 0) == adjm
                @test sg.edges == collect(1:7)
                @test sg.E == E
            end
        end

        @testset "conversions" begin
            sg = SparseGraph(adjm, true)
            @test Graphs.adjacency_matrix(sg) == adjm
            @test_skip Array(sg) == adjm
            @test_skip SparseArrays.sprase(sg) == adjm
            @test adjacency_list(sg) == adjl
        end

        sg = SparseGraph(adjm, true)
        @test nv(sg) == V
        @test ne(sg) == E
        @test Graphs.is_directed(sg)
        @test Graphs.is_directed(typeof(sg))
        @test repr(sg) == "SparseGraph(#V=5, #E=7)"
        @test Graphs.has_self_loops(sg)

        @test Graphs.has_vertex(sg, 3)
        @test Graphs.vertices(sg) == 1:V
        @test Graphs.edgetype(sg) == typeof((1, 3))
        @test Graphs.has_edge(sg, 1, 3)
        @test sg[1, 3] == 1
        @test sg[CartesianIndex((1, 3))] == 1
        @test edge_index(sg, 1, 3) == 3
        @test collect(edges(sg)) == [
            (1, (2, 1)),
            (2, (5, 1)),
            (3, (1, 3)),
            (4, (4, 3)),
            (5, (4, 4)),
            (6, (1, 5)),
            (7, (4, 5)),
        ]
        @test GraphSignals.edgevals(sg) == sg.edges
        @test GraphSignals.edgevals(sg, 1) == sg.edges[1:2]
        @test GraphSignals.edgevals(sg, 1:2) == sg.edges[1:2]

        @test Graphs.neighbors(sg, 1) == adjl[1]
        @test Graphs.neighbors(sg, 2) == adjl[2]
        @test_throws ArgumentError Graphs.neighbors(sg, 2, dir=:none)
        @test incident_edges(sg, 1, dir=:out) == [1, 2]
        @test incident_edges(sg, 1, dir=:in) == [3, 6]
        @test sort!(incident_edges(sg, 1, dir=:both)) == [1, 2, 3, 6]
        @test_throws ArgumentError incident_edges(sg, 2, dir=:none)

        @test GraphSignals.aggregate_index(sg, :edge, :inward) == [2, 5, 1, 4, 4, 1, 4]
        @test GraphSignals.aggregate_index(sg, :edge, :outward) == [1, 1, 3, 3, 4, 5, 5]
        @test GraphSignals.aggregate_index(sg, :vertex, :inward) == [[2, 5], [], [1, 4], [4], [1, 4]]
        @test GraphSignals.aggregate_index(sg, :vertex, :outward) == [[3, 5], [1], [], [3, 4, 5], [1]]

        @testset "scatter" begin
            vdim = 5
            edim = 7
            nf = rand(vdim, V)
            ef = rand(edim, E)

            @test size(edge_scatter(+, ef, sg, direction=:inward)) == (edim, V)
            @test size(edge_scatter(+, ef, sg, direction=:outward)) == (edim, V)
            X = neighbor_scatter(+, nf, sg, direction=:inward)
            @test size(X) == (vdim, V)
            @test X[:,1] == vec(sum(nf[:, [2, 5]], dims=2))
            X = neighbor_scatter(+, nf, sg, direction=:outward)
            @test size(X) == (vdim, V)
            @test X[:,1] == vec(sum(nf[:, [3, 5]], dims=2))

            gradtest(x -> edge_scatter(+, x, sg, direction=:inward), ef)
            gradtest(x -> edge_scatter(+, x, sg, direction=:outward), ef)
            gradtest(x -> neighbor_scatter(+, x, sg, direction=:inward), nf)
            gradtest(x -> neighbor_scatter(+, x, sg, direction=:outward), nf)
        end
    end
end