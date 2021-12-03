@testset "random" begin
    adjm = [0 1 0 1 1;  # symmetric
            1 0 0 0 0;
            0 0 1 0 0;
            1 0 0 0 1;
            1 0 0 1 0]

    start = 1
    fg = FeaturedGraph(adjm)

    @test random_walk(adjm, start) ⊆ [2, 4, 5]
    @test random_walk(fg, start) ⊆ [2, 4, 5]
    markov_chain = random_walk(fg, start, 5)
    @test length(markov_chain) == 5
    @test Set(markov_chain) ⊆ collect(1:5)

    @test neighbor_sample(adjm, start) ⊆ [2, 4, 5]
    @test neighbor_sample(fg, start) ⊆ [2, 4, 5]
    neighbors = neighbor_sample(fg, start, 5, replace=true)
    @test length(neighbors) == 5
    @test Set(neighbors) ⊆ [2, 4, 5]
end
