@testset "mask" begin
    adj1 = [0 1 0 1; # symmetric
            1 0 1 0;
            0 1 0 1;
            1 0 1 0]
    adj2 = [1 1 0 1 0; # asymmetric
            1 1 1 0 0;
            0 1 1 1 1;
            1 0 1 1 0;
            1 0 1 0 1]
    mask1 = [2, 3, 4]
    mask2 = [1, 2, 3, 4]

    fg1 = FeaturedGraph(adj1)
    fg2 = FeaturedGraph(adj2)
    gm1 = mask(fg1, mask1)
    gm2 = mask(fg2, mask2)

    @test subgraph(fg1, mask1) == gm1
    @test subgraph(fg2, mask2) == gm2
    @test adjacency_matrix(gm1) == [0 1 0;
                                    1 0 1;
                                    0 1 0]
    @test adjacency_matrix(gm2) == [1 1 0 1;
                                    1 1 1 0;
                                    0 1 1 1;
                                    1 0 1 1]
end
