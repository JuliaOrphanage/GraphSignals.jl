adj1 = [0 1 0 1; # symmetric
        1 0 1 0;
        0 1 0 1;
        1 0 1 0]
adj2 = [1 1 0 1 0; # asymmetric
        1 1 1 0 0;
        0 1 1 1 1;
        1 0 1 1 0;
        1 0 1 0 1]
mask1 = [0 0 0 0; # symmetric
         0 1 1 1;
         0 1 1 1;
         0 1 1 1]
mask2 = [1 1 0 0 0; # asymmetric
         1 1 1 0 0;
         1 1 0 0 0;
         0 0 1 1 0;
         0 0 0 0 0]

@testset "mask" begin
    fg1 = FeaturedGraph(adj1)
    fg2 = FeaturedGraph(adj2)

    @test graph(GraphMask(fg1, mask1)) == [0 0 0 0;
                                           0 0 1 0;
                                           0 1 0 1;
                                           0 0 1 0]
    @test graph(GraphMask(fg2, mask2)) == [1 1 0 0 0;
                                           1 1 1 0 0;
                                           0 1 0 0 0;
                                           0 0 1 1 0;
                                           0 0 0 0 0]
end
