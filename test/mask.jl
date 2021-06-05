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

nf = rand(3, 4)
ef = rand(5, 4)
gf = rand(7)

@testset "mask" begin
    fg1 = FeaturedGraph(adj1, nf=nf, ef=ef, gf=gf)
    fg2 = FeaturedGraph(adj2)
    gm1 = GraphMask(fg1, mask1)
    gm2 = GraphMask(fg2, mask2)

    @test mask(fg1, mask1) == gm1
    @test mask(fg2, mask2) == gm2
    @test graph(gm1) == [0 0 0 0;
                         0 0 1 0;
                         0 1 0 1;
                         0 0 1 0]
    @test graph(gm2) == [1 1 0 0 0;
                         1 1 1 0 0;
                         0 1 0 0 0;
                         0 0 1 1 0;
                         0 0 0 0 0]
    @test node_feature(gm1) == hcat(zeros(3), nf[:,2], nf[:,3], nf[:,4])
    @test edge_feature(gm1) == hcat(zeros(5), zeros(5), ef[:,3], ef[:,4])
    @test global_feature(gm1) == gf
end
