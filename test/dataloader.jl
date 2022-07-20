@testset "dataloader" begin
    vdim = 3
    edim = 5
    gdim = 7
    pdim = 11

    V = 4
    E = 5

    obs_size = 100
    batch_size = 10

    nf = rand(vdim, V, obs_size)
    ef = rand(edim, E, obs_size)
    gf = rand(gdim, obs_size)
    pf = rand(pdim, V, obs_size)

    adjm = [0 1 1 1;
            1 0 1 0;
            1 1 0 1;
            1 0 1 0]
    fg = FeaturedGraph(adjm; nf=nf, ef=ef ,gf=gf, pf=pf)

    @testset "shuffleobs" begin
        
    end

    @testset "splitobs" begin

    end

    @testset "DataLoader" begin
        @test numobs(fg) == obs_size
        @test getobs(fg) == fg

        for idx in (2, 2:5, [1, 3, 5])
            idxed_fg = getobs(fg, idx)
            @test graph(idxed_fg) == graph(fg)
            @test node_feature(idxed_fg) == node_feature(fg)[:, :, idx]
            @test edge_feature(idxed_fg) == edge_feature(fg)[:, :, idx]
            @test global_feature(idxed_fg) == global_feature(fg)[:, idx]
            @test positional_feature(idxed_fg) == positional_feature(fg)[:, :, idx]
        end
    end
end
