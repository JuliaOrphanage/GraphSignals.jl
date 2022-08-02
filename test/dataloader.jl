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

    y = rand(30, obs_size)

    adjm = [0 1 1 1;
            1 0 1 0;
            1 1 0 1;
            1 0 1 0]

    @testset "numobs, getobs" begin
        fg = FeaturedGraph(adjm; nf=nf, ef=ef, gf=gf, pf=pf)
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

        fg = FeaturedGraph(adjm; nf=nf, ef=ef)
        @test numobs(fg) == obs_size
        @test getobs(fg) == fg

        for idx in (2, 2:5, [1, 3, 5])
            idxed_fg = getobs(fg, idx)
            @test graph(idxed_fg) == graph(fg)
            @test node_feature(idxed_fg) == node_feature(fg)[:, :, idx]
            @test edge_feature(idxed_fg) == edge_feature(fg)[:, :, idx]
            @test global_feature(idxed_fg) == global_feature(fg)
            @test positional_feature(idxed_fg) == positional_feature(fg)
        end

        fg = subgraph(FeaturedGraph(adjm; nf=nf, ef=ef), [1, 3, 4, 5])
        @test numobs(fg) == obs_size
        @test getobs(fg) == fg

        for idx in (2, 2:5, [1, 3, 5])
            idxed_fg = getobs(fg, idx)
            @test graph(idxed_fg) == graph(fg)
            @test node_feature(idxed_fg) == node_feature(fg)[:, :, idx]
            @test edge_feature(idxed_fg) == edge_feature(fg)[:, :, idx]
            @test global_feature(idxed_fg) == global_feature(fg)
            @test positional_feature(idxed_fg) == positional_feature(fg)
        end
    end

    @testset "shuffleobs" begin
        fg = FeaturedGraph(adjm; nf=nf, ef=ef, gf=gf, pf=pf)
        shuffled_obs = shuffleobs(fg)
        @test shuffled_obs isa MLUtils.ObsView
        @test shuffled_obs.data isa FeaturedGraph
    end

    @testset "splitobs" begin
        fg = FeaturedGraph(adjm; nf=nf, ef=ef, gf=gf, pf=pf)
        train, test = splitobs(fg, at=0.7)
        @test train isa MLUtils.ObsView
        @test test isa MLUtils.ObsView
        @test length(train) == 0.7 * obs_size
        @test length(test) == 0.3 * obs_size
    end

    @testset "DataLoader" begin
        fg = FeaturedGraph(adjm; nf=nf, ef=ef, gf=gf, pf=pf)
        loader = DataLoader((fg, y), batchsize = batch_size)
        @test length(loader) == obs_size ÷ batch_size

        obs, next = iterate(loader)
        batched_x, batched_y = obs
        @test batched_x isa FeaturedGraph
        @test node_feature(batched_x) == node_feature(fg)[:, :, 1:batch_size]
        @test edge_feature(batched_x) == edge_feature(fg)[:, :, 1:batch_size]
        @test global_feature(batched_x) == global_feature(fg)[:, 1:batch_size]
        @test positional_feature(batched_x) == positional_feature(fg)[:, :, 1:batch_size]
        @test batched_y == y[:, 1:batch_size]

        obs, next = iterate(loader, next)
        batched_x, batched_y = obs
        @test batched_x isa FeaturedGraph
        @test node_feature(batched_x) == node_feature(fg)[:, :, (batch_size+1):2batch_size]
        @test edge_feature(batched_x) == edge_feature(fg)[:, :, (batch_size+1):2batch_size]
        @test global_feature(batched_x) == global_feature(fg)[:, (batch_size+1):2batch_size]
        @test positional_feature(batched_x) == positional_feature(fg)[:, :, (batch_size+1):2batch_size]
        @test batched_y == y[:, (batch_size+1):2batch_size]
    end
end
