@testset "tokenizer" begin
    V, E = 4, 5
    vdim, edim = 3, 5
    batch_size = 10

    nf = rand(vdim, V, batch_size)
    ef = rand(edim, E, batch_size)

    adjm = [0 1 1 1;
            1 0 1 0;
            1 1 0 1;
            1 0 1 0]

    orthonormal = repeat(Matrix{Float64}(I(V)), outer=(1, 1, batch_size))

    node_id = node_identifier(adjm, batch_size; method=GraphSignals.orthogonal_random_features)
    @test NNlib.batched_mul(NNlib.batched_transpose(node_id), node_id) ≈ orthonormal

    node_id = node_identifier(adjm, batch_size; method=GraphSignals.laplacian_matrix)
    @test NNlib.batched_mul(NNlib.batched_transpose(node_id), node_id) ≈ orthonormal

    node_token, edge_token = tokenize(adjm, nf, ef)
    @test size(node_token) == (vdim + 2V, V, batch_size)
    @test size(edge_token) == (edim + 2V, 2E, batch_size)
end
